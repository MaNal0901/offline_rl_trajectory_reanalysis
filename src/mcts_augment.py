"""
mcts_augment.py — Monte Carlo Tree Search Augmentation
=======================================================
Implémenté par : B
Dépend de      : interfaces.py, data_loader.py, world_model.py

Corrections finales :
    1. calibrate_threshold() centralisé depuis interfaces.py
    2. terminals=0 — rollout tronqué ≠ fin d épisode réelle
    3. double check uncertainty dans _collect_transitions supprimé
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional
import torch

from interfaces import (
    AugmenterInterface,
    ReplayBufferInterface,
    WorldModelInterface,
    DEVICE,
    ROLLOUT_LEN,
    GAMMA,
    DATASET_CONFIGS,
    DEFAULT_DATASET,
    calibrate_threshold,        # ← CORRECTION 1
)


# ─────────────────────────────────────────────────────────────────────────────
# Nœud de l'arbre MCTS
# ─────────────────────────────────────────────────────────────────────────────

class MCTSNode:
    def __init__(
        self,
        state  : torch.Tensor,
        parent : Optional["MCTSNode"] = None,
        action : Optional[torch.Tensor] = None,
        reward : float = 0.0,
        depth  : int   = 0,
    ):
        self.state       = state.cpu()
        self.parent      = parent
        self.action      = action.cpu() if action is not None else None
        self.reward      = reward
        self.children    : List["MCTSNode"] = []
        self.visit_count : int   = 0
        self.value_sum   : float = 0.0
        self.depth       : int   = depth

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def ucb1(self, c_puct: float = 1.41) -> float:
        if self.visit_count == 0:
            return float("inf")
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration   = c_puct * math.sqrt(math.log(parent_visits) / self.visit_count)
        return self.q_value + exploration

    def best_child(self, c_puct: float = 1.41) -> "MCTSNode":
        return max(self.children, key=lambda n: n.ucb1(c_puct))

    def trajectory_to_root(self) -> List["MCTSNode"]:
        path, node = [], self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))


# ─────────────────────────────────────────────────────────────────────────────
# MCTS Augmenter
# ─────────────────────────────────────────────────────────────────────────────

class MCTSAugmenter(AugmenterInterface):
    """
    Augmentation par Monte Carlo Tree Search.

    Pour chaque branch point dans D :
        1. Construit un arbre MCTS depuis root_state
        2. n_simulations itérations : sélection → expansion → simulation → backprop
        3. Collecte les transitions des arcs valides de l arbre
        4. Filtre via calibrate_threshold() — même seuil que Vine et VAE
    """

    def __init__(
        self,
        n_simulations : int   = 20,
        rollout_depth : int   = ROLLOUT_LEN,
        n_actions     : int   = 5,
        c_puct        : float = 1.41,
        device        : str   = DEVICE,
    ):
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.n_actions     = n_actions
        self.c_puct        = c_puct
        self.device        = device

    @property
    def name(self) -> str:
        return "MCTS"

    # ── Phase 1 : Sélection ───────────────────────────────────────────────────

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_leaf() and node.depth < self.rollout_depth:
            node = node.best_child(self.c_puct)
        return node

    # ── Phase 2 : Expansion ───────────────────────────────────────────────────

    def _expand(
        self,
        node        : MCTSNode,
        world_model : WorldModelInterface,
        act_dim     : int,
        threshold   : float,
    ) -> List[MCTSNode]:
        if node.depth >= self.rollout_depth:
            return []

        actions  = torch.rand(self.n_actions, act_dim) * 2 - 1
        states_k = node.state.unsqueeze(0).expand(self.n_actions, -1)

        with torch.no_grad():
            u = world_model.uncertainty(
                states_k.to(self.device),
                actions.to(self.device),
            )

        valid_mask = (u.squeeze(1) <= threshold)
        if valid_mask.sum() == 0:
            return []

        valid_actions = actions[valid_mask]

        with torch.no_grad():
            next_states, rewards = world_model.predict(
                states_k[valid_mask].to(self.device),
                valid_actions.to(self.device),
            )

        new_children = []
        for i in range(valid_actions.shape[0]):
            child = MCTSNode(
                state  = next_states[i].cpu(),
                parent = node,
                action = valid_actions[i].cpu(),
                reward = rewards[i].item(),
                depth  = node.depth + 1,
            )
            new_children.append(child)

        node.children.extend(new_children)
        return new_children

    # ── Phase 3 : Simulation ──────────────────────────────────────────────────

    def _simulate(
        self,
        node        : MCTSNode,
        world_model : WorldModelInterface,
        act_dim     : int,
        threshold   : float,
    ) -> float:
        current_state = node.state.clone()
        total_return  = node.reward
        discount      = GAMMA
        depth         = node.depth

        while depth < self.rollout_depth:
            action = torch.rand(1, act_dim) * 2 - 1

            with torch.no_grad():
                u = world_model.uncertainty(
                    current_state.unsqueeze(0).to(self.device),
                    action.to(self.device),
                )

            if u.item() > threshold:
                break

            with torch.no_grad():
                next_state, reward = world_model.predict(
                    current_state.unsqueeze(0).to(self.device),
                    action.to(self.device),
                )

            total_return  += discount * reward.item()
            discount      *= GAMMA
            current_state  = next_state.squeeze(0).cpu()
            depth         += 1

        return total_return

    # ── Phase 4 : Backpropagation ─────────────────────────────────────────────

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum   += value
            current = current.parent

    # ── Collecte des transitions ──────────────────────────────────────────────

    def _collect_transitions(
        self,
        root : MCTSNode,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Parcourt l arbre en DFS et collecte toutes les transitions.
        Les transitions dans node.children ont déjà passé le filtre
        dans _expand() — pas de double check nécessaire.
        """
        obs_list, act_list, rew_list, next_list, done_list = [], [], [], [], []

        stack = [root]
        while stack:
            node = stack.pop()
            for child in node.children:
                if child.action is None:
                    continue
                obs_list.append(node.state)
                act_list.append(child.action)
                rew_list.append(torch.tensor([child.reward]))
                next_list.append(child.state)
                done_list.append(torch.tensor([0.0]))   # terminals=0
                stack.append(child)

        return {
            "observations"      : obs_list,
            "actions"           : act_list,
            "rewards"           : rew_list,
            "next_observations" : next_list,
            "terminals"         : done_list,
        }

    # ── Augmentation principale ───────────────────────────────────────────────

    def augment(
        self,
        buffer           : ReplayBufferInterface,
        world_model      : WorldModelInterface,
        n_new_transitions: int = 50_000,
    ) -> ReplayBufferInterface:
        print(f"\n[MCTS] Augmentation — cible {n_new_transitions:,} transitions "
              f"(sims={self.n_simulations}, depth={self.rollout_depth}, "
              f"k={self.n_actions})")

        # ── CORRECTION 1 : seuil centralisé ──────────────────────────────
        threshold  = calibrate_threshold(world_model, buffer)
        new_buffer = buffer.clone()
        all_data   = buffer.get_all()
        act_dim    = buffer.act_dim

        transitions_per_tree = self.rollout_depth * self.n_actions
        n_branch_points = max(1, (n_new_transitions // transitions_per_tree) + 1)
        n_branch_points = min(n_branch_points, buffer.size)

        idx         = torch.randint(0, buffer.size, (n_branch_points,))
        root_states = all_data["observations"][idx]

        print(f"[MCTS] Branch points : {n_branch_points:,}")

        all_obs, all_act, all_rew, all_next, all_done = [], [], [], [], []
        n_generated = 0

        for bp_idx in range(n_branch_points):
            if n_generated >= n_new_transitions:
                break

            root = MCTSNode(state=root_states[bp_idx], depth=0)

            for _ in range(self.n_simulations):
                leaf         = self._select(root)
                new_children = self._expand(leaf, world_model, act_dim, threshold)

                if new_children:
                    value = self._simulate(new_children[0], world_model, act_dim, threshold)
                    self._backpropagate(new_children[0], value)
                else:
                    self._backpropagate(leaf, 0.0)

            transitions = self._collect_transitions(root)
            n_new = len(transitions["observations"])
            if n_new == 0:
                continue

            all_obs.extend(transitions["observations"])
            all_act.extend(transitions["actions"])
            all_rew.extend(transitions["rewards"])
            all_next.extend(transitions["next_observations"])
            all_done.extend(transitions["terminals"])
            n_generated += n_new

        if n_generated == 0:
            print("[MCTS] WARN : aucune transition générée")
            return new_buffer

        if n_generated > n_new_transitions:
            all_obs   = all_obs[:n_new_transitions]
            all_act   = all_act[:n_new_transitions]
            all_rew   = all_rew[:n_new_transitions]
            all_next  = all_next[:n_new_transitions]
            all_done  = all_done[:n_new_transitions]
            n_generated = n_new_transitions

        batch = {
            "observations"      : torch.stack(all_obs),
            "actions"           : torch.stack(all_act),
            "rewards"           : torch.stack(all_rew),
            "next_observations" : torch.stack(all_next),
            "terminals"         : torch.stack(all_done),
        }
        new_buffer.add_batch(batch)

        print(f"[MCTS] Générées  : {n_generated:,}")
        print(f"[MCTS] Buffer D′ : {new_buffer.size:,} transitions")
        return new_buffer


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide — python src/mcts_augment.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("Test mcts_augment.py")
    print("=" * 55)

    from data_loader import ReplayBuffer
    from world_model import WorldModel

    cfg     = DATASET_CONFIGS[DEFAULT_DATASET]
    OBS_DIM = cfg["obs_dim"]
    ACT_DIM = cfg["act_dim"]
    N       = 5_000

    buf = ReplayBuffer(device=DEVICE)
    buf._data = {
        "observations"      : torch.randn(N, OBS_DIM),
        "actions"           : torch.randn(N, ACT_DIM).clamp(-1, 1),
        "rewards"           : torch.randn(N, 1) * 0.5 + 0.5,
        "next_observations" : torch.randn(N, OBS_DIM),
        "terminals"         : (torch.rand(N, 1) < 0.05).float(),
    }
    buf._size    = N
    buf._obs_dim = OBS_DIM
    buf._act_dim = ACT_DIM

    print("\n── Entraînement world model (5 epochs) ────────────────")
    wm = WorldModel(OBS_DIM, ACT_DIM, ensemble_size=5, device=DEVICE)
    wm.train_model(buf, n_epochs=5, batch_size=256)

    print("\n── Test MCTSAugmenter ─────────────────────────────────")
    mcts = MCTSAugmenter(
        n_simulations=10, rollout_depth=3,
        n_actions=4, c_puct=1.41, device=DEVICE,
    )
    assert mcts.name == "MCTS"

    buf_prime = mcts.augment(buf, wm, n_new_transitions=500)

    assert buf_prime.size > buf.size
    assert buf.size == N

    print(f"\n  D  original : {buf.size:,}")
    print(f"  D′ augmenté : {buf_prime.size:,}")
    print(f"  Ajoutées    : {buf_prime.size - buf.size:,}")

    print("\n── Test terminals = 0 ─────────────────────────────────")
    synth_terms = buf_prime.get_all()["terminals"][N:]
    assert synth_terms.max().item() == 0.0
    print("  ✓ terminals = 0.0")

    print("\n── Test sample() ──────────────────────────────────────")
    batch = buf_prime.sample(256)
    for k, v in batch.items():
        print(f"  {k:<22} shape={tuple(v.shape)}  ✓")

    print("\n── Test indépendance D / D′ ───────────────────────────")
    buf_prime._data["rewards"][0] = 9999.0
    assert buf._data["rewards"][0] != 9999.0
    print("  ✓ D original non modifié")

    print("\n" + "=" * 55)
    print("Tous les tests passent — mcts_augment.py prêt.")
    print("=" * 55)