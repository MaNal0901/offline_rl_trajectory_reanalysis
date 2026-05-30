"""
vine_augment.py — Vine Branching Augmentation
==============================================
Implémenté par : A
Dépend de      : interfaces.py, data_loader.py, world_model.py

Améliorations :
    1. calibrate_threshold() centralisé dans interfaces.py
    2. Actions vectorisées
    3. terminals=0 — rollout tronqué ≠ fin d épisode réelle
    4. Action fixe pendant tout le rollout — définition correcte de Vine
"""

from __future__ import annotations
from typing import Dict
import torch

from interfaces import (
    AugmenterInterface,
    ReplayBufferInterface,
    WorldModelInterface,
    DEVICE,
    ROLLOUT_LEN,
    DATASET_CONFIGS,
    DEFAULT_DATASET,
    calibrate_threshold,        # ← CORRECTION 1 : importé depuis interfaces
)


class VineAugmenter(AugmenterInterface):

    def __init__(
        self,
        rollout_horizon : int   = ROLLOUT_LEN,
        n_actions       : int   = 5,
        device          : str   = DEVICE,
    ):
        self.rollout_horizon = rollout_horizon
        self.n_actions       = n_actions
        self.device          = device
        # CORRECTION 1 : percentile supprimé — géré dans interfaces.py

    @property
    def name(self) -> str:
        return "VINE"

    # CORRECTION 1 : _calibrate_threshold() supprimée — remplacée par calibrate_threshold()

    def augment(
        self,
        buffer           : ReplayBufferInterface,
        world_model      : WorldModelInterface,
        n_new_transitions: int = 50_000,
    ) -> ReplayBufferInterface:
        print(f"\n[Vine] Augmentation — cible {n_new_transitions:,} transitions "
              f"(h={self.rollout_horizon}, k={self.n_actions})")

        # ── 1. Calibration — CORRECTION 1 : appel centralisé ─────────────
        threshold = calibrate_threshold(world_model, buffer)

        # ── 2. Clone du buffer original ───────────────────────────────────
        new_buffer = buffer.clone()

        # ── 3. Nombre de branch points ────────────────────────────────────
        transitions_per_point = self.rollout_horizon * self.n_actions
        n_branch_points = max(1, (n_new_transitions // transitions_per_point) + 1)
        n_branch_points = min(n_branch_points, buffer.size)

        print(f"[Vine] Branch points : {n_branch_points:,}  "
              f"× h={self.rollout_horizon} × k={self.n_actions}")

        # ── 4. Sélectionner les branch points ─────────────────────────────
        all_data    = buffer.get_all()
        idx         = torch.randint(0, buffer.size, (n_branch_points,))
        root_states = all_data["observations"][idx]

        # ── 5. Générer les branches ───────────────────────────────────────
        synth_obs, synth_act, synth_rew, synth_next, synth_done = [], [], [], [], []
        n_rejected  = 0
        n_generated = 0

        for bp_idx in range(n_branch_points):
            if n_generated >= n_new_transitions:
                break

            root     = root_states[bp_idx]
            actions  = torch.rand(self.n_actions, buffer.act_dim) * 2 - 1
            states_k = root.unsqueeze(0).expand(self.n_actions, -1)

            with torch.no_grad():
                u_init = world_model.uncertainty(
                    states_k.to(self.device),
                    actions.to(self.device),
                )

            valid_mask    = (u_init.squeeze(1) <= threshold)
            n_rejected   += (~valid_mask).sum().item()
            valid_actions = actions[valid_mask]

            for act_idx in range(valid_actions.shape[0]):
                if n_generated >= n_new_transitions:
                    break

                action        = valid_actions[act_idx]
                current_state = root.clone()

                for step in range(self.rollout_horizon):
                    if n_generated >= n_new_transitions:
                        break

                    with torch.no_grad():
                        u = world_model.uncertainty(
                            current_state.unsqueeze(0).to(self.device),
                            action.unsqueeze(0).to(self.device),
                        )

                    if u.item() > threshold:
                        n_rejected += 1
                        break

                    with torch.no_grad():
                        next_state, reward = world_model.predict(
                            current_state.unsqueeze(0).to(self.device),
                            action.unsqueeze(0).to(self.device),
                        )

                    next_state = next_state.squeeze(0).cpu()
                    reward     = reward.squeeze(0).cpu().view(1)
                    reward = reward.clamp(
                            all_data["rewards"].min(),
                            all_data["rewards"].max()
                        )


                    synth_obs.append(current_state.cpu())
                    synth_act.append(action.cpu())
                    synth_rew.append(reward)
                    synth_next.append(next_state)
                    synth_done.append(torch.tensor([0.0]))

                    n_generated  += 1
                    current_state = next_state.detach()

        # ── 6. Ajouter au buffer ──────────────────────────────────────────
        if n_generated == 0:
            print("[Vine] WARN : aucune transition générée")
            return new_buffer

        batch = {
            "observations"      : torch.stack(synth_obs),
            "actions"           : torch.stack(synth_act),
            "rewards"           : torch.stack(synth_rew),
            "next_observations" : torch.stack(synth_next),
            "terminals"         : torch.stack(synth_done),
        }
        new_buffer.add_batch(batch)

        acceptance_rate = n_generated / (n_generated + n_rejected) * 100
        print(f"[Vine] Générées   : {n_generated:,}")
        print(f"[Vine] Rejetées   : {n_rejected:,}  "
              f"(acceptance rate = {acceptance_rate:.1f}%)")
        print(f"[Vine] Buffer D′  : {new_buffer.size:,} transitions")

        return new_buffer


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide — python src/vine_augment.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("Test vine_augment.py")
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

    print("\n── Test VineAugmenter ─────────────────────────────────")
    vine = VineAugmenter(rollout_horizon=3, n_actions=5, device=DEVICE)
    assert vine.name == "VINE"

    buf_prime = vine.augment(buf, wm, n_new_transitions=500)

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
    print("Tous les tests passent — vine_augment.py prêt.")
    print("=" * 55)