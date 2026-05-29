"""
vine_augment.py — Vine Branching Augmentation
==============================================
Implémenté par : A
Dépend de      : interfaces.py, data_loader.py, world_model.py

Améliorations vs version initiale :
    1. Seuil adaptatif calibré sur les données réelles (percentile 90)
    2. Actions vectorisées — évite les boucles Python
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
    UNCERTAINTY_LIMIT,
    ROLLOUT_LEN,
    DATASET_CONFIGS,
    DEFAULT_DATASET,
)


class VineAugmenter(AugmenterInterface):
    """
    Vine branching — augmentation par rollouts courts.

    Pour chaque branch point sélectionné dans D :
        1. Tire k actions uniformes dans [-1, 1] — vectorisé
        2. Filtre d incertitude sur les k (state, action) en 1 appel
        3. Fait h pas de rollout pour les actions acceptées
        4. Rejette les transitions où uncertainty > seuil adaptatif
        5. Ajoute les transitions valides au nouveau buffer

    Hyperparamètres :
        rollout_horizon h : nombre de pas par branche (défaut = ROLLOUT_LEN)
        n_actions       k : actions testées par branch point (défaut = 5)
        percentile        : percentile pour calibrer le seuil (défaut = 90)
    """

    def __init__(
        self,
        rollout_horizon : int   = ROLLOUT_LEN,
        n_actions       : int   = 5,
        percentile      : float = 90.0,
        device          : str   = DEVICE,
    ):
        self.rollout_horizon = rollout_horizon
        self.n_actions       = n_actions
        self.percentile      = percentile
        self.device          = device

    @property
    def name(self) -> str:
        return "VINE"

    # ─────────────────────────────────────────────
    # SEUIL ADAPTATIF — nouveauté principale
    # ─────────────────────────────────────────────

    def _calibrate_threshold(
        self,
        buffer      : ReplayBufferInterface,
        world_model : WorldModelInterface,
        n_samples   : int = 2000,
    ) -> float:
        """
        Calcule le seuil d incertitude dynamiquement.

        Logique :
            1. Prend n_samples transitions réelles dans D
            2. Calcule l uncertainty du world model sur ces transitions
            3. Retourne le percentile 90
               → 90% des transitions réelles ont uncertainty < seuil
               → Vine rejette ce qui est plus incertain que les données réelles

        Exemple :
            uncertainty sur données réelles : [0.01 ... 0.08]
            percentile 90                   : 0.07
            → seuil = 0.07
            → on rejette tout ce qui a uncertainty > 0.07
        """
        n_samples = min(n_samples, buffer.size)
        batch     = buffer.sample(n_samples)

        states  = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)

        with torch.no_grad():
            u = world_model.uncertainty(states, actions)  # (n_samples, 1)

        threshold = torch.quantile(u.squeeze(), self.percentile / 100.0).item()

        print(f"[Vine] Calibration seuil :")
        print(f"       uncertainty min  = {u.min().item():.4f}")
        print(f"       uncertainty max  = {u.max().item():.4f}")
        print(f"       uncertainty mean = {u.mean().item():.4f}")
        print(f"       seuil (p{self.percentile:.0f}) = {threshold:.4f}")

        return threshold

    # ─────────────────────────────────────────────
    # AUGMENTATION PRINCIPALE
    # ─────────────────────────────────────────────

    def augment(
        self,
        buffer           : ReplayBufferInterface,
        world_model      : WorldModelInterface,
        n_new_transitions: int = 50_000,
    ) -> ReplayBufferInterface:
        """
        Génère n_new_transitions synthétiques par branchement.
        Retourne un NOUVEAU buffer = D + transitions synthétiques.
        Le buffer original n est PAS modifié.
        """
        print(f"\n[Vine] Augmentation — cible {n_new_transitions:,} transitions "
              f"(h={self.rollout_horizon}, k={self.n_actions})")

        # ── 1. Calibration du seuil adaptatif ────────────────────────────
        threshold = self._calibrate_threshold(buffer, world_model)

        # ── 2. Clone du buffer original — ne JAMAIS modifier D ───────────
        new_buffer = buffer.clone()

        # ── 3. Calculer le nombre de branch points nécessaires ───────────
        transitions_per_point = self.rollout_horizon * self.n_actions
        n_branch_points = max(
            1,
            (n_new_transitions // transitions_per_point) + 1
        )
        n_branch_points = min(n_branch_points, buffer.size)

        print(f"[Vine] Branch points : {n_branch_points:,}  "
              f"× h={self.rollout_horizon} × k={self.n_actions}")

        # ── 4. Sélectionner les branch points dans D ──────────────────────
        all_data    = buffer.get_all()
        idx         = torch.randint(0, buffer.size, (n_branch_points,))
        root_states = all_data["observations"][idx]  # (n_branch_points, obs_dim)

        # ── 5. Générer les branches ───────────────────────────────────────
        synth_obs, synth_act, synth_rew, synth_next, synth_done = (
            [], [], [], [], []
        )
        n_rejected  = 0
        n_generated = 0

        for bp_idx in range(n_branch_points):
            if n_generated >= n_new_transitions:
                break

            root = root_states[bp_idx]  # (obs_dim,)

            # Tirer k actions en une fois — vectorisé
            actions  = torch.rand(self.n_actions, buffer.act_dim) * 2 - 1
            states_k = root.unsqueeze(0).expand(self.n_actions, -1)

            # Filtrage initial sur les k actions — 1 seul appel
            with torch.no_grad():
                u_init = world_model.uncertainty(
                    states_k.to(self.device),
                    actions.to(self.device),
                )  # (k, 1)

            valid_mask    = (u_init.squeeze(1) <= threshold)
            n_rejected   += (~valid_mask).sum().item()
            valid_actions = actions[valid_mask]  # (n_valid, act_dim)

            # Action fixe pendant tout le rollout — définition correcte de Vine
            for act_idx in range(valid_actions.shape[0]):
                if n_generated >= n_new_transitions:
                    break

                action        = valid_actions[act_idx]
                current_state = root.clone()

                for step in range(self.rollout_horizon):
                    if n_generated >= n_new_transitions:
                        break

                    # Vérifier incertitude à chaque step
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

                    synth_obs.append(current_state.cpu())
                    synth_act.append(action.cpu())
                    synth_rew.append(reward)
                    synth_next.append(next_state)
                    # terminals=0 : rollout tronqué ≠ fin d épisode réelle
                    synth_done.append(torch.tensor([0.0]))

                    n_generated  += 1
                    current_state = next_state.detach()

        # ── 6. Ajouter au buffer ──────────────────────────────────────────
        if n_generated == 0:
            print("[Vine] WARN : aucune transition générée — "
                  "world model non entraîné ou seuil trop bas")
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
    print("Test vine_augment.py — version améliorée")
    print("=" * 55)

    import sys
    sys.path.insert(0, 'src')
    from data_loader import ReplayBuffer
    from world_model import WorldModel

    cfg     = DATASET_CONFIGS[DEFAULT_DATASET]
    OBS_DIM = cfg["obs_dim"]
    ACT_DIM = cfg["act_dim"]
    N       = 5_000

    # ── Buffer synthétique ────────────────────────────────────────────────
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

    # ── World model entraîné ──────────────────────────────────────────────
    print("\n── Entraînement world model (5 epochs) ────────────────")
    wm = WorldModel(OBS_DIM, ACT_DIM, ensemble_size=5, device=DEVICE)
    wm.train_model(buf, n_epochs=5, batch_size=256)

    # ── Test Vine amélioré ────────────────────────────────────────────────
    print("\n── Test VineAugmenter (seuil adaptatif) ───────────────")
    vine = VineAugmenter(
        rollout_horizon = 3,
        n_actions       = 5,
        percentile      = 90.0,
        device          = DEVICE,
    )
    assert vine.name == "VINE"

    buf_prime = vine.augment(buf, wm, n_new_transitions=500)

    assert buf_prime.size > buf.size, "D′ doit être plus grand que D"
    assert buf.size == N,             "D original ne doit pas être modifié"

    print(f"\n  D  original : {buf.size:,} transitions")
    print(f"  D′ augmenté : {buf_prime.size:,} transitions")
    print(f"  Ajoutées    : {buf_prime.size - buf.size:,}")

    # ── Vérifier terminals = 0 ────────────────────────────────────────────
    print("\n── Test terminals = 0 ─────────────────────────────────")
    all_d_prime = buf_prime.get_all()
    synth_terms = all_d_prime["terminals"][N:]
    assert synth_terms.max().item() == 0.0
    print("  ✓ terminals synthétiques = 0.0 partout")

    # ── Test sample() ─────────────────────────────────────────────────────
    print("\n── Test sample() sur D′ ───────────────────────────────")
    batch = buf_prime.sample(256)
    for k, v in batch.items():
        print(f"  {k:<22} shape={tuple(v.shape)}  ✓")

    # ── Test indépendance D / D′ ──────────────────────────────────────────
    print("\n── Test indépendance D / D′ ───────────────────────────")
    buf_prime._data["rewards"][0] = 9999.0
    assert buf._data["rewards"][0] != 9999.0
    print("  ✓ D original non modifié")

    print("\n" + "=" * 55)
    print("Tous les tests passent — vine_augment.py prêt.")
    print("=" * 55)