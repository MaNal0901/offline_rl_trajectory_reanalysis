"""
world_model.py — MLP Ensemble World Model
==========================================
Implémenté par : A
Dépend de      : interfaces.py, data_loader.py

Implémente WorldModelInterface.
B appelle uniquement .predict() et .uncertainty() — rien d'autre.

Rôle dans le pipeline :
    - Vine  : génère les rollouts synthétiques
    - MCTS  : moteur de simulation de l'arbre
    - VAE   : filtre les transitions générées (uncertainty > calibrate_threshold(wm, buffer))

Usage rapide :
    python src/world_model.py
"""

from __future__ import annotations
from typing import Dict, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from interfaces import (
    WorldModelInterface,
    ReplayBufferInterface,
    DEVICE,
    BATCH_SIZE,
    UNCERTAINTY_PERCENTILE, 
    DATASET_CONFIGS,
    DEFAULT_DATASET,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MLP membre de l'ensemble
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleMember(nn.Module):
    """
    Un seul MLP de l'ensemble.
    Prédit (delta_obs, reward) depuis (obs, action).
    On prédit le DELTA d'état (s' - s) plutôt que s' directement
    → apprentissage plus stable, meilleure généralisation.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        inp = obs_dim + act_dim
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, obs_dim + 1),  # delta_obs + reward
        )
        # initialisation orthogonale — meilleure stabilité que Xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        obs: torch.Tensor,   # (B, obs_dim)
        act: torch.Tensor,   # (B, act_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x         = torch.cat([obs, act], dim=-1)
        out       = self.net(x)
        delta_obs = out[:, :-1]   # (B, obs_dim)
        reward    = out[:, -1:]   # (B, 1)
        return delta_obs, reward


# ─────────────────────────────────────────────────────────────────────────────
# 2. World Model — ensemble de MLPs
# ─────────────────────────────────────────────────────────────────────────────

class WorldModel(WorldModelInterface):
    """
    Ensemble de N MLPs identiques entraînés avec des mini-batchs différents
    (bootstrap). La variance inter-membres = incertitude épistémique.

    B appelle :
        next_obs, reward = world_model.predict(obs, act)
        u                = world_model.uncertainty(obs, act)
        # rejeter si u >calibrate_threshold(wm, buffer)
    """

    def __init__(
        self,
        obs_dim       : int,
        act_dim       : int,
        ensemble_size : int = 5,
        hidden        : int = 256,
        device        : str = DEVICE,
    ):
        self.obs_dim       = obs_dim
        self.act_dim       = act_dim
        self.ensemble_size = ensemble_size
        self.device        = device

        self.members: nn.ModuleList = nn.ModuleList([
            EnsembleMember(obs_dim, act_dim, hidden).to(device)
            for _ in range(ensemble_size)
        ])
        self._trained = False

        self._obs_mean: torch.Tensor | None = None
        self._obs_std:  torch.Tensor | None = None
        self._act_mean: torch.Tensor | None = None
        self._act_std:  torch.Tensor | None = None

    # ── Normalisation interne ─────────────────────────────────────────────────

    def _fit_normalizer(self, obs: torch.Tensor, act: torch.Tensor) -> None:
        """Calcule mean/std sur le dataset complet — appelé une fois au début du train."""
        self._obs_mean = obs.mean(0, keepdim=True).to(self.device)
        self._obs_std  = obs.std(0, keepdim=True).clamp(min=1e-6).to(self.device)
        self._act_mean = act.mean(0, keepdim=True).to(self.device)
        self._act_std  = act.std(0, keepdim=True).clamp(min=1e-6).to(self.device)

    def _normalize(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_n = (obs - self._obs_mean) / self._obs_std
        act_n = (act - self._act_mean) / self._act_std
        return obs_n, act_n

    # ── Interface obligatoire ─────────────────────────────────────────────────

    def predict(
        self,
        state : torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne la prédiction MOYENNE de l'ensemble.
        Supporte (obs_dim,) et (B, obs_dim) — auto-reshape.

        Retourne : (next_state, reward)
            next_state : (B, obs_dim)
            reward     : (B, 1)
        """
        if not self._trained:
            raise RuntimeError("Appelle train_model() avant predict().")

        squeeze = state.ndim == 1
        if squeeze:
            state  = state.unsqueeze(0)
            action = action.unsqueeze(0)

        state  = state.to(self.device)
        action = action.to(self.device)
        obs_n, act_n = self._normalize(state, action)

        deltas, rewards = [], []
        with torch.no_grad():
            for member in self.members:
                d, r = member(obs_n, act_n)
                deltas.append(d)
                rewards.append(r)

        delta_mean  = torch.stack(deltas,  dim=0).mean(0)
        reward_mean = torch.stack(rewards, dim=0).mean(0)
        next_state  = state + delta_mean

        if squeeze:
            next_state  = next_state.squeeze(0)
            reward_mean = reward_mean.squeeze(0)

        return next_state, reward_mean

    def uncertainty(
        self,
        state : torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retourne la variance inter-membres sur le delta prédit.
        Règle partagée : rejeter si uncertainty > calibrate_threshold(wm, buffer)

        Retourne : (B, 1)
        """
        if not self._trained:
            raise RuntimeError("Appelle train_model() avant uncertainty().")

        state  = state.to(self.device)
        action = action.to(self.device)
        obs_n, act_n = self._normalize(state, action)

        deltas = []
        with torch.no_grad():
            for member in self.members:
                d, _ = member(obs_n, act_n)
                deltas.append(d)

        stacked  = torch.stack(deltas, dim=0)              # (E, B, obs_dim)
        variance = stacked.var(dim=0).mean(dim=-1, keepdim=True)  # (B, 1)
        return variance

    def train_model(
        self,
        buffer    : ReplayBufferInterface,
        n_epochs  : int = 50,
        batch_size: int = BATCH_SIZE,
    ) -> Dict[str, float]:
        """
        Entraîne l'ensemble sur le buffer D.
        Chaque membre voit un sous-ensemble bootstrap différent.

        Retourne : {"train_loss": float, "val_loss": float}
        """
        print(f"[WorldModel] Entraînement — {self.ensemble_size} membres "
              f"× {n_epochs} epochs")

        all_data = buffer.get_all()
        obs      = all_data["observations"].float()
        act      = all_data["actions"].float()
        rew      = all_data["rewards"].float()
        next_obs = all_data["next_observations"].float()

        n     = obs.shape[0]
        delta = next_obs - obs

        self._fit_normalizer(obs, act)
        obs_n = (obs - self._obs_mean.cpu()) / self._obs_std.cpu()
        act_n = (act - self._act_mean.cpu()) / self._act_std.cpu()

        n_val     = max(1000, int(0.1 * n))
        n_train   = n - n_val
        idx_perm  = torch.randperm(n)
        idx_train = idx_perm[:n_train]
        idx_val   = idx_perm[n_train:]

        obs_tr, obs_val = obs_n[idx_train], obs_n[idx_val]
        act_tr, act_val = act_n[idx_train], act_n[idx_val]
        dlt_tr, dlt_val = delta[idx_train], delta[idx_val]
        rew_tr, rew_val = rew[idx_train],   rew[idx_val]

        optimizers = [
            Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
            for m in self.members
        ]

        best_val_loss = float("inf")
        train_losses, val_losses = [], []

        for epoch in range(1, n_epochs + 1):
            epoch_train = 0.0
            for member, opt in zip(self.members, optimizers):
                member.train()
                boot_idx = torch.randint(0, n_train, (n_train,))
                ds = TensorDataset(
                    obs_tr[boot_idx], act_tr[boot_idx],
                    dlt_tr[boot_idx], rew_tr[boot_idx],
                )
                loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

                member_loss = 0.0
                for ob, ac, dt, rw in loader:
                    ob, ac, dt, rw = (
                        ob.to(self.device), ac.to(self.device),
                        dt.to(self.device), rw.to(self.device),
                    )
                    pred_delta, pred_rew = member(ob, ac)
                    loss = F.mse_loss(pred_delta, dt) + F.mse_loss(pred_rew, rw)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(member.parameters(), 1.0)
                    opt.step()
                    member_loss += loss.item()

                epoch_train += member_loss / len(loader)

            epoch_train /= self.ensemble_size
            train_losses.append(epoch_train)

            val_loss = self._eval_val(obs_val, act_val, dlt_val, rew_val)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:>3}/{n_epochs}  "
                      f"train={epoch_train:.4f}  val={val_loss:.4f}")

        self._trained = True
        result = {
            "train_loss": float(train_losses[-1]),
            "val_loss":   float(val_losses[-1]),
        }
        print(f"[WorldModel] Terminé — val_loss={result['val_loss']:.4f}")
        return result

    def _eval_val(
        self,
        obs_val: torch.Tensor,
        act_val: torch.Tensor,
        dlt_val: torch.Tensor,
        rew_val: torch.Tensor,
    ) -> float:
        total = 0.0
        for member in self.members:
            member.eval()
            with torch.no_grad():
                pd, pr = member(
                    obs_val.to(self.device),
                    act_val.to(self.device),
                )
                total += (
                    F.mse_loss(pd, dlt_val.to(self.device)) +
                    F.mse_loss(pr, rew_val.to(self.device))
                ).item()
        return total / self.ensemble_size

    # ── Sauvegarde / chargement ───────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Sauvegarde le modèle. Usage : wm.save("checkpoints/wm_hopper.pt")"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "members_state": [m.state_dict() for m in self.members],
            "obs_mean": self._obs_mean,
            "obs_std":  self._obs_std,
            "act_mean": self._act_mean,
            "act_std":  self._act_std,
            "obs_dim":  self.obs_dim,
            "act_dim":  self.act_dim,
            "trained":  self._trained,
        }, path)
        print(f"[WorldModel] Sauvegardé → {path}")

    def load(self, path: str) -> "WorldModel":
        """Charge un modèle sauvegardé. Retourne self pour chaining."""
        # CORRECTION 2 : weights_only=False — évite le warning PyTorch 2.x
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        for member, state in zip(self.members, ckpt["members_state"]):
            member.load_state_dict(state)
        self._obs_mean = ckpt["obs_mean"]
        self._obs_std  = ckpt["obs_std"]
        self._act_mean = ckpt["act_mean"]
        self._act_std  = ckpt["act_std"]
        self._trained  = ckpt["trained"]
        print(f"[WorldModel] Chargé depuis {path}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fonction utilitaire
# ─────────────────────────────────────────────────────────────────────────────

def build_world_model(env_name: str, device: str = DEVICE) -> WorldModel:
    """
    Construit un WorldModel avec les bonnes dimensions pour l'env donné.

    Usage :
        wm = build_world_model("hopper-medium-v2")
        wm.train_model(buffer)
    """
    if env_name not in DATASET_CONFIGS:
        raise KeyError(f"'{env_name}' inconnu.")
    cfg = DATASET_CONFIGS[env_name]
    return WorldModel(obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"], device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide — python src/world_model.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # CORRECTION 1 : sys.path.insert supprimé — inutile et dangereux
    # CORRECTION 3 : mkdir -p checkpoints à faire avant de lancer

    print("=" * 55)
    print("Test world_model.py")
    print("=" * 55)

    cfg     = DATASET_CONFIGS[DEFAULT_DATASET]
    OBS_DIM = cfg["obs_dim"]   # 11
    ACT_DIM = cfg["act_dim"]   # 3
    N       = 5_000

    # ── 1. Buffer synthétique ─────────────────────────────────────────────
    print("\n── Création buffer synthétique ────────────────────────")
    from data_loader import ReplayBuffer
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
    print(f"  {buf}")

    # ── 2. Train 10 epochs ────────────────────────────────────────────────
    print("\n── Entraînement (10 epochs) ───────────────────────────")
    wm      = WorldModel(OBS_DIM, ACT_DIM, ensemble_size=5, device=DEVICE)
    metrics = wm.train_model(buf, n_epochs=10, batch_size=256)
    print(f"  train_loss = {metrics['train_loss']:.4f}")
    print(f"  val_loss   = {metrics['val_loss']:.4f}")

    # ── 3. Test predict() batch ───────────────────────────────────────────
    print("\n── Test predict() batch ───────────────────────────────")
    obs_t = torch.randn(8, OBS_DIM)
    act_t = torch.randn(8, ACT_DIM)
    next_s, rew = wm.predict(obs_t, act_t)
    assert next_s.shape == (8, OBS_DIM)
    assert rew.shape    == (8, 1)
    print(f"  next_state : {tuple(next_s.shape)}  ✓")
    print(f"  reward     : {tuple(rew.shape)}      ✓")

    # ── 4. Test predict() single sample ──────────────────────────────────
    print("\n── Test predict() single sample ───────────────────────")
    ns1, r1 = wm.predict(torch.randn(OBS_DIM), torch.randn(ACT_DIM))
    assert ns1.shape == (OBS_DIM,)
    print(f"  single sample : {tuple(ns1.shape)}  ✓")

    # ── 5. Test uncertainty() ─────────────────────────────────────────────
    print("\n── Test uncertainty() ─────────────────────────────────")
    u = wm.uncertainty(obs_t, act_t)
    assert u.shape == (8, 1)
    print(f"  min={u.min():.4f}  max={u.max():.4f}  mean={u.mean():.4f}")
    print(f"  (seuil calibré dynamiquement via calibrate_threshold)")

    # ── 6. Test save / load ───────────────────────────────────────────────
    print("\n── Test save / load ───────────────────────────────────")
    wm.save("checkpoints/wm_test.pt")
    wm2 = WorldModel(OBS_DIM, ACT_DIM, device=DEVICE)
    wm2.load("checkpoints/wm_test.pt")
    ns2, _ = wm2.predict(obs_t, act_t)
    assert torch.allclose(next_s, ns2, atol=1e-5)
    print("  ✓ save/load — prédictions identiques")

    print("\n" + "=" * 55)
    print("Tous les tests passent — world_model.py prêt.")
    print("B peut appeler :")
    print("  wm.predict(obs, act)      → next_obs, reward")
    print("  wm.uncertainty(obs, act)  → u  (rejeter si > 0.5)")
    print("=" * 55)