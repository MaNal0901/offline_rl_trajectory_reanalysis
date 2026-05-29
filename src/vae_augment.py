"""
vae_augment.py — VAE-based Trajectory Augmentation
====================================================
Dépend de      : interfaces.py, data_loader.py, world_model.py

Corrections :
    1. calibrate_threshold() importé depuis interfaces.py
    2. sys.path.insert supprimé
"""

from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from interfaces import (
    AugmenterInterface,
    ReplayBufferInterface,
    WorldModelInterface,
    DEVICE,
    BATCH_SIZE,
    DATASET_CONFIGS,
    DEFAULT_DATASET,
    calibrate_threshold,        # CORRECTION 1
)


# ─────────────────────────────────────────────────────────────────────────────
# Encodeur
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),   nn.SiLU(),
        )
        self.fc_mu      = nn.Linear(hidden, latent_dim)
        self.fc_log_var = nn.Linear(hidden, latent_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_log_var(h).clamp(-4, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Décodeur
# ─────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),     nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ─────────────────────────────────────────────────────────────────────────────
# VAE complet
# ─────────────────────────────────────────────────────────────────────────────

class TransitionVAE(nn.Module):
    """
    VAE entraîné sur les paires (observations, actions) du buffer.
    Loss = reconstruction_loss + β * KL_divergence
    """

    def __init__(
        self,
        obs_dim    : int,
        act_dim    : int,
        latent_dim : int   = 32,
        hidden     : int   = 256,
        beta       : float = 1.0,
        device     : str   = DEVICE,
    ):
        super().__init__()
        input_dim        = obs_dim + act_dim
        self.obs_dim     = obs_dim
        self.act_dim     = act_dim
        self.latent_dim  = latent_dim
        self.beta        = beta
        self.device_name = device

        self.encoder = Encoder(input_dim, latent_dim, hidden)
        self.decoder = Decoder(latent_dim, input_dim, hidden)
        self.to(device)

        self._mean : torch.Tensor | None = None
        self._std  : torch.Tensor | None = None

    def fit_normalizer(self, obs: torch.Tensor, act: torch.Tensor) -> None:
        x          = torch.cat([obs, act], dim=-1).to(self.device_name)
        self._mean = x.mean(0, keepdim=True)
        self._std  = x.std(0, keepdim=True).clamp(min=1e-6)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is None:
            return x
        return (x - self._mean) / self._std

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is None:
            return x
        return x * self._std + self._mean

    def reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def forward(self, obs, act):
        x     = torch.cat([obs, act], dim=-1)
        x_n   = self._normalize(x)
        mu, log_var = self.encoder(x_n)
        z     = self.reparametrize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var, z

    def loss(self, obs, act):
        x         = torch.cat([obs, act], dim=-1)
        x_n       = self._normalize(x)
        x_hat, mu, log_var, _ = self.forward(obs, act)
        recon_loss = F.mse_loss(x_hat, x_n)
        kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            z     = torch.randn(n, self.latent_dim, device=self.device_name)
            x_hat = self.decoder(z)
            x     = self._denormalize(x_hat)
        obs_gen = x[:, :self.obs_dim].clamp(-10, 10)
        act_gen = x[:, self.obs_dim:].clamp(-1, 1)
        return obs_gen, act_gen

    def train_vae(
        self,
        buffer    : ReplayBufferInterface,
        n_epochs  : int   = 30,
        batch_size: int   = BATCH_SIZE,
        lr        : float = 1e-3,
    ) -> Dict[str, float]:
        print(f"[VAE] Entraînement — {n_epochs} epochs, "
              f"latent_dim={self.latent_dim}, β={self.beta}")

        all_data = buffer.get_all()
        obs, act = all_data["observations"].float(), all_data["actions"].float()
        self.fit_normalizer(obs, act)

        loader = DataLoader(TensorDataset(obs, act),
                            batch_size=batch_size, shuffle=True)
        opt    = Adam(self.parameters(), lr=lr, weight_decay=1e-5)

        last_total = last_recon = last_kl = 0.0
        for epoch in range(1, n_epochs + 1):
            self.train()
            e_total = e_recon = e_kl = 0.0
            for ob_b, ac_b in loader:
                ob_b, ac_b = ob_b.to(self.device_name), ac_b.to(self.device_name)
                total, recon, kl = self.loss(ob_b, ac_b)
                opt.zero_grad(); total.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()
                e_total += total.item(); e_recon += recon.item(); e_kl += kl.item()

            n = len(loader)
            last_total, last_recon, last_kl = e_total/n, e_recon/n, e_kl/n
            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:>3}/{n_epochs}  "
                      f"loss={last_total:.4f}  recon={last_recon:.4f}  kl={last_kl:.4f}")

        print(f"[VAE] Terminé — loss={last_total:.4f}")
        return {"train_loss": last_total, "recon_loss": last_recon, "kl_loss": last_kl}


# ─────────────────────────────────────────────────────────────────────────────
# VAE Augmenter
# ─────────────────────────────────────────────────────────────────────────────

class VAEAugmenter(AugmenterInterface):
    """
    Augmentation basée sur un VAE entraîné sur les transitions de D.
    Utilise calibrate_threshold() depuis interfaces.py — même seuil que Vine et MCTS.
    """

    def __init__(
        self,
        latent_dim : int   = 32,
        vae_epochs : int   = 30,
        beta       : float = 1.0,
        batch_gen  : int   = 2048,
        device     : str   = DEVICE,
    ):
        self.latent_dim = latent_dim
        self.vae_epochs = vae_epochs
        self.beta       = beta
        self.batch_gen  = batch_gen
        self.device     = device
        self._vae : TransitionVAE | None = None

    @property
    def name(self) -> str:
        return "VAE"

    def augment(
        self,
        buffer           : ReplayBufferInterface,
        world_model      : WorldModelInterface,
        n_new_transitions: int = 50_000,
    ) -> ReplayBufferInterface:
        print(f"\n[VAE] Augmentation — cible {n_new_transitions:,} transitions")

        # ── 1. Entraîner le VAE sur D ─────────────────────────────────────
        self._vae = TransitionVAE(
            obs_dim    = buffer.obs_dim,
            act_dim    = buffer.act_dim,
            latent_dim = self.latent_dim,
            beta       = self.beta,
            device     = self.device,
        )
        self._vae.train_vae(buffer, n_epochs=self.vae_epochs)

        # ── 2. Calibrer seuil — CORRECTION 1 : centralisé ────────────────
        threshold  = calibrate_threshold(world_model, buffer)
        new_buffer = buffer.clone()

        # ── 3. Générer jusqu'à n_new_transitions ──────────────────────────
        all_obs, all_act, all_rew, all_next, all_done = [], [], [], [], []
        n_generated = 0
        n_rejected  = 0

        while n_generated < n_new_transitions:
            remaining  = n_new_transitions - n_generated
            batch_size = min(self.batch_gen, remaining * 4)

            obs_gen, act_gen = self._vae.sample(batch_size)

            with torch.no_grad():
                u = world_model.uncertainty(
                    obs_gen.to(self.device),
                    act_gen.to(self.device),
                )

            valid_mask  = (u.squeeze(1) <= threshold)
            n_rejected += (~valid_mask).sum().item()

            obs_valid = obs_gen[valid_mask]
            act_valid = act_gen[valid_mask]

            if obs_valid.shape[0] == 0:
                if n_rejected > n_new_transitions * 20:
                    print("[VAE] WARN : taux de rejet trop élevé — arrêt anticipé")
                    break
                continue

            with torch.no_grad():
                next_obs, rewards = world_model.predict(
                    obs_valid.to(self.device),
                    act_valid.to(self.device),
                )

            n_to_add = min(obs_valid.shape[0], n_new_transitions - n_generated)
            all_obs.append(obs_valid[:n_to_add].cpu())
            all_act.append(act_valid[:n_to_add].cpu())
            all_rew.append(rewards[:n_to_add].cpu())
            all_next.append(next_obs[:n_to_add].cpu())
            all_done.append(torch.zeros(n_to_add, 1))
            n_generated += n_to_add

        # ── 4. Ajouter au buffer ──────────────────────────────────────────
        if n_generated == 0:
            print("[VAE] WARN : aucune transition générée.")
            return new_buffer

        batch = {
            "observations"      : torch.cat(all_obs),
            "actions"           : torch.cat(all_act),
            "rewards"           : torch.cat(all_rew),
            "next_observations" : torch.cat(all_next),
            "terminals"         : torch.cat(all_done),
        }
        new_buffer.add_batch(batch)

        total_attempted = n_generated + n_rejected
        acceptance_rate = n_generated / total_attempted * 100 if total_attempted > 0 else 0
        print(f"[VAE] Générées  : {n_generated:,}")
        print(f"[VAE] Rejetées  : {n_rejected:,}  "
              f"(acceptance rate = {acceptance_rate:.1f}%)")
        print(f"[VAE] Buffer D′ : {new_buffer.size:,} transitions")
        return new_buffer


# ─────────────────────────────────────────────────────────────────────────────
# Test — python src/vae_augment.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # CORRECTION 2 — sys.path.insert supprimé

    print("=" * 55)
    print("Test vae_augment.py")
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

    print("\n── Test VAEAugmenter ──────────────────────────────────")
    vae_aug = VAEAugmenter(
        latent_dim=16, vae_epochs=10, beta=1.0, device=DEVICE,
    )
    assert vae_aug.name == "VAE"

    buf_prime = vae_aug.augment(buf, wm, n_new_transitions=500)

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

    print("\n── Test VAE.sample() standalone ───────────────────────")
    obs_s, act_s = vae_aug._vae.sample(64)
    assert obs_s.shape == (64, OBS_DIM)
    assert act_s.shape == (64, ACT_DIM)
    assert act_s.abs().max().item() <= 1.0 + 1e-5
    print(f"  obs_gen : {tuple(obs_s.shape)}  ✓")
    print(f"  act_gen : {tuple(act_s.shape)}  ✓")

    print("\n" + "=" * 55)
    print("Tous les tests passent — vae_augment.py prêt.")
    print("=" * 55)