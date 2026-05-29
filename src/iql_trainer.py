"""
iql_trainer.py — Implicit Q-Learning Trainer
=============================================
Implémenté par : A
Dépend de      : interfaces.py, data_loader.py

Implémente IQL identique pour les 4 runs :
    - baseline     : train(buffer_original)
    - vine         : train(buffer_vine)
    - mcts         : train(buffer_mcts)
    - vae          : train(buffer_vae)

Seul le buffer change — tout le reste est identique.

Usage rapide :
    python src/iql_trainer.py
"""

from __future__ import annotations
from typing import Dict
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from interfaces import (
    ReplayBufferInterface,
    DEVICE,
    BATCH_SIZE,
    GAMMA,
    DATASET_CONFIGS,
    DEFAULT_DATASET,
    normalized_score,
)

# CORRECTION 3 — constante calculée une seule fois
LOG2 = math.log(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. RÉSEAUX
# ─────────────────────────────────────────────────────────────────────────────

def mlp(dims: list, activation=nn.ReLU) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class ValueNet(nn.Module):
    """V(s) — réseau de valeur d'état."""
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, 1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QNet(nn.Module):
    """Double Q-network — Q1(s,a) et Q2(s,a)."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, hidden, hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, act)
        return torch.min(q1, q2)


class GaussianActor(nn.Module):
    """Politique gaussienne π(a|s)."""
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net     = mlp([obs_dim, hidden, hidden])
        self.mu_head = nn.Linear(hidden, act_dim)
        self.ls_head = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor):
        """Sample une action — utilisé pendant l'entraînement SAC-like."""
        h       = self.net(obs)
        mu      = self.mu_head(h)
        log_std = self.ls_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std     = log_std.exp()
        dist    = torch.distributions.Normal(mu, std)
        raw     = dist.rsample()
        act     = torch.tanh(raw)
        log_pi  = dist.log_prob(raw).sum(-1, keepdim=True)
        # CORRECTION 3 — LOG2 constant au lieu de torch.log(torch.tensor(2.0))
        log_pi -= (2 * (LOG2 - raw - F.softplus(-2 * raw))).sum(-1, keepdim=True)
        return act, log_pi

    def log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """
        CORRECTION 1 — log_prob des actions du DATASET.
        Utilisé dans l'actor update IQL (AWR sur actions réelles).

        C'est la différence clé avec AWR/SAC :
            IQL  : log π(a_dataset | s)  ← cette méthode
            SAC  : log π(a_sampled  | s)  ← forward()

        act : actions du dataset dans [-1, 1] — on inverse tanh pour
              obtenir les raw actions avant squashing.
        """
        h       = self.net(obs)
        mu      = self.mu_head(h)
        log_std = self.ls_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std     = log_std.exp()

        # inverser tanh : raw = atanh(act), clampé pour stabilité numérique
        act_clamped = act.clamp(-1 + 1e-6, 1 - 1e-6)
        raw         = torch.atanh(act_clamped)

        dist   = torch.distributions.Normal(mu, std)
        log_pi = dist.log_prob(raw).sum(-1, keepdim=True)
        log_pi -= (2 * (LOG2 - raw - F.softplus(-2 * raw))).sum(-1, keepdim=True)
        return log_pi

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Inférence déterministe — utilise la moyenne."""
        h  = self.net(obs.unsqueeze(0))
        mu = self.mu_head(h)
        return torch.tanh(mu).squeeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. IQL AGENT
# ─────────────────────────────────────────────────────────────────────────────

class IQLAgent:
    """
    Implicit Q-Learning (Kostrikov et al., 2021).

    Trois mises à jour par step :
        1. V(s)   — expectile regression sur Q_target(s,a)
        2. Q(s,a) — Bellman avec V(s′)
        3. π(a|s) — AWR sur actions du dataset (IQL strict)
    """

    def __init__(
        self,
        obs_dim   : int,
        act_dim   : int,
        device    : str   = DEVICE,
        lr        : float = 3e-4,
        gamma     : float = GAMMA,
        tau       : float = 0.005,
        expectile : float = 0.7,
        temp      : float = 3.0,
        hidden    : int   = 256,
    ):
        self.device    = device
        self.gamma     = gamma
        self.tau       = tau
        self.expectile = expectile
        self.temp      = temp

        self.actor       = GaussianActor(obs_dim, act_dim, hidden).to(device)
        self.qnet        = QNet(obs_dim, act_dim, hidden).to(device)
        self.qnet_target = QNet(obs_dim, act_dim, hidden).to(device)
        self.vnet        = ValueNet(obs_dim, hidden).to(device)

        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.actor_opt = Adam(self.actor.parameters(), lr=lr)
        self.q_opt     = Adam(self.qnet.parameters(),  lr=lr)
        self.v_opt     = Adam(self.vnet.parameters(),  lr=lr)

    @staticmethod
    def _expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
        # torch.full_like — même device et dtype que diff, pas de scalaire flottant
        weight = torch.where(
            diff > 0,
            torch.full_like(diff, tau),
            torch.full_like(diff, 1.0 - tau),
        )
        return (weight * diff.pow(2)).mean()

    def _soft_update(self) -> None:
        for p, pt in zip(self.qnet.parameters(),
                         self.qnet_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs      = batch["observations"]
        act      = batch["actions"]
        rew      = batch["rewards"]
        next_obs = batch["next_observations"]
        done     = batch["terminals"]

        # ── 1. Value update ───────────────────────────────────────────────
        with torch.no_grad():
            q1_t, q2_t = self.qnet_target(obs, act)
            q_t        = torch.min(q1_t, q2_t)

        v_pred = self.vnet(obs)
        v_loss = self._expectile_loss(q_t - v_pred, self.expectile)

        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()

        # ── 2. Q update ───────────────────────────────────────────────────
        with torch.no_grad():
            v_next = self.vnet(next_obs)
            target = rew + self.gamma * (1 - done) * v_next

        q1, q2 = self.qnet(obs, act)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # ── 3. Actor update — IQL strict ──────────────────────────────────
        # CORRECTION 1 : log_prob des actions du DATASET (pas samplées)
        # L_π = E[exp(β*A(s,a)) * log π(a_dataset|s)]
        with torch.no_grad():
            v_s = self.vnet(obs)
            q_s = self.qnet_target.q_min(obs, act)
            adv = q_s - v_s
            w   = (self.temp * adv).exp().clamp(max=100.0)

        log_pi     = self.actor.log_prob(obs, act)   # ← actions du dataset
        actor_loss = -(w * log_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update()

        return {
            "v_loss"    : v_loss.item(),
            "q_loss"    : q_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "qnet" : self.qnet.state_dict(),
            "vnet" : self.vnet.state_dict(),
        }, path)
        print(f"[IQL] Sauvegardé → {path}")

    def load(self, path: str) -> "IQLAgent":
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.qnet.load_state_dict(ckpt["qnet"])
        self.vnet.load_state_dict(ckpt["vnet"])
        self.qnet_target.load_state_dict(ckpt["qnet"])
        print(f"[IQL] Chargé depuis {path}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class IQLTrainer:
    """
    Boucle d'entraînement IQL.
    Identique pour les 4 runs — seul le buffer change.
    """

    def __init__(
        self,
        obs_dim   : int,
        act_dim   : int,
        env_name  : str,
        device    : str = DEVICE,
        batch_size: int = BATCH_SIZE,
        log_every : int = 5_000,
    ):
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.env_name   = env_name
        self.device     = device
        self.batch_size = batch_size
        self.log_every  = log_every

        self.agent = IQLAgent(obs_dim, act_dim, device=device)
        self._log: Dict[str, list] = {
            "step": [], "v_loss": [], "q_loss": [],
            "actor_loss": [], "score": []
        }

    def train(
        self,
        buffer : ReplayBufferInterface,
        n_steps: int = 100_000,
        method : str = "baseline",
    ) -> Dict[str, list]:
        print(f"\n[IQL] Entraînement — method={method}  "
              f"buffer={buffer.size:,}  steps={n_steps:,}")

        for step in range(1, n_steps + 1):
            batch   = buffer.sample(self.batch_size)
            metrics = self.agent.update(batch)

            if step % self.log_every == 0 or step == 1:
                self._log["step"].append(step)
                for k in ("v_loss", "q_loss", "actor_loss"):
                    self._log[k].append(metrics[k])
                print(f"  step {step:>7,}/{n_steps:,}  "
                      f"v={metrics['v_loss']:.4f}  "
                      f"q={metrics['q_loss']:.4f}  "
                      f"π={metrics['actor_loss']:.4f}")

        print(f"[IQL] Terminé — method={method}")
        return self._log

    def evaluate(self, n_episodes: int = 10, env=None) -> float:
        if env is None:
            print("[IQL] Pas d'environnement gym — score=0.0 (normal en test)")
            return 0.0

        returns = []
        for _ in range(n_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            done, ep_ret = False, 0.0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
                act   = self.agent.actor.act(obs_t).cpu().numpy()
                step  = env.step(act)
                if len(step) == 5:
                    obs, rew, term, trunc, _ = step
                    done = term or trunc
                else:
                    obs, rew, done, _ = step
                ep_ret += rew
            returns.append(ep_ret)

        raw   = sum(returns) / len(returns)
        score = normalized_score(self.env_name, raw)
        print(f"[IQL] raw={raw:.2f}  normalized={score:.2f}")
        return score

    def save(self, path: str) -> None:
        self.agent.save(path)

    def load(self, path: str) -> "IQLTrainer":
        self.agent.load(path)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide — python src/iql_trainer.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # CORRECTION 5 — seed pour reproductibilité
    torch.manual_seed(42)

    print("=" * 55)
    print("Test iql_trainer.py")
    print("=" * 55)

    cfg     = DATASET_CONFIGS[DEFAULT_DATASET]
    OBS_DIM = cfg["obs_dim"]
    ACT_DIM = cfg["act_dim"]
    N       = 10_000

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

    # ── Test train ────────────────────────────────────────────────────────
    print("\n── Test train() 1000 steps ────────────────────────────")
    trainer = IQLTrainer(
        obs_dim=OBS_DIM, act_dim=ACT_DIM,
        env_name=DEFAULT_DATASET, device=DEVICE, log_every=500,
    )
    log = trainer.train(buf, n_steps=1_000, method="baseline")
    assert len(log["step"]) > 0
    print(f"  ✓ train OK — {len(log['step'])} logs")

    # ── Test log_prob (CORRECTION 1) ──────────────────────────────────────
    print("\n── Test log_prob() actions dataset ───────────────────")
    obs_t = torch.randn(16, OBS_DIM).to(DEVICE)
    act_t = torch.randn(16, ACT_DIM).clamp(-1 + 1e-6, 1 - 1e-6).to(DEVICE)
    lp    = trainer.agent.actor.log_prob(obs_t, act_t)
    assert lp.shape == (16, 1), f"shape inattendu : {lp.shape}"
    assert not lp.isnan().any(), "log_prob contient des NaN"
    print(f"  ✓ log_prob OK — shape={tuple(lp.shape)}, pas de NaN")

    # ── Test evaluate ─────────────────────────────────────────────────────
    print("\n── Test evaluate() sans gym ───────────────────────────")
    score = trainer.evaluate(env=None)
    assert score == 0.0
    print(f"  ✓ evaluate OK")

    # ── Test save / load ──────────────────────────────────────────────────
    print("\n── Test save / load ───────────────────────────────────")
    trainer.save("checkpoints/iql_test.pt")
    trainer2 = IQLTrainer(OBS_DIM, ACT_DIM, DEFAULT_DATASET, device=DEVICE)
    trainer2.load("checkpoints/iql_test.pt")
    obs_t = torch.randn(OBS_DIM).to(DEVICE)
    act1  = trainer.agent.actor.act(obs_t)
    act2  = trainer2.agent.actor.act(obs_t)
    assert torch.allclose(act1, act2, atol=1e-5)
    print("  ✓ save/load OK")

    # ── Simulation 4 runs ─────────────────────────────────────────────────
    print("\n── Simulation 4 runs ──────────────────────────────────")
    results = {}
    for method in ["baseline", "vine", "mcts", "vae"]:
        t = IQLTrainer(OBS_DIM, ACT_DIM, DEFAULT_DATASET,
                       device=DEVICE, log_every=500)
        t.train(buf, n_steps=500, method=method)
        results[method] = t.evaluate(env=None)

    print(f"\n  {'method':<12} {'score':>8}")
    print(f"  {'-'*22}")
    for m, s in results.items():
        print(f"  {m:<12} {s:>8.2f}")

    print("\n" + "=" * 55)
    print("Tous les tests passent — iql_trainer.py prêt.")
    print("=" * 55)