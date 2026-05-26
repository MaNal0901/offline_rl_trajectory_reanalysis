"""
data_loader.py — Implémentation de ReplayBuffer
================================================
Implémenté par : A
Dépend de      : interfaces.py

Lit les fichiers HDF5 D4RL directement (sans gym/d4rl installé).
Implémente ReplayBufferInterface — B appelle uniquement .sample() et .add_batch()

Usage rapide :
    python src/data_loader.py
"""

from __future__ import annotations
from typing import Dict
import os

import h5py
import numpy as np
import torch

from interfaces import (
    ReplayBufferInterface,
    DATASET_CONFIGS,
    DEFAULT_DATASET,
    REQUIRED_KEYS,
    DEVICE,
    BATCH_SIZE,
)


class ReplayBuffer(ReplayBufferInterface):
    """
    Buffer offline — chargé depuis HDF5, jamais collecté en ligne.

    Structure interne : dict de Tensors sur CPU.
    Les batchs retournés par .sample() sont envoyés sur DEVICE.
    """

    def __init__(self, device: str = DEVICE):
        self._device  = device
        self._data: Dict[str, torch.Tensor] = {}
        self._size    = 0
        self._obs_dim = 0
        self._act_dim = 0

    # ── Chargement ────────────────────────────────────────────────────────────

    def load_hdf5(self, hdf5_path: str) -> "ReplayBuffer":
        """
        Charge un fichier HDF5 D4RL dans le buffer.
        Gère les noms de clés D4RL standards et leurs variantes.
        Retourne self pour chaining.
        """
        print(f"[ReplayBuffer] Chargement : {hdf5_path}")

        with h5py.File(hdf5_path, "r") as f:
            obs      = self._read_key(f, ["observations", "obs"])
            act      = self._read_key(f, ["actions", "act"])
            rew      = self._read_key(f, ["rewards", "reward"])
            next_obs = self._read_key(f, ["next_observations", "next_obs"])
            terminals= self._read_key(f, ["terminals", "dones", "done"])

        obs       = torch.tensor(obs,       dtype=torch.float32)
        act       = torch.tensor(act,       dtype=torch.float32)
        rew       = torch.tensor(rew,       dtype=torch.float32)
        next_obs  = torch.tensor(next_obs,  dtype=torch.float32)
        terminals = torch.tensor(terminals, dtype=torch.float32)

        # forcer shape (N, 1) pour rewards et terminals
        if rew.ndim == 1:
            rew = rew.unsqueeze(1)
        if terminals.ndim == 1:
            terminals = terminals.unsqueeze(1)

        self._data = {
            "observations"      : obs,
            "actions"           : act,
            "rewards"           : rew,
            "next_observations" : next_obs,
            "terminals"         : terminals,
        }

        self._size    = obs.shape[0]
        self._obs_dim = obs.shape[1]
        self._act_dim = act.shape[1]

        self._print_stats(hdf5_path)
        return self

    @staticmethod
    def _read_key(f: h5py.File, candidates: list) -> np.ndarray:
        """Essaie plusieurs noms de clés — retourne le premier trouvé."""
        for key in candidates:
            if key in f:
                return f[key][:]
        raise KeyError(
            f"Aucune clé trouvée parmi {candidates}. "
            f"Clés disponibles : {list(f.keys())}"
        )

    def _print_stats(self, path: str) -> None:
        print(f"  ✓ {self._size:>10,} transitions")
        print(f"  ✓ obs_dim  = {self._obs_dim}")
        print(f"  ✓ act_dim  = {self._act_dim}")
        print(f"  ✓ rewards  : min={self._data['rewards'].min():.3f}  "
              f"max={self._data['rewards'].max():.3f}  "
              f"mean={self._data['rewards'].mean():.3f}")
        print(f"  ✓ épisodes terminés : {int(self._data['terminals'].sum().item())}")

    # ── Interface obligatoire ─────────────────────────────────────────────────

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Retourne un batch aléatoire sur DEVICE.
        Lève ValueError si batch_size > self.size.
        """
        if self._size == 0:
            raise RuntimeError("Buffer vide — appelle load_hdf5() d'abord.")
        if batch_size > self._size:
            raise ValueError(
                f"batch_size={batch_size} > buffer.size={self._size}"
            )

        # ── FIX 1 : index sur CPU — données stockées sur CPU ─────────────
        idx = torch.randint(0, self._size, (batch_size,), device="cpu")

        return {
            k: v[idx].to(self._device)
            for k, v in self._data.items()
        }

    def add_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Ajoute des transitions synthétiques au buffer.
        Appelé par vine_augment, mcts_augment, vae_augment.
        Lève KeyError si clés incorrectes.
        Lève ValueError si dimensions incompatibles.
        """
        # ── validation clés ───────────────────────────────────────────────
        missing = REQUIRED_KEYS - set(batch.keys())
        extra   = set(batch.keys()) - REQUIRED_KEYS
        if missing:
            raise KeyError(f"Clés manquantes dans batch : {missing}")
        if extra:
            raise KeyError(f"Clés inconnues dans batch : {extra}")

        # ── FIX 2 : validation dimensions ─────────────────────────────────
        if self._size > 0:
            for k in ("observations", "next_observations"):
                if batch[k].shape[1] != self._obs_dim:
                    raise ValueError(
                        f"{k} : attendu obs_dim={self._obs_dim}, "
                        f"reçu {batch[k].shape[1]}"
                    )
            if batch["actions"].shape[1] != self._act_dim:
                raise ValueError(
                    f"actions : attendu act_dim={self._act_dim}, "
                    f"reçu {batch['actions'].shape[1]}"
                )

        n_new = batch["observations"].shape[0]

        # ── concaténer sur CPU ────────────────────────────────────────────
        for k in REQUIRED_KEYS:
            tensor_new = batch[k].cpu().float()

            if k in ("rewards", "terminals") and tensor_new.ndim == 1:
                tensor_new = tensor_new.unsqueeze(1)

            if self._size == 0:
                self._data[k] = tensor_new
            else:
                self._data[k] = torch.cat([self._data[k], tensor_new], dim=0)

        self._size += n_new
        print(f"[ReplayBuffer] +{n_new:,} transitions → total {self._size:,}")

    @property
    def size(self) -> int:
        return self._size

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def act_dim(self) -> int:
        return self._act_dim

    @property
    def device(self) -> str:
        return self._device

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def clone(self) -> "ReplayBuffer":
        """
        Retourne une copie indépendante du buffer.
        Les augmenteurs font buffer.clone() avant add_batch()
        pour ne pas modifier le buffer original D.
        """
        new = ReplayBuffer(device=self._device)
        new._data    = {k: v.clone() for k, v in self._data.items()}
        new._size    = self._size
        new._obs_dim = self._obs_dim
        new._act_dim = self._act_dim
        return new

    def get_all(self) -> Dict[str, torch.Tensor]:
        """
        Retourne toutes les transitions sur CPU.
        Utilisé par world_model.train_model() pour accéder à tout D.
        """
        return {k: v.clone() for k, v in self._data.items()}

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer("
            f"size={self._size:,}, "
            f"obs_dim={self._obs_dim}, "
            f"act_dim={self._act_dim}, "
            f"device={self._device})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fonction utilitaire — chargement rapide par nom d'env
# ─────────────────────────────────────────────────────────────────────────────

def load_buffer(env_name: str, device: str = DEVICE) -> ReplayBuffer:
    """
    Charge le buffer D4RL pour un environnement donné.

    Usage :
        buffer = load_buffer("hopper-medium-v2")
        batch  = buffer.sample(256)
    """
    if env_name not in DATASET_CONFIGS:
        raise KeyError(
            f"'{env_name}' inconnu. "
            f"Disponibles : {list(DATASET_CONFIGS.keys())}"
        )
    cfg = DATASET_CONFIGS[env_name]
    buf = ReplayBuffer(device=device)
    buf.load_hdf5(cfg["hdf5_path"])
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide — python src/data_loader.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("Test data_loader.py")
    print("=" * 55)

    cfg       = DATASET_CONFIGS[DEFAULT_DATASET]
    hdf5_path = cfg["hdf5_path"]

    if not os.path.exists(hdf5_path):
        print(f"\n[WARN] HDF5 absent : {hdf5_path}")
        print("Lance d'abord : bash setup_data.sh")
        print("\nFallback : données synthétiques...\n")
        buf = ReplayBuffer(device=DEVICE)
        n   = 10_000
        buf._data = {
            "observations"      : torch.randn(n, cfg["obs_dim"]),
            "actions"           : torch.randn(n, cfg["act_dim"]).clamp(-1, 1),
            "rewards"           : torch.randn(n, 1) * 0.5 + 0.5,
            "next_observations" : torch.randn(n, cfg["obs_dim"]),
            "terminals"         : (torch.rand(n, 1) < 0.05).float(),
        }
        buf._size    = n
        buf._obs_dim = cfg["obs_dim"]
        buf._act_dim = cfg["act_dim"]
    else:
        buf = load_buffer(DEFAULT_DATASET)

    print(f"\n{buf}")

    # ── Test sample() ─────────────────────────────────────────────────────
    print("\n── Test sample() ──────────────────────────────────────")
    batch = buf.sample(BATCH_SIZE)
    assert set(batch.keys()) == REQUIRED_KEYS, "ERREUR clés"
    for k, v in batch.items():
        print(f"  {k:<22} shape={tuple(v.shape)}  dtype={v.dtype}")
    print("  ✓ OK")

    # ── Test clone() ──────────────────────────────────────────────────────
    print("\n── Test clone() ───────────────────────────────────────")
    buf2 = buf.clone()
    assert buf2.size == buf.size
    buf2._data["rewards"][0] = 9999
    assert buf._data["rewards"][0] != 9999   # indépendance vérifiée
    print(f"  ✓ OK — copie indépendante, taille {buf2.size:,}")

    # ── Test add_batch() ──────────────────────────────────────────────────
    print("\n── Test add_batch() ───────────────────────────────────")
    n_synth = 500
    synth = {
        "observations"      : torch.randn(n_synth, buf.obs_dim),
        "actions"           : torch.randn(n_synth, buf.act_dim).clamp(-1, 1),
        "rewards"           : torch.randn(n_synth, 1),
        "next_observations" : torch.randn(n_synth, buf.obs_dim),
        "terminals"         : torch.zeros(n_synth, 1),
    }
    size_avant = buf2.size
    buf2.add_batch(synth)
    assert buf2.size == size_avant + n_synth
    print(f"  ✓ OK — {size_avant:,} + {n_synth} = {buf2.size:,}")

    # ── Test validation dims (FIX 2) ──────────────────────────────────────
    print("\n── Test validation dims ───────────────────────────────")
    try:
        bad = {k: torch.randn(10, 99) if "obs" in k
               else torch.randn(10, buf.act_dim) if k == "actions"
               else torch.zeros(10, 1)
               for k in REQUIRED_KEYS}
        buf2.add_batch(bad)
        print("  ERREUR : aurait dû lever ValueError")
    except ValueError as e:
        print(f"  ✓ ValueError correctement levée : {e}")

    # ── Test validation clés manquantes ───────────────────────────────────
    print("\n── Test clés manquantes ───────────────────────────────")
    try:
        buf2.add_batch({"observations": torch.randn(10, buf.obs_dim)})
        print("  ERREUR : aurait dû lever KeyError")
    except KeyError as e:
        print(f"  ✓ KeyError correctement levée : {e}")

    print("\n" + "=" * 55)
    print("Tous les tests passent — data_loader.py prêt.")
    print("Binôme peut utiliser :")
    print("  from data_loader import ReplayBuffer, load_buffer")
    print("=" * 55)