"""
interfaces.py — Contrats partagés Jour 1
=========================================
Ce fichier définit les signatures exactes que A et B respectent.
NE PAS modifier sans accord des deux coéquipiers.

Datasets retenus :
    - hopper-medium-v2
    - halfcheetah-medium-v2
    - walker2d-medium-v2

Clés standard du batch :
    "observations"      : Tensor (batch, obs_dim)  float32
    "actions"           : Tensor (batch, act_dim)  float32
    "rewards"           : Tensor (batch, 1)        float32
    "next_observations" : Tensor (batch, obs_dim)  float32
    "terminals"         : Tensor (batch, 1)        float32
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch

# ─────────────────────────────────────────────
# HYPERPARAMÈTRES COMMUNS
# ─────────────────────────────────────────────

BATCH_SIZE             = 256
BUFFER_SIZE            = 1_000_000
ROLLOUT_LEN            = 5
GAMMA                  = 0.99
UNCERTAINTY_PERCENTILE = 90.0   # percentile calibré sur D — partagé par Vine, MCTS, VAE
DEVICE                 = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# CLÉS STANDARD DU BATCH
# ─────────────────────────────────────────────

REQUIRED_KEYS: frozenset[str] = frozenset({
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminals",
})

# ─────────────────────────────────────────────
# DATASETS — configs par environnement
# ─────────────────────────────────────────────

DATASET_CONFIGS: Dict[str, dict] = {
    "hopper-medium-v2": {
        "hdf5_path"         : "data/hopper-medium-v2.hdf5",
        "obs_dim"           : 11,
        "act_dim"           : 3,
        "max_episode_steps" : 1000,
    },
    "halfcheetah-medium-v2": {
        "hdf5_path"         : "data/halfcheetah-medium-v2.hdf5",
        "obs_dim"           : 17,
        "act_dim"           : 6,
        "max_episode_steps" : 1000,
    },
    "walker2d-medium-v2": {
        "hdf5_path"         : "data/walker2d-medium-v2.hdf5",
        "obs_dim"           : 17,
        "act_dim"           : 6,
        "max_episode_steps" : 1000,
    },
}

DEFAULT_DATASET = "hopper-medium-v2"

# ─────────────────────────────────────────────
# SCORES DE NORMALISATION D4RL
# ─────────────────────────────────────────────

NORMALIZATION_SCORES: Dict[str, Dict[str, float]] = {
    "hopper-medium-v2": {
        "random" : 20.27,
        "expert" : 3234.3,
    },
    "halfcheetah-medium-v2": {
        "random" : -280.18,
        "expert" : 12135.0,
    },
    "walker2d-medium-v2": {
        "random" : 1.63,
        "expert" : 4592.3,
    },
}


def normalized_score(env_name: str, raw_return: float) -> float:
    """
    Normalise un return brut selon la convention D4RL :
        score = (raw - random) / (expert - random) * 100
    Retourne un score en % (100 = expert, 0 = random).
    """
    if env_name not in NORMALIZATION_SCORES:
        raise ValueError(
            f"env_name '{env_name}' inconnu. "
            f"Valeurs acceptées : {list(NORMALIZATION_SCORES.keys())}"
        )
    ref = NORMALIZATION_SCORES[env_name]
    return (raw_return - ref["random"]) / (ref["expert"] - ref["random"]) * 100


# ─────────────────────────────────────────────
# SEUIL D'INCERTITUDE — fonction partagée
# ─────────────────────────────────────────────

def calibrate_threshold(
    world_model,
    buffer,
    percentile : float = UNCERTAINTY_PERCENTILE,
    n_samples  : int   = 2000,
) -> float:
    """
    Calibre le seuil d'incertitude sur le percentile p90 de D.

    Logique :
        1. Échantillonne n_samples transitions réelles depuis D
        2. Calcule l'incertitude du world model sur ces transitions
        3. Retourne le percentile 90
           → 90% des transitions réelles passent le filtre
           → 10% les plus incertaines sont rejetées

    Appelé une fois au début de chaque augment().
    Même résultat pour les 3 méthodes si même buffer D.

    Exemple :
        uncertainty sur D : max=0.054
        seuil p90         : ~0.021
        → rejette ce qui est plus incertain que 90% des données réelles
    """
    n_samples = min(n_samples, buffer.size)
    batch     = buffer.sample(n_samples)

    with torch.no_grad():
        u = world_model.uncertainty(
            batch["observations"].to(DEVICE),
            batch["actions"].to(DEVICE),
        )  # (n_samples, 1)

    threshold = torch.quantile(u.squeeze(), percentile / 100.0).item()

    print(f"  [calibrate_threshold] p{int(percentile)} = {threshold:.4f}  "
          f"(mean={u.mean():.4f}  max={u.max():.4f})")

    return threshold


# ─────────────────────────────────────────────
# INTERFACE 1 — ReplayBuffer  (implémenté par A)
# ─────────────────────────────────────────────

class ReplayBufferInterface(ABC):
    """
    Contrat que data_loader.py doit respecter.
    B appelle uniquement .sample() et .add_batch() — rien d'autre.
    """

    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Retourne un batch aléatoire normalisé sous forme de dict :
        {
            "observations"      : Tensor (batch, obs_dim)  float32
            "actions"           : Tensor (batch, act_dim)  float32
            "rewards"           : Tensor (batch, 1)        float32
            "next_observations" : Tensor (batch, obs_dim)  float32
            "terminals"         : Tensor (batch, 1)        float32
        }
        Tous les tenseurs sont sur DEVICE et dtype=float32.
        Lève ValueError si batch_size > self.size.
        """
        ...

    @abstractmethod
    def add_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Ajoute des transitions synthétiques au buffer.
        batch doit contenir exactement REQUIRED_KEYS.
        Lève KeyError si une clé est absente ou non reconnue.
        """
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        ...

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def act_dim(self) -> int:
        ...


# ─────────────────────────────────────────────
# INTERFACE 2 — WorldModel  (implémenté par A)
# ─────────────────────────────────────────────

class WorldModelInterface(ABC):
    """
    Contrat que world_model.py doit respecter.
    B appelle uniquement .predict() et .uncertainty() — rien d'autre.
    """

    @abstractmethod
    def predict(
        self,
        state  : torch.Tensor,
        action : torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne : (next_state, reward)
            next_state : Tensor (batch, obs_dim)
            reward     : Tensor (batch, 1)
        Supporte batch ET single sample (auto-reshape interne).
        """
        ...

    @abstractmethod
    def uncertainty(
        self,
        state  : torch.Tensor,
        action : torch.Tensor,
    ) -> torch.Tensor:
        """
        Retourne la variance inter-membres : Tensor (batch, 1).
        Utilisé par calibrate_threshold() et les 3 augmenteurs.
        """
        ...

    @abstractmethod
    def train_model(
        self,
        buffer    : ReplayBufferInterface,
        n_epochs  : int = 50,
        batch_size: int = BATCH_SIZE,
    ) -> dict:
        """Retourne : {"train_loss": float, "val_loss": float}"""
        ...


# ─────────────────────────────────────────────
# INTERFACE 3 — Augmenteur  (implémenté par B)
# ─────────────────────────────────────────────

class AugmenterInterface(ABC):
    """
    Contrat commun pour vine_augment, mcts_augment, vae_augment.
    A appelle uniquement .augment() dans run_all.py.
    """

    @abstractmethod
    def augment(
        self,
        buffer           : ReplayBufferInterface,
        world_model      : WorldModelInterface,
        n_new_transitions: int = 50_000,
    ) -> ReplayBufferInterface:
        """
        Génère n_new_transitions synthétiques.
        Retourne un NOUVEAU buffer = D original + transitions synthétiques.
        Le buffer original n'est PAS modifié.

        Seuil : appeler calibrate_threshold(world_model, buffer)
        au début de augment() — même valeur pour les 3 méthodes.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """'VINE', 'MCTS', ou 'VAE'"""
        ...


# ─────────────────────────────────────────────
# VÉRIFICATION RAPIDE — python src/interfaces.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Vérification interfaces.py ===")
    print(f"Device                 : {DEVICE}")
    print(f"Batch size             : {BATCH_SIZE}")
    print(f"Uncertainty percentile : {UNCERTAINTY_PERCENTILE}")
    print(f"Datasets               : {list(DATASET_CONFIGS.keys())}")
    print(f"Default dataset        : {DEFAULT_DATASET}")
    print()
    print("Clés standard du batch :")
    for k in sorted(REQUIRED_KEYS):
        print(f"  - {k}")
    print()
    print("Interfaces définies :")
    print("  ✓ ReplayBufferInterface — .sample() .add_batch() .size .obs_dim .act_dim")
    print("  ✓ WorldModelInterface   — .predict() .uncertainty() .train_model()")
    print("  ✓ AugmenterInterface    — .augment() .name")
    print("  ✓ calibrate_threshold() — partagé par Vine, MCTS, VAE")
    print()
    print("Test normalized_score :")
    for env in DATASET_CONFIGS:
        ref = NORMALIZATION_SCORES[env]
        s_r = normalized_score(env, ref["random"])
        s_e = normalized_score(env, ref["expert"])
        print(f"  {env:<30}  random={s_r:.1f}%  expert={s_e:.1f}%")
    print()
    print("Contrats OK — A et B peuvent commencer.")