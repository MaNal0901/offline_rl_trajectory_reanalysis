"""
interfaces.py — Contrats partagés Jour 1
=========================================
Ce fichier définit les signatures exactes que A et B respectent.
NE PAS modifier sans accord des deux coéquipiers.

Datasets retenus :
    - hopper-medium-v2
    - halfcheetah-medium-v2
    - walker2d-medium-v2

Hyperparamètres communs :
    OBS_DIM      : selon dataset (voir DATASET_CONFIGS)
    ACT_DIM      : selon dataset (voir DATASET_CONFIGS)
    BATCH_SIZE   : 256
    BUFFER_SIZE  : 1_000_000
    ROLLOUT_LEN  : 5      (longueur max des rollouts synthétiques)
    DEVICE       : "cuda" si disponible, sinon "cpu"

Clés standard du batch (alignées sur Minari / D4RL) :
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

BATCH_SIZE        = 256
BUFFER_SIZE       = 1_000_000
ROLLOUT_LEN       = 5
GAMMA             = 0.99
UNCERTAINTY_LIMIT = 0.5   # seuil unique partagé par Vine, MCTS, VAE
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

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

# Dataset par défaut pour les tests rapides
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

    Retourne un score en % (100 = niveau expert, 0 = niveau random).
    Utilisé dans evaluate.py pour comparer les 4 policies sur le même axe.
    """
    if env_name not in NORMALIZATION_SCORES:
        raise ValueError(
            f"env_name '{env_name}' inconnu. "
            f"Valeurs acceptées : {list(NORMALIZATION_SCORES.keys())}"
        )
    ref = NORMALIZATION_SCORES[env_name]
    return (raw_return - ref["random"]) / (ref["expert"] - ref["random"]) * 100


# ─────────────────────────────────────────────
# INTERFACE 1 — ReplayBuffer  (implémenté par A)
# ─────────────────────────────────────────────

class ReplayBufferInterface(ABC):
    """
    Contrat que data_loader.py doit respecter.
    B appelle uniquement .sample() et .add_batch() — rien d'autre.

    Toutes les clés du batch suivent REQUIRED_KEYS.
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
        Appelé par vine_augment, mcts_augment, vae_augment.

        batch doit contenir exactement REQUIRED_KEYS.
        Lève KeyError si une clé est absente ou non reconnue.

        Exemple d'appel depuis un augmenteur :
            buffer.add_batch({
                "observations"      : s,       # Tensor (N, obs_dim)
                "actions"           : a,       # Tensor (N, act_dim)
                "rewards"           : r,       # Tensor (N, 1)
                "next_observations" : s_next,  # Tensor (N, obs_dim)
                "terminals"         : d,       # Tensor (N, 1)
            })
        """
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """Nombre de transitions actuellement dans le buffer."""
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
        state  : torch.Tensor,  # (batch, obs_dim) ou (obs_dim,)
        action : torch.Tensor,  # (batch, act_dim) ou (act_dim,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prédit le prochain état et la récompense.
        Retourne : (next_state, reward)
            next_state : Tensor (batch, obs_dim)  float32
            reward     : Tensor (batch, 1)        float32

        Supporte batch ET single sample (auto-reshape interne).
        Tous les tenseurs retournés sont sur DEVICE.
        """
        ...

    @abstractmethod
    def uncertainty(
        self,
        state  : torch.Tensor,  # (batch, obs_dim)
        action : torch.Tensor,  # (batch, act_dim)
    ) -> torch.Tensor:          # (batch, 1) — variance de l'ensemble
        """
        Retourne l'incertitude épistémique du modèle (variance inter-membres).
        Utilisé par Vine, MCTS et VAE pour filtrer les transitions hors distribution.

        Seuil à appliquer : rejeter si uncertainty > UNCERTAINTY_LIMIT (= 0.5).
        Ce seuil est défini une seule fois dans ce fichier pour garantir
        une comparaison équitable entre les 3 méthodes.
        """
        ...

    @abstractmethod
    def train_model(
        self,
        buffer    : ReplayBufferInterface,
        n_epochs  : int = 50,
        batch_size: int = BATCH_SIZE,
    ) -> dict:
        """
        Entraîne le world model sur le buffer.
        Retourne : {"train_loss": float, "val_loss": float}
        """
        ...


# ─────────────────────────────────────────────
# INTERFACE 3 — Augmenteur  (implémenté par B)
# ─────────────────────────────────────────────

class AugmenterInterface(ABC):
    """
    Contrat commun pour vine_augment, mcts_augment, vae_augment.
    A appelle uniquement .augment() dans iql_trainer.
    """

    @abstractmethod
    def augment(
        self,
        buffer           : ReplayBufferInterface,
        world_model      : WorldModelInterface,
        n_new_transitions: int = 50_000,
    ) -> ReplayBufferInterface:
        """
        Génère n_new_transitions synthétiques via world_model.
        Retourne un NOUVEAU buffer contenant :
            - les transitions originales de buffer
            - les transitions synthétiques ajoutées
        Le buffer original n'est PAS modifié.

        Les transitions dont uncertainty > UNCERTAINTY_LIMIT
        doivent être rejetées avant l'ajout au nouveau buffer.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom de la méthode : 'VINE', 'MCTS', ou 'VAE'"""
        ...


# ─────────────────────────────────────────────
# VÉRIFICATION RAPIDE — python interfaces.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Vérification interfaces.py ===")
    print(f"Device            : {DEVICE}")
    print(f"Batch size        : {BATCH_SIZE}")
    print(f"Uncertainty limit : {UNCERTAINTY_LIMIT}")
    print(f"Datasets          : {list(DATASET_CONFIGS.keys())}")
    print(f"Default dataset   : {DEFAULT_DATASET}")
    print()
    print("Clés standard du batch :")
    for k in sorted(REQUIRED_KEYS):
        print(f"  - {k}")
    print()
    print("Interfaces définies :")
    print("  ✓ ReplayBufferInterface — .sample() → dict, .add_batch(dict), .size, .obs_dim, .act_dim")
    print("  ✓ WorldModelInterface   — .predict(), .uncertainty(), .train_model()")
    print("  ✓ AugmenterInterface    — .augment(), .name")
    print()
    print("Test normalized_score :")
    for env in DATASET_CONFIGS:
        ref = NORMALIZATION_SCORES[env]
        score_random = normalized_score(env, ref["random"])
        score_expert = normalized_score(env, ref["expert"])
        print(f"  {env:<30}  random={score_random:.1f}%  expert={score_expert:.1f}%")
    print()
    print("Contrats OK — A et B peuvent commencer.")
