"""
critic_guided.py — Critic-Guided Trajectory Reanalysis
=======================================================
Pipeline :
    1. IQL warm-up sur D → critic apprend Q(s,a) et V(s)
    2. Critic calcule A(s,a) sur tout D → sélectionne top-K trajectoires
    3. Meilleure méthode réanalyse ces trajectoires → D′
    4. IQL final sur D′ → score

Usage :
    python src/critic_guided.py --env hopper-medium-v2 --best_method vine
"""

from __future__ import annotations
import argparse
import json
import os
import torch

from interfaces  import DATASET_CONFIGS, DEVICE, normalized_score
from data_loader import load_buffer, ReplayBuffer
from world_model import build_world_model
from iql_trainer import IQLTrainer, IQLAgent


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Warm-up IQL pour entraîner le critic
# ─────────────────────────────────────────────────────────────────────────────

def warmup_critic(
    buffer   : ReplayBuffer,
    obs_dim  : int,
    act_dim  : int,
    n_steps  : int = 5_000,
    device   : str = DEVICE,
) -> IQLAgent:
    """
    Entraîne IQL quelques steps sur D pour obtenir un critic Q(s,a) et V(s).
    Retourne l'agent IQL avec le critic entraîné.
    """
    print(f"\n[Critic warm-up] {n_steps:,} steps sur D original...")
    agent = IQLAgent(obs_dim=obs_dim, act_dim=act_dim, device=device)

    for step in range(1, n_steps + 1):
        batch = buffer.sample(256)
        agent.update(batch)
        if step % 1000 == 0:
            print(f"  step {step:>5,}/{n_steps:,}")

    print(f"[Critic warm-up] Terminé — critic Q(s,a) et V(s) prêts")
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — Sélection des trajectoires par le critic
# ─────────────────────────────────────────────────────────────────────────────

def select_by_advantage(
    buffer  : ReplayBuffer,
    agent   : IQLAgent,
    n_points: int,
    device  : str = DEVICE,
) -> tuple:
    """
    Calcule A(s,a) = Q(s,a) - V(s) sur tout D.
    Retourne les indices des n_points états avec A le plus élevé.

    Pourquoi A élevé → mérite réanalyse :
        A(s,a) > 0 = cette action est meilleure que la moyenne
                   = la trajectoire est sous-exploitée
                   = le critic pense qu'on peut faire mieux ici
    """
    print(f"\n[Critic selection] Calcul A(s,a) sur {buffer.size:,} transitions...")

    all_data = buffer.get_all()
    obs      = all_data["observations"]
    act      = all_data["actions"]

    # Traiter par batch pour éviter OOM sur 1M transitions
    batch_size  = 4096
    advantages  = []

    for i in range(0, buffer.size, batch_size):
        obs_b = obs[i:i+batch_size].to(device)
        act_b = act[i:i+batch_size].to(device)

        with torch.no_grad():
            q   = agent.qnet_target.q_min(obs_b, act_b)  # Q(s,a)
            v   = agent.vnet(obs_b)                       # V(s)
            adv = (q - v).squeeze(1)                      # A(s,a)

        advantages.append(adv.cpu())

    advantages = torch.cat(advantages)  # (N,)

    # Stats pour le rapport
    print(f"  A(s,a) min  = {advantages.min():.4f}")
    print(f"  A(s,a) max  = {advantages.max():.4f}")
    print(f"  A(s,a) mean = {advantages.mean():.4f}")
    print(f"  A(s,a) > 0  = {(advantages > 0).sum().item():,} "
          f"({(advantages > 0).float().mean()*100:.1f}%)")

    # Sélectionner les top-K états
    _, top_idx = torch.topk(advantages, k=min(n_points, buffer.size))

    print(f"\n[Critic selection] Top-{n_points:,} états sélectionnés")
    print(f"  A moyen (sélectionnés) = "
          f"{advantages[top_idx].mean():.4f}  "
          f"(vs {advantages.mean():.4f} pour tout D)")

    return top_idx.cpu(), advantages


# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 — Réanalyse avec la meilleure méthode
# ─────────────────────────────────────────────────────────────────────────────

def reanalyze_selected(
    buffer     : ReplayBuffer,
    wm,
    agent      : IQLAgent,
    best_method: str,
    top_idx    : torch.Tensor,
    n_new      : int = 50_000,
    device     : str = DEVICE,
) -> ReplayBuffer:
    """
    Réanalyse SEULEMENT les trajectoires sélectionnées par le critic.
    Utilise la meilleure méthode parmi Vine/MCTS/VAE.
    """
    print(f"\n[Reanalysis] Méthode = {best_method.upper()}  "
          f"sur {len(top_idx):,} états sélectionnés")

    # Extraire les états sélectionnés depuis D
    all_data     = buffer.get_all()
    selected_obs = all_data["observations"][top_idx]
    selected_act = all_data["actions"][top_idx]
    selected_rew = all_data["rewards"][top_idx]
    selected_next= all_data["next_observations"][top_idx]
    selected_done= all_data["terminals"][top_idx]

    # Créer un mini-buffer avec seulement ces états
    mini_buffer = ReplayBuffer(device=device)
    mini_buffer._data = {
        "observations"      : selected_obs,
        "actions"           : selected_act,
        "rewards"           : selected_rew,
        "next_observations" : selected_next,
        "terminals"         : selected_done,
    }
    mini_buffer._size    = len(top_idx)
    mini_buffer._obs_dim = buffer.obs_dim
    mini_buffer._act_dim = buffer.act_dim

    print(f"  Mini-buffer : {mini_buffer.size:,} transitions sélectionnées")

    # Appliquer la meilleure méthode sur ce mini-buffer
    if best_method == "vine":
        from vine_augment import VineAugmenter
        augmenter = VineAugmenter(rollout_horizon=5, n_actions=5, device=device)

    elif best_method == "mcts":
        from mcts_augment import MCTSAugmenter
        augmenter = MCTSAugmenter(
            n_simulations=20, rollout_depth=5, n_actions=5, device=device
        )

    elif best_method == "vae":
        from vae_augment import VAEAugmenter
        augmenter = VAEAugmenter(latent_dim=32, vae_epochs=30, device=device)

    else:
        raise ValueError(f"Méthode inconnue : {best_method}")

    # Augmenter le mini-buffer
    mini_augmented = augmenter.augment(mini_buffer, wm, n_new_transitions=n_new)

    # Construire D′ = D original + nouvelles transitions synthétiques
    new_buffer     = buffer.clone()
    synth_size     = mini_augmented.size - mini_buffer.size

    if synth_size > 0:
        synth_data = mini_augmented.get_all()
        # Prendre seulement les transitions synthétiques (pas les originales)
        synth_batch = {
            k: v[mini_buffer.size:] for k, v in synth_data.items()
        }
        new_buffer.add_batch(synth_batch)
        print(f"  Transitions synthétiques ajoutées à D : {synth_size:,}")
    else:
        print("  [WARN] Aucune transition synthétique générée")

    print(f"  D′ final : {new_buffer.size:,} transitions")
    return new_buffer


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def run_critic_guided(
    env_name   : str,
    best_method: str,
    warmup_steps: int = 5_000,
    final_steps : int = 50_000,
    n_selected  : int = 10_000,
    n_new       : int = 50_000,
    seed        : int = 42,
) -> dict:

    torch.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"CRITIC-GUIDED REANALYSIS")
    print(f"env={env_name}  method={best_method.upper()}")
    print(f"warmup={warmup_steps:,}  final={final_steps:,}")
    print(f"{'='*60}")

    cfg     = DATASET_CONFIGS[env_name]
    obs_dim = cfg["obs_dim"]
    act_dim = cfg["act_dim"]

    # ── 1. Charger dataset ────────────────────────────────────────────────
    print("\n[1/5] Chargement dataset...")
    buffer = load_buffer(env_name)

    # ── 2. World model ────────────────────────────────────────────────────
    print("\n[2/5] World model...")
    wm_path = f"checkpoints/wm_{env_name}.pt"
    wm = build_world_model(env_name)
    if os.path.exists(wm_path):
        wm.load(wm_path)
        print(f"  Checkpoint chargé")
    else:
        wm.train_model(buffer, n_epochs=50)
        wm.save(wm_path)

    # ── 3. Warm-up critic ─────────────────────────────────────────────────
    print("\n[3/5] Warm-up critic...")
    agent = warmup_critic(buffer, obs_dim, act_dim,
                          n_steps=warmup_steps, device=DEVICE)

    # ── 4. Sélection critic-guided ────────────────────────────────────────
    print("\n[4/5] Sélection par le critic...")
    top_idx, advantages = select_by_advantage(
        buffer, agent, n_points=n_selected, device=DEVICE
    )

    # ── 5. Réanalyse des trajectoires sélectionnées ───────────────────────
    print("\n[5/5] Réanalyse...")
    buf_prime = reanalyze_selected(
        buffer, wm, agent, best_method,
        top_idx, n_new=n_new, device=DEVICE
    )

    # ── 6. IQL final sur D′ ───────────────────────────────────────────────
    print(f"\n[IQL final] {final_steps:,} steps sur D′...")
    trainer = IQLTrainer(
        obs_dim=obs_dim, act_dim=act_dim,
        env_name=env_name, device=DEVICE,
    )
    log = trainer.train(buf_prime, n_steps=final_steps,
                        method=f"critic_guided_{best_method}")
    trainer.save(
        f"checkpoints/iql_critic_guided_{best_method}_{env_name}.pt"
    )

    final_q_loss = log["q_loss"][-1] if log["q_loss"] else 0.0
    score_proxy  = round(-final_q_loss, 4)

    # ── 7. Résultats ──────────────────────────────────────────────────────
    results = {
        "env"           : env_name,
        "best_method"   : best_method,
        "n_selected"    : n_selected,
        "n_new"         : buf_prime.size - buffer.size,
        "buffer_size_D" : buffer.size,
        "buffer_size_Dp": buf_prime.size,
        "final_q_loss"  : round(final_q_loss, 4),
        "score_proxy"   : score_proxy,
        "adv_mean_all"  : round(advantages.mean().item(), 4),
        "adv_mean_sel"  : round(advantages[top_idx].mean().item(), 4),
    }

    # Sauvegarder
    os.makedirs("results", exist_ok=True)
    out = f"results/critic_guided_{best_method}_{env_name}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RÉSULTATS — Critic-Guided {best_method.upper()}")
    print(f"{'='*60}")
    print(f"  États sélectionnés    : {n_selected:,}")
    print(f"  A moyen (tout D)      : {results['adv_mean_all']:.4f}")
    print(f"  A moyen (sélectionnés): {results['adv_mean_sel']:.4f}  ← plus élevé ✓")
    print(f"  Transitions ajoutées  : {results['n_new']:,}")
    print(f"  Q-loss final          : {final_q_loss:.4f}")
    print(f"  Score proxy           : {score_proxy:.4f}")
    print(f"  Sauvegardé → {out}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        default="hopper-medium-v2",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--best_method",
                        default="vine",
                        choices=["vine", "mcts", "vae"])
    parser.add_argument("--warmup",  type=int, default=5_000)
    parser.add_argument("--steps",   type=int, default=50_000)
    parser.add_argument("--n_sel",   type=int, default=10_000)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    run_critic_guided(
        env_name    = args.env,
        best_method = args.best_method,
        warmup_steps= args.warmup,
        final_steps = args.steps,
        n_selected  = args.n_sel,
        seed        = args.seed,
    )