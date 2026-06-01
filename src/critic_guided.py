"""
critic_guided.py — Critic-Guided Trajectory Reanalysis
=======================================================
Phase 2 du projet :
    1. IQL warm-up sur D → critic Q(s,a) et V(s)
    2. Critic calcule A(s,a) → sélectionne top-K trajectoires ciblées
    3. MCTS réanalyse SEULEMENT ces trajectoires
    4. IQL final sur D′ → comparaison vs random selection

Usage :
    python src/critic_guided.py --env hopper-medium-v2 --steps 5000
"""

from __future__ import annotations
import argparse
import json
import os
import torch

from interfaces   import DATASET_CONFIGS, DEVICE
from data_loader  import load_buffer, ReplayBuffer
from world_model  import build_world_model
from mcts_augment import MCTSAugmenter
from iql_trainer  import IQLTrainer, IQLAgent


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Warm-up IQL → critic Q(s,a) et V(s)
# ─────────────────────────────────────────────────────────────────────────────

def warmup_critic(
    buffer  : ReplayBuffer,
    obs_dim : int,
    act_dim : int,
    n_steps : int = 5_000,
    device  : str = DEVICE,
) -> IQLAgent:
    print(f"\n[1] Warm-up critic — {n_steps:,} steps...")
    agent = IQLAgent(obs_dim=obs_dim, act_dim=act_dim, device=device)

    for step in range(1, n_steps + 1):
        batch = buffer.sample(256)
        agent.update(batch)
        if step % 1_000 == 0:
            print(f"  step {step:>5,}/{n_steps:,}")

    print("[1] Critic prêt — Q(s,a) et V(s) disponibles")
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — Sélection par advantage A(s,a) = Q(s,a) - V(s)
# ─────────────────────────────────────────────────────────────────────────────

def select_by_advantage(
    buffer   : ReplayBuffer,
    agent    : IQLAgent,
    n_points : int,
    device   : str = DEVICE,
) -> tuple:
    """
    Calcule A(s,a) sur tout D.
    Sélectionne les n_points états avec A le plus élevé.

    A(s,a) > 0 → action meilleure que la moyenne
             → trajectoire sous-exploitée → mérite réanalyse
    """
    print(f"\n[2] Sélection critic-guided — top {n_points:,} / {buffer.size:,}...")

    all_data   = buffer.get_all()
    obs        = all_data["observations"]
    act        = all_data["actions"]
    batch_size = 4096
    advantages = []

    for i in range(0, buffer.size, batch_size):
        obs_b = obs[i:i+batch_size].to(device)
        act_b = act[i:i+batch_size].to(device)
        with torch.no_grad():
            q   = agent.qnet_target.q_min(obs_b, act_b)
            v   = agent.vnet(obs_b)
            adv = (q - v).squeeze(1)
        advantages.append(adv.cpu())

    advantages = torch.cat(advantages)

    print(f"  A(s,a) mean = {advantages.mean():.4f}")
    print(f"  A(s,a) max  = {advantages.max():.4f}")
    print(f"  A > 0       = {(advantages > 0).float().mean()*100:.1f}%")

    _, top_idx = torch.topk(advantages, k=min(n_points, buffer.size))
    adv_sel    = advantages[top_idx].mean().item()

    print(f"  A moyen sélectionnés = {adv_sel:.4f}  "
          f"(vs {advantages.mean():.4f} tout D)  ← doit être plus élevé ✓")

    return top_idx.cpu(), advantages


# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 — MCTS sur trajectoires ciblées seulement
# ─────────────────────────────────────────────────────────────────────────────

def reanalyze_targeted(
    buffer  : ReplayBuffer,
    wm,
    top_idx : torch.Tensor,
    n_new   : int = 50_000,
    device  : str = DEVICE,
) -> tuple:

    print(f"\n[3] MCTS sur {len(top_idx):,} états ciblés...")

    all_data    = buffer.get_all()

    # ── Créer le mini-buffer des états sélectionnés ───────────────────────
    mini_buffer = ReplayBuffer(device=device)
    mini_buffer.add_batch({k: v[top_idx] for k, v in all_data.items()})

    # FIX — add_batch ne set pas _obs_dim/_act_dim, on les copie manuellement
    mini_buffer._obs_dim = buffer.obs_dim
    mini_buffer._act_dim = buffer.act_dim

    print(f"  Mini-buffer : {mini_buffer.size:,} transitions ciblées "
          f"(obs_dim={mini_buffer.obs_dim}, act_dim={mini_buffer.act_dim})")

    # ── Limiter n_new à la capacité réelle du mini-buffer ─────────────────
    max_feasible = mini_buffer.size * 5
    n_new_safe   = min(n_new, max_feasible)

    if n_new_safe != n_new:
        print(f"  [INFO] n_new réduit : {n_new:,} → {n_new_safe:,}")

    # ── MCTS sur le mini-buffer ───────────────────────────────────────────
    mcts = MCTSAugmenter(
        n_simulations=20, rollout_depth=5, n_actions=5, device=device
    )
    mini_aug = mcts.augment(mini_buffer, wm, n_new_transitions=n_new_safe)

    # ── D′ = D original + synthétiques ciblées seulement ─────────────────
    new_buffer = buffer.clone()
    synth_size = mini_aug.size - mini_buffer.size

    if synth_size > 0:
        mini_data   = mini_aug.get_all()
        synth_batch = {k: v[mini_buffer.size:] for k, v in mini_data.items()}
        new_buffer.add_batch(synth_batch)
        print(f"  +{synth_size:,} transitions synthétiques ciblées → D′")
    else:
        print("  [WARN] Aucune transition synthétique générée")

    print(f"  D′ final : {new_buffer.size:,} transitions")
    return new_buffer, synth_size


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def run_critic_guided(
    env_name     : str,
    warmup_steps : int = 5_000,
    final_steps  : int = 50_000,
    n_selected   : int = 10_000,
    n_new        : int = 50_000,
    seed         : int = 42,
) -> dict:

    torch.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"CRITIC-GUIDED REANALYSIS — {env_name}")
    print(f"warmup={warmup_steps:,}  final={final_steps:,}  "
          f"n_sel={n_selected:,}  n_new={n_new:,}")
    print(f"{'='*60}")

    cfg     = DATASET_CONFIGS[env_name]
    obs_dim = cfg["obs_dim"]
    act_dim = cfg["act_dim"]

    # ── Charger dataset + world model ─────────────────────────────────────
    print("\n[0] Chargement dataset et world model...")
    buffer = load_buffer(env_name)

    wm_path = f"checkpoints/wm_{env_name}.pt"
    wm      = build_world_model(env_name)
    if os.path.exists(wm_path):
        wm.load(wm_path)
    else:
        wm.train_model(buffer, n_epochs=50)
        wm.save(wm_path)

    # ── Étape 1 : warm-up critic ──────────────────────────────────────────
    agent = warmup_critic(buffer, obs_dim, act_dim,
                          n_steps=warmup_steps, device=DEVICE)

    # ── Étape 2 : sélection critic-guided ────────────────────────────────
    top_idx, advantages = select_by_advantage(
        buffer, agent, n_points=n_selected, device=DEVICE
    )

    # ── Étape 3 : MCTS ciblé ─────────────────────────────────────────────
    buf_prime, n_synth = reanalyze_targeted(
        buffer, wm, top_idx, n_new=n_new, device=DEVICE
    )

    # ── Étape 4 : IQL final ───────────────────────────────────────────────
    print(f"\n[4] IQL final — {final_steps:,} steps sur D′ ciblé...")
    trainer = IQLTrainer(
        obs_dim   = obs_dim,
        act_dim   = act_dim,
        env_name  = env_name,
        device    = DEVICE,
        log_every = max(1, final_steps // 5),
    )
    log = trainer.train(buf_prime, n_steps=final_steps,
                        method="critic_guided_mcts")
    trainer.save(f"checkpoints/iql_critic_guided_{env_name}.pt")

    final_q_loss = log["q_loss"][-1] if log["q_loss"] else 0.0

    # ── Résultats ─────────────────────────────────────────────────────────
    results = {
        "env"           : env_name,
        "n_selected"    : n_selected,
        "n_synth_added" : n_synth,
        "buffer_D"      : buffer.size,
        "buffer_Dprime" : buf_prime.size,
        "adv_mean_all"  : round(advantages.mean().item(), 4),
        "adv_mean_sel"  : round(advantages[top_idx].mean().item(), 4),
        "final_q_loss"  : round(final_q_loss, 4),
    }

    os.makedirs("results", exist_ok=True)
    out = f"results/critic_guided_{env_name}_conservative.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RÉSULTATS — Critic-Guided MCTS")
    print(f"{'='*60}")
    print(f"  États sélectionnés     : {n_selected:,}")
    print(f"  A moyen tout D         : {results['adv_mean_all']:.4f}")
    print(f"  A moyen sélectionnés   : {results['adv_mean_sel']:.4f}  ← doit être > tout D")
    print(f"  Transitions ajoutées   : {n_synth:,}")
    print(f"  Q-loss final           : {final_q_loss:.4f}")
    print(f"  Sauvegardé → {out}")
    print(f"{'='*60}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        default="hopper-medium-v2",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--warmup", type=int, default=5_000)
    parser.add_argument("--steps",  type=int, default=50_000)
    parser.add_argument("--n_sel",  type=int, default=10_000)
    parser.add_argument("--n_new",  type=int, default=50_000)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    run_critic_guided(
        env_name     = args.env,
        warmup_steps = args.warmup,
        final_steps  = args.steps,
        n_selected   = args.n_sel,
        n_new        = args.n_new,
        seed         = args.seed,
    )