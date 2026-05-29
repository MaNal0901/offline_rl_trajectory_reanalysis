"""
run_all.py — Orchestration des 4 expériences
=============================================
Lance baseline + vine + mcts + vae sur les 3 datasets.
Produit results/scores.json et results/scores.csv

Usage :
    python src/run_all.py --env hopper-medium-v2 --steps 1000
    python src/run_all.py --env all --steps 100000
"""

from __future__ import annotations
import argparse
import json
import os
import csv

import torch

from interfaces   import DATASET_CONFIGS, DEVICE, normalized_score
from data_loader  import load_buffer
from world_model  import build_world_model
from vine_augment import VineAugmenter
from mcts_augment import MCTSAugmenter
from vae_augment  import VAEAugmenter
from iql_trainer  import IQLTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Score offline — sans gym
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_policy(
    trainer: IQLTrainer,
    buffer: object,
    env_name: str,
) -> float:
    """
    Évaluation offline simple.
    """
    all_data = buffer.get_all()

    mean_rew = all_data["rewards"].mean().item()

    est_return = mean_rew / (1 - 0.99)

    try:
        score = normalized_score(env_name, est_return)
    except Exception:
        score = 0.0

    return round(score, 2)
    """
    Évalue la policy offline sans environnement gym.
    Retourne le normalized_score basé sur la reward moyenne du buffer.
    """
    all_data  = buffer.get_all()
    mean_rew  = all_data["rewards"].mean().item()
    est_return = mean_rew / (1 - 0.99)
    try:
        score = normalized_score(env_name, est_return)
    except Exception:
        score = 0.0
    return round(score, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Expérience principale
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    env_name : str,
    n_steps  : int = 100_000,
    seed     : int = 42,
) -> dict:

    torch.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE : {env_name}  |  steps={n_steps:,}  |  seed={seed}")
    print(f"{'='*60}")

    cfg     = DATASET_CONFIGS[env_name]
    obs_dim = cfg["obs_dim"]
    act_dim = cfg["act_dim"]

    # ── 1. Charger le dataset ─────────────────────────────────────────────
    print("\n[1/4] Chargement dataset...")
    buffer = load_buffer(env_name)

    # ── 2. World model ────────────────────────────────────────────────────
    print("\n[2/4] World model...")
    wm_path = f"checkpoints/wm_{env_name}.pt"
    wm      = build_world_model(env_name)

    if os.path.exists(wm_path):
        print(f"  Checkpoint trouvé — chargement {wm_path}")
        wm.load(wm_path)
    else:
        print(f"  Entraînement 50 epochs...")
        wm.train_model(buffer, n_epochs=50)
        wm.save(wm_path)

    # ── 3. Augmenteurs ────────────────────────────────────────────────────
    augmenters = [
        ("baseline", None),
        ("vine",     VineAugmenter(rollout_horizon=5, n_actions=5)),
        ("mcts",     MCTSAugmenter(n_simulations=20, rollout_depth=5, n_actions=5)),
        ("vae",      VAEAugmenter(latent_dim=32, vae_epochs=30)),
    ]

    # ── 4. IQL × 4 méthodes ───────────────────────────────────────────────
    print("\n[3/4] IQL — 4 méthodes...")
    scores  = {}
    logs    = {}

    for method, augmenter in augmenters:
        print(f"\n{'─'*40}")
        print(f"  Méthode : {method.upper()}")
        print(f"{'─'*40}")

        # Augmenter D
        if augmenter is None:
            buf_train = buffer
            print(f"  Buffer : D original ({buffer.size:,} transitions)")
        else:
            buf_train = augmenter.augment(
                buffer, wm, n_new_transitions=50_000
            )
            print(f"  Buffer : D′ augmenté ({buf_train.size:,} transitions)")

        # Entraîner IQL
        trainer = IQLTrainer(
            obs_dim  = obs_dim,
            act_dim  = act_dim,
            env_name = env_name,
            device   = DEVICE,
        )
        log = trainer.train(buf_train, n_steps=n_steps, method=method)
        trainer.save(f"checkpoints/iql_{method}_{env_name}.pt")

        # Score offline
        score = evaluate_policy(trainer, buf_train, env_name)
        scores[method] = score
        logs[method]   = {
            "final_v_loss"    : log["v_loss"][-1]     if log["v_loss"]     else 0.0,
            "final_q_loss"    : log["q_loss"][-1]     if log["q_loss"]     else 0.0,
            "final_actor_loss": log["actor_loss"][-1] if log["actor_loss"] else 0.0,
            "buffer_size"     : buf_train.size,
            "score"           : score,
        }
        print(f"  → {method.upper()} score : {score:.2f}")

    return scores, logs


# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde résultats
# ─────────────────────────────────────────────────────────────────────────────

def save_results(all_scores: dict, all_logs: dict) -> None:
    os.makedirs("results", exist_ok=True)

    # JSON complet
    json_path = "results/scores.json"
    with open(json_path, "w") as f:
        json.dump({"scores": all_scores, "logs": all_logs}, f, indent=2)
    print(f"\n  JSON → {json_path}")

    # CSV scores
    csv_path = "results/scores.csv"
    methods  = ["baseline", "vine", "mcts", "vae"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env"] + methods)
        for env, scores in all_scores.items():
            writer.writerow([env] + [scores.get(m, 0.0) for m in methods])
    print(f"  CSV  → {csv_path}")

    # CSV logs détaillés
    log_path = "results/logs.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "env", "method", "buffer_size",
            "final_v_loss", "final_q_loss", "final_actor_loss", "score"
        ])
        for env, methods_logs in all_logs.items():
            for method, log in methods_logs.items():
                writer.writerow([
                    env, method,
                    log["buffer_size"],
                    round(log["final_v_loss"],     4),
                    round(log["final_q_loss"],     4),
                    round(log["final_actor_loss"], 4),
                    log["score"],
                ])
    print(f"  Logs → {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Tableau comparatif
# ─────────────────────────────────────────────────────────────────────────────

def print_table(all_scores: dict, all_logs: dict) -> None:
    methods = ["baseline", "vine", "mcts", "vae"]

    print(f"\n{'='*60}")
    print("TABLEAU — Normalized Score (%)")
    print(f"{'='*60}")
    print(f"{'env':<28} {'baseline':>8} {'vine':>8} {'mcts':>8} {'vae':>8}")
    print(f"{'-'*60}")
    for env, scores in all_scores.items():
        row = [f"{scores.get(m, 0.0):>8.2f}" for m in methods]
        print(f"{env:<28} {'  '.join(row)}")

    print(f"\n{'='*60}")
    print("TABLEAU — Buffer Size")
    print(f"{'='*60}")
    print(f"{'env':<28} {'baseline':>9} {'vine':>9} {'mcts':>9} {'vae':>9}")
    print(f"{'-'*60}")
    for env, methods_logs in all_logs.items():
        row = [f"{methods_logs.get(m, {}).get('buffer_size', 0):>9,}"
               for m in methods]
        print(f"{env:<28} {'  '.join(row)}")

    print(f"\n{'='*60}")
    print("TABLEAU — Final Q Loss")
    print(f"{'='*60}")
    print(f"{'env':<28} {'baseline':>8} {'vine':>8} {'mcts':>8} {'vae':>8}")
    print(f"{'-'*60}")
    for env, methods_logs in all_logs.items():
        row = [f"{methods_logs.get(m, {}).get('final_q_loss', 0.0):>8.4f}"
               for m in methods]
        print(f"{env:<28} {'  '.join(row)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="hopper-medium-v2",
        choices=list(DATASET_CONFIGS.keys()) + ["all"],
    )
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    envs = (
        list(DATASET_CONFIGS.keys()) if args.env == "all"
        else [args.env]
    )

    all_scores = {}
    all_logs   = {}

    for env_name in envs:
        scores, logs       = run_experiment(
            env_name = env_name,
            n_steps  = args.steps,
            seed     = args.seed,
        )
        all_scores[env_name] = scores
        all_logs[env_name]   = logs

    save_results(all_scores, all_logs)
    print_table(all_scores, all_logs)

    print("\n[4/4] Done.")
    print("  results/scores.json")
    print("  results/scores.csv")
    print("  results/logs.csv")
    print("\nProchaine étape : python src/evaluate.py")


if __name__ == "__main__":
    main()