"""
sensitivity_vine.py — Sensitivity Analysis : Uncertainty Percentile
====================================================================
Teste Vine avec percentile = 50, 75, 90 sur hopper-medium-v2.
Produit results/sensitivity_vine.csv

Usage :
    python src/sensitivity_vine.py --steps 5000
"""

from __future__ import annotations
import argparse
import csv
import os
import torch

from interfaces   import DATASET_CONFIGS, DEVICE, normalized_score, calibrate_threshold
from data_loader  import load_buffer
from world_model  import build_world_model
from vine_augment import VineAugmenter
from iql_trainer  import IQLTrainer


def run_percentile(
    env_name   : str,
    percentile : float,
    n_steps    : int,
    wm,
    buffer,
    seed       : int = 42,
) -> dict:

    torch.manual_seed(seed)
    cfg = DATASET_CONFIGS[env_name]

    print(f"\n{'─'*50}")
    print(f"  Vine — percentile={percentile}  steps={n_steps:,}")
    print(f"{'─'*50}")

    # Vine avec ce percentile
    vine = VineAugmenter(
        rollout_horizon = 5,
        n_actions       = 5,
        device          = DEVICE,
    )

    # Calibrer manuellement le seuil pour ce percentile
    n_samples = min(2000, buffer.size)
    batch     = buffer.sample(n_samples)
    with torch.no_grad():
        u = wm.uncertainty(
            batch["observations"].to(DEVICE),
            batch["actions"].to(DEVICE),
        )
    threshold = torch.quantile(u.squeeze(), percentile / 100.0).item()
    print(f"  seuil p{int(percentile)} = {threshold:.4f}  "
          f"(mean={u.mean():.4f}  max={u.max():.4f})")
    print(
    f"TEST percentile={percentile} "
    f"threshold={threshold:.6f}"
    )
    # Patcher le seuil directement dans augment via monkey-patch
    import functools
    original_augment = vine.augment

    def patched_augment(buffer, world_model, n_new_transitions=50_000):
        # Remplacer calibrate_threshold par notre seuil fixe
        from data_loader import ReplayBuffer as RB
        import vine_augment as va_module

        # Sauvegarder calibrate_threshold original
        import vine_augment as va_module

        original_ct = va_module.calibrate_threshold

        def fixed_threshold(*args, **kwargs):
            print(
                f"  [calibrate_threshold] p{int(percentile)} "
                f"= {threshold:.4f} (fixé)"
            )
            return threshold

        va_module.calibrate_threshold = fixed_threshold

        try:
            result = original_augment(
                buffer,
                world_model,
                n_new_transitions,
            )
        finally:
            va_module.calibrate_threshold = original_ct

        return result

    vine.augment = patched_augment

    # Augmenter D
    n_before = buffer.size
    buf_aug  = vine.augment(buffer, wm, n_new_transitions=50_000)
    n_added  = buf_aug.size - n_before

    # Calculer acceptance rate depuis le log (approximation)
    print(f"  Transitions ajoutées : {n_added:,}")

    # Entraîner IQL
    trainer = IQLTrainer(
        obs_dim  = cfg["obs_dim"],
        act_dim  = cfg["act_dim"],
        env_name = env_name,
        device   = DEVICE,
    )
    log = trainer.train(buf_aug, n_steps=n_steps, method=f"vine_p{int(percentile)}")

    final_q_loss = log["q_loss"][-1] if log["q_loss"] else 0.0
    score        = round(-final_q_loss, 4)

    print(f"  q_loss final = {final_q_loss:.4f}  |  score proxy = {score}")

    return {
        "percentile"    : percentile,
        "threshold"     : round(threshold, 4),
        "n_added"       : n_added,
        "buffer_size"   : buf_aug.size,
        "final_q_loss"  : round(final_q_loss, 4),
        "final_v_loss"  : round(log["v_loss"][-1] if log["v_loss"] else 0.0, 4),
        "score_proxy"   : score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",   default="hopper-medium-v2")
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()

    env_name    = args.env
    percentiles = [50.0, 75.0, 90.0]

    print(f"\n{'='*60}")
    print(f"SENSITIVITY ANALYSIS — Vine Uncertainty Percentile")
    print(f"env={env_name}  steps={args.steps:,}")
    print(f"{'='*60}")

    # Charger buffer et world model une seule fois
    print("\n[1/2] Chargement buffer...")
    buffer = load_buffer(env_name)

    print("\n[2/2] World model...")
    wm_path = f"checkpoints/wm_{env_name}.pt"
    wm      = build_world_model(env_name)
    if os.path.exists(wm_path):
        wm.load(wm_path)
        print(f"  Checkpoint chargé depuis {wm_path}")
    else:
        wm.train_model(buffer, n_epochs=50)
        wm.save(wm_path)

    # Lancer pour chaque percentile
    results = []
    for p in percentiles:
        r = run_percentile(env_name, p, args.steps, wm, buffer)
        results.append(r)

    # Sauvegarder CSV
    os.makedirs("results", exist_ok=True)
    csv_path = "results/sensitivity_vine.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  CSV → {csv_path}")

    # Tableau final
    print(f"\n{'='*60}")
    print("TABLEAU — Sensitivity Analysis : Vine Percentile")
    print(f"{'='*60}")
    print(f"{'percentile':>12} {'threshold':>10} {'n_added':>9} "
          f"{'buffer_size':>12} {'q_loss':>8} {'score_proxy':>12}")
    print(f"{'-'*60}")
    for r in results:
        print(f"  p{int(r['percentile']):<10} "
              f"{r['threshold']:>10.4f} "
              f"{r['n_added']:>9,} "
              f"{r['buffer_size']:>12,} "
              f"{r['final_q_loss']:>8.4f} "
              f"{r['score_proxy']:>12.4f}")


if __name__ == "__main__":
    main()