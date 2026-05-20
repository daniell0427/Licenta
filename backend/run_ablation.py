"""Phase 7 — drive the full ablation grid.

Runs train_pooled.py once per (target, condition, horizon, seed) combination as
a subprocess, so a crash in one run never loses the others (each writes its own
manifest + predictions under artifacts/runs/). Re-running skips combinations
whose manifest already exists, so the grid is resumable.

Examples:
    # full Part-2 grid (volatility), single seed:
    python run_ablation.py --target volatility

    # multi-seed stability runs for the two key conditions (Phase 7.5):
    python run_ablation.py --target volatility --conditions E3 E7 --seeds 1 2 3

    # Part-1 directional null, for the thesis's negative-result section:
    python run_ablation.py --target direction --horizons 5

    # shrinkage-k sweep for E7 on the validation horizon (Phase 4.4):
    python run_ablation.py --target volatility --conditions E7 --horizons 5 --k-sweep
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import config

RUNS_DIR = Path(__file__).parent / "artifacts" / "runs"


def manifest_path(target, condition, horizon, seed, k) -> Path:
    tag = f"{target}_{condition}_h{horizon}_s{seed}"
    if condition == "E7":
        tag += f"_k{k:g}"
    return RUNS_DIR / f"manifest_{tag}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["direction", "volatility"], default="volatility")
    ap.add_argument("--conditions", nargs="+", default=config.CONDITIONS)
    ap.add_argument("--horizons", nargs="+", type=int, default=config.HORIZONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[config.HP.seed])
    ap.add_argument("--k", type=float, default=config.CHOSEN_K)
    ap.add_argument("--k-sweep", action="store_true",
                    help="for E7, run every k in config.SHRINKAGE_K_GRID instead of --k")
    ap.add_argument("--force", action="store_true", help="re-run even if a manifest exists")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Build the run list.
    runs = []
    for condition in args.conditions:
        ks = (config.SHRINKAGE_K_GRID if (args.k_sweep and condition == "E7") else [args.k])
        for horizon in args.horizons:
            for seed in args.seeds:
                for k in ks:
                    runs.append((condition, horizon, seed, k))

    print(f"[ablation] target={args.target}  {len(runs)} runs queued")
    done = skipped = failed = 0
    t0 = time.time()

    for i, (condition, horizon, seed, k) in enumerate(runs, 1):
        mpath = manifest_path(args.target, condition, horizon, seed, k)
        label = f"{args.target}/{condition}/h{horizon}/s{seed}" + (f"/k{k:g}" if condition == "E7" else "")
        if mpath.exists() and not args.force:
            print(f"[{i}/{len(runs)}] skip (done): {label}")
            skipped += 1
            continue

        print(f"[{i}/{len(runs)}] run: {label}")
        cmd = [sys.executable, "train_pooled.py",
               "--target", args.target, "--condition", condition,
               "--horizon", str(horizon), "--seed", str(seed), "--k", str(k)]
        result = subprocess.run(cmd)
        if result.returncode == 0 and mpath.exists():
            m = json.loads(mpath.read_text())
            acc = m["pooled_test_metrics"]["accuracy"]
            print(f"          -> pooled acc {acc:.4f}")
            done += 1
        else:
            print(f"          -> FAILED (exit {result.returncode})")
            failed += 1

    elapsed = (time.time() - t0) / 60
    print(f"\n[ablation] done={done} skipped={skipped} failed={failed}  ({elapsed:.1f} min)")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
