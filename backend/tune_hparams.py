"""Phase 6.3 (preparatory) — one-time hyperparameter stabilization search.

The frozen-HP smoke tests overfit 11k windows from epoch 1. This script does a
SINGLE search over capacity + regularization on the E3 / horizon-5 validation
split, picks the config that actually trains (validation loss descends below the
random baseline ln(2)=0.693 before it overfits), and prints it so config.HP can
be locked. The winner is then used IDENTICALLY for every ablation condition —
this is not per-condition tuning.

Run once:  python tune_hparams.py
Writes:    artifacts/hparam_search.csv
"""
from __future__ import annotations
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
from config import TECH_FEATURES
from thesis_model import SentimentFusionLSTM
from train_pooled import (build_panel, materialize_sentiment, build_samples,
                          _standardize, _batched_logits)

ARTIFACTS = Path(__file__).parent / "artifacts"
RANDOM_BCE = math.log(2)          # 0.6931 — loss of a 50/50 model
TUNE_CONDITION = "E3"
TUNE_HORIZON = 5
WINDOW = 20
MAX_EPOCHS = 80
PATIENCE = 12

# Focused grid — all levers that fight instant overfitting on a small dataset.
GRID = {
    "hidden":       [16, 32],
    "dropout":      [0.3, 0.5],
    "weight_decay": [1e-3, 1e-2],
    "lr":           [3e-4, 1e-3],
}


def train_eval(data, n_sent, device, hidden, dropout, weight_decay, lr, seed=42):
    """Train one config; return (best_val_loss, best_epoch, val_acc_at_best)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SentimentFusionLSTM(
        n_tech=len(TECH_FEATURES), n_sentiment=n_sent,
        hidden=hidden, num_layers=1, dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    tr = data["train"]
    Xt, Xs, y = torch.from_numpy(tr["Xt"]), torch.from_numpy(tr["Xs"]), torch.from_numpy(tr["y"])
    n = len(Xt)
    va = data["val"]
    yv = torch.from_numpy(va["y"])

    best_loss, best_epoch, best_acc, bad = float("inf"), -1, float("nan"), 0
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            sel = perm[i:i + 128]
            xt = Xt[sel].to(device)
            xs = Xs[sel].to(device) if n_sent > 0 else None
            opt.zero_grad()
            loss_fn(model(xt, xs), y[sel].to(device)).backward()
            opt.step()
        logits = _batched_logits(model, va["Xt"], va["Xs"], device, n_sent)
        vloss = loss_fn(logits, yv).item()
        vacc = (((logits > 0).float().numpy()) == va["y"]).mean()
        if vloss < best_loss - 1e-4:
            best_loss, best_epoch, best_acc, bad = vloss, ep + 1, vacc, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best_loss, best_epoch, best_acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[tune] device={device}  condition={TUNE_CONDITION}  horizon={TUNE_HORIZON}")

    panel = build_panel()
    panel = materialize_sentiment(panel, config.CHOSEN_K)
    data = build_samples(panel, TUNE_CONDITION, TUNE_HORIZON, WINDOW)
    n_sent = len(config.SENTIMENT_FEATURE_SETS[TUNE_CONDITION])
    _standardize(data, n_sent)
    print(f"[tune] samples train={len(data['train']['y'])} val={len(data['val']['y'])}")
    print(f"[tune] random-baseline val loss = {RANDOM_BCE:.4f}\n")

    rows = []
    combos = list(itertools.product(*GRID.values()))
    for i, (hidden, dropout, wd, lr) in enumerate(combos, 1):
        bl, be, ba = train_eval(data, n_sent, device, hidden, dropout, wd, lr)
        beats = "BEATS random" if bl < RANDOM_BCE else "—"
        print(f"[{i:2d}/{len(combos)}] hidden={hidden} drop={dropout} wd={wd:.0e} "
              f"lr={lr:.0e}  ->  val_loss={bl:.4f} @ep{be:2d}  val_acc={ba:.4f}  {beats}")
        rows.append(dict(hidden=hidden, dropout=dropout, weight_decay=wd, lr=lr,
                         best_val_loss=bl, best_epoch=be, val_acc=ba))

    res = pd.DataFrame(rows).sort_values("best_val_loss").reset_index(drop=True)
    res.to_parquet(ARTIFACTS / "hparam_search.parquet", index=False)
    res.to_csv(ARTIFACTS / "hparam_search.csv", index=False)

    print("\n=== ranked by validation loss ===")
    print(res.to_string(index=False))
    win = res.iloc[0]
    print(f"\n[winner] hidden={int(win.hidden)} dropout={win.dropout} "
          f"weight_decay={win.weight_decay:g} lr={win.lr:g}")
    print(f"         val_loss={win.best_val_loss:.4f} (random={RANDOM_BCE:.4f}) "
          f"val_acc={win.val_acc:.4f} @epoch {int(win.best_epoch)}")
    if win.best_val_loss >= RANDOM_BCE:
        print("\n[!] No config beat the random baseline — the signal may be too "
              "weak at this dataset size. Review before locking config.HP.")
    else:
        print("\n[ok] Update config.HP with the winner, then run the ablation.")


if __name__ == "__main__":
    main()
