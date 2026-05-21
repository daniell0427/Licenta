"""Honest direction prediction via cross-sectional momentum + selective prediction.

The directional target is RELATIVE: 1 if a stock's horizon-day return beats the
universe-median return. The model is built on cross-sectional momentum — the
Jegadeesh-Titman anomaly, a genuine non-leaky edge — and evaluated with
SELECTIVE PREDICTION: accuracy as a function of coverage. The model abstains on
low-confidence cases; the headline accuracy is reported on the high-conviction
subset, which is how real trading signals are actually measured.

  LSTM window  : 11 technical features  (short-term price action)
  Injected vec : 7 cross-sectional momentum ranks  (the documented edge)
  Label        : 1 if stock horizon-return > universe median
  CV           : walk-forward; per fold an ENSEMBLE of N seeds is trained and
                 their probabilities averaged — this kills the seed variance
                 that wrecked single-seed runs and calibrates confidence.

Usage:
    python train_direction.py --horizon 20
    python train_direction.py --horizon 5 --seeds 5
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from config import TECH_FEATURES
from build_xs_panel import XS_FEATURES
from thesis_model import SentimentFusionLSTM
from train_price import walk_forward_folds
from train_pooled import add_relative_label, _batched_logits

ARTIFACTS = Path(__file__).parent / "artifacts"
RUNS_DIR = ARTIFACTS / "runs"
PANEL_PATH = ARTIFACTS / "panel_sp500_xs.parquet"

# Model config — distinct from the frozen ablation HP: this is a different task
# (500k samples, not 13 tickers) so it gets capacity to match the data.
WINDOW = 20
HIDDEN = 32
DROPOUT = 0.3
LR = 5e-4
WEIGHT_DECAY = 1e-3
BATCH = 512
MAX_EPOCHS = 60
PATIENCE = 8
N_FOLDS = 5
WINDOW_START = "2018-01-01"
WINDOW_END = "2026-03-31"

COVERAGE_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]


def build_samples(panel: pd.DataFrame, horizon: int):
    """Windows of technical features (LSTM) + cross-sectional momentum vector
    (injected) + relative-direction label. Sorted by date."""
    panel = add_relative_label(panel, horizon)
    lo, hi = pd.Timestamp(WINDOW_START), pd.Timestamp(WINDOW_END)
    Xt, Xs, y, dates, tickers = [], [], [], [], []

    for ticker, sub in panel.groupby(level="ticker", sort=False):
        sub = sub.droplevel("ticker").sort_index()
        sub = sub.dropna(subset=TECH_FEATURES + XS_FEATURES + ["close", "y_rel"])
        if len(sub) < WINDOW + horizon + 5:
            continue
        tech = sub[TECH_FEATURES].values.astype(np.float32)
        xs = sub[XS_FEATURES].values.astype(np.float32)
        yrel = sub["y_rel"].values.astype(np.float32)
        idx = sub.index.values
        for t in range(WINDOW - 1, len(sub) - horizon):
            d = pd.Timestamp(idx[t])
            if not (lo <= d <= hi):
                continue
            Xt.append(tech[t - WINDOW + 1 : t + 1])
            Xs.append(xs[t])
            y.append(yrel[t])
            dates.append(idx[t])
            tickers.append(ticker)

    Xt = np.asarray(Xt, np.float32)
    Xs = np.asarray(Xs, np.float32)
    y = np.asarray(y, np.float32)
    dates = np.asarray(dates)
    tickers = np.asarray(tickers, dtype=object)
    order = np.argsort(dates, kind="stable")
    return Xt[order], Xs[order], y[order], dates[order], tickers[order]


def train_one(Xt_tr, Xs_tr, y_tr, Xt_va, Xs_va, y_va, device, seed):
    """Train a single model with early stopping; return it."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SentimentFusionLSTM(n_tech=len(TECH_FEATURES), n_sentiment=len(XS_FEATURES),
                                hidden=HIDDEN, num_layers=1, dropout=DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    n_pos = float(y_tr.sum())
    pos_weight = torch.tensor([(len(y_tr) - n_pos) / max(n_pos, 1.0)], device=device)
    train_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_loss = nn.BCEWithLogitsLoss()

    Xt_t, Xs_t, y_t = torch.from_numpy(Xt_tr), torch.from_numpy(Xs_tr), torch.from_numpy(y_tr)
    n = len(Xt_t)
    yv = torch.from_numpy(y_va)
    best, best_state, bad = float("inf"), None, 0

    for _ in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            sel = perm[i:i + BATCH]
            opt.zero_grad()
            train_loss(model(Xt_t[sel].to(device), Xs_t[sel].to(device)),
                       y_t[sel].to(device)).backward()
            opt.step()
        vloss = eval_loss(_batched_logits(model, Xt_va, Xs_va, device, len(XS_FEATURES)), yv).item()
        if vloss < best - 1e-4:
            best, best_state, bad = vloss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def selective_report(y_true: np.ndarray, p_up: np.ndarray) -> pd.DataFrame:
    """Accuracy at each coverage level, ranking by model confidence |p-0.5|."""
    conf = np.abs(p_up - 0.5)
    order = np.argsort(-conf)
    correct = ((p_up > 0.5).astype(int) == y_true.astype(int))[order]
    rows = []
    for cov in COVERAGE_GRID:
        n = max(1, int(len(correct) * cov))
        rows.append({"coverage": cov, "n": n, "accuracy": float(correct[:n].mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5, help="ensemble size per fold")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== direction h{args.horizon} | ensemble={args.seeds} | device={device} ===")

    panel = pd.read_parquet(PANEL_PATH)
    Xt, Xs, y, dates, tickers = build_samples(panel, args.horizon)
    print(f"samples: {len(y):,}  | overall up-rate={y.mean():.4f}")

    idx = np.arange(len(y))
    pooled = []
    t0 = time.time()

    for fold in walk_forward_folds(idx, idx, dates, n_folds=N_FOLDS):
        kf, tr, va, te = fold["fold"], fold["Xtr"], fold["Xva"], fold["Xte"]
        tech_scaler = StandardScaler().fit(Xt[tr].reshape(-1, Xt.shape[-1]))
        xs_scaler = StandardScaler().fit(Xs[tr])

        def sc_t(a):
            s = a.shape
            return tech_scaler.transform(a.reshape(-1, s[-1])).reshape(s).astype(np.float32)
        Xt_tr, Xt_va, Xt_te = sc_t(Xt[tr]), sc_t(Xt[va]), sc_t(Xt[te])
        Xs_tr = xs_scaler.transform(Xs[tr]).astype(np.float32)
        Xs_va = xs_scaler.transform(Xs[va]).astype(np.float32)
        Xs_te = xs_scaler.transform(Xs[te]).astype(np.float32)

        # Ensemble: average the probabilities of `seeds` independently trained models.
        probs = np.zeros(len(te), dtype=np.float64)
        for s in range(args.seeds):
            model = train_one(Xt_tr, Xs_tr, y[tr], Xt_va, Xs_va, y[va], device, seed=100 + s)
            logits = _batched_logits(model, Xt_te, Xs_te, device, len(XS_FEATURES))
            probs += torch.sigmoid(logits).numpy()
        probs /= args.seeds

        acc = (((probs > 0.5).astype(int)) == y[te].astype(int)).mean()
        print(f"  fold {kf}: train={len(tr):,} test={len(te):,}  blanket_acc={acc:.4f}")
        pooled.append(pd.DataFrame({"ticker": tickers[te], "date": dates[te], "fold": kf,
                                    "y_true": y[te].astype(int), "p_up": probs}))

    pooled_df = pd.concat(pooled, ignore_index=True)
    pooled_df["horizon"] = args.horizon
    tag = f"direction_xsmom_h{args.horizon}"
    pooled_df.to_parquet(RUNS_DIR / f"preds_{tag}.parquet", index=False)

    blanket = (((pooled_df["p_up"] > 0.5).astype(int)) == pooled_df["y_true"]).mean()
    curve = selective_report(pooled_df["y_true"].values, pooled_df["p_up"].values)
    print(f"\n[{tag}] blanket accuracy (all predictions): {blanket:.4f}")
    print("[selective prediction] accuracy vs coverage:")
    print(curve.to_string(index=False))
    curve.to_csv(RUNS_DIR / f"selective_{tag}.csv", index=False)

    manifest = {
        "tag": tag, "horizon": args.horizon, "ensemble_seeds": args.seeds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(y)), "blanket_accuracy": float(blanket),
        "selective_curve": curve.to_dict(orient="records"),
        "model": {"window": WINDOW, "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (RUNS_DIR / f"manifest_{tag}.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[saved] runs/preds_{tag}.parquet, selective_{tag}.csv, manifest_{tag}.json")


if __name__ == "__main__":
    main()
