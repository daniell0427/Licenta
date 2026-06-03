"""Early-fusion variant of the direction model.

Instead of injecting sentiment as a static vector after the LSTM (late fusion),
this model feeds sentiment into the LSTM at every timestep alongside the technical
features. The LSTM can then learn how sentiment evolves over the 20-day window.

  LSTM input/step : 11 technical + 5 sentiment features = 16 features per day
  Late injection  : 7 XS momentum ranks (cross-sectional, point-in-time)
  Label           : 1 if stock horizon-return > universe median

Usage:
    python train_direction_early.py --horizon 1 --seeds 3
    python train_direction_early.py --horizon 5 --seeds 3
    python train_direction_early.py --horizon 21 --seeds 3
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
from build_xs_panel import XS_FEATURES, SENT_FEATURES
from thesis_model import EarlyFusionLSTM
from train_price import walk_forward_folds
from train_pooled import add_relative_label

ARTIFACTS = Path(__file__).parent / "artifacts"
RUNS_DIR = ARTIFACTS / "runs"
PANEL_PATH = ARTIFACTS / "panel_sp500_xs.parquet"

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
    """Build samples with sentiment as a sequence inside the LSTM window.

    Returns:
        Xt   : (n, window, n_tech)      — technical features sequence
        Xs   : (n, window, n_sent)      — sentiment sequence (early fusion)
        Xxs  : (n, n_xs)                — XS momentum ranks (late injection)
        y    : (n,)                     — relative direction label
        dates, tickers
    """
    panel = add_relative_label(panel, horizon)
    lo, hi = pd.Timestamp(WINDOW_START), pd.Timestamp(WINDOW_END)

    sent_cols = [c for c in SENT_FEATURES if c in panel.columns]
    xs_cols = [c for c in XS_FEATURES if c in panel.columns]

    Xt, Xs, Xxs, y, dates, tickers = [], [], [], [], [], []

    for ticker, sub in panel.groupby(level="ticker", sort=False):
        sub = sub.droplevel("ticker").sort_index()
        sub = sub.dropna(subset=TECH_FEATURES + XS_FEATURES + ["close", "y_rel"])
        if len(sub) < WINDOW + horizon + 5:
            continue

        tech = sub[TECH_FEATURES].values.astype(np.float32)
        sent = sub[sent_cols].fillna(0.0).values.astype(np.float32) if sent_cols else np.zeros((len(sub), 0), np.float32)
        xs = sub[xs_cols].values.astype(np.float32)
        yrel = sub["y_rel"].values.astype(np.float32)
        idx = sub.index.values

        for t in range(WINDOW - 1, len(sub) - horizon):
            d = pd.Timestamp(idx[t])
            if not (lo <= d <= hi):
                continue
            Xt.append(tech[t - WINDOW + 1 : t + 1])
            Xs.append(sent[t - WINDOW + 1 : t + 1])
            Xxs.append(xs[t])
            y.append(yrel[t])
            dates.append(idx[t])
            tickers.append(ticker)

    Xt = np.asarray(Xt, np.float32)
    Xs = np.asarray(Xs, np.float32)
    Xxs = np.asarray(Xxs, np.float32)
    y = np.asarray(y, np.float32)
    dates = np.asarray(dates)
    tickers = np.asarray(tickers, dtype=object)
    order = np.argsort(dates, kind="stable")
    return Xt[order], Xs[order], Xxs[order], y[order], dates[order], tickers[order]


def batched_logits(model, Xt, Xs, Xxs, device, batch=4096):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch):
            xt = torch.from_numpy(Xt[i:i+batch]).to(device)
            xs = torch.from_numpy(Xs[i:i+batch]).to(device)
            xxs = torch.from_numpy(Xxs[i:i+batch]).to(device)
            preds.append(model(xt, xs, xxs).cpu())
    return torch.cat(preds) if preds else torch.zeros(0)


def train_one(Xt_tr, Xs_tr, Xxs_tr, y_tr, Xt_va, Xs_va, Xxs_va, y_va, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_sent_seq = Xs_tr.shape[-1]
    n_xs = Xxs_tr.shape[-1]
    model = EarlyFusionLSTM(
        n_tech=len(TECH_FEATURES), n_sent_seq=n_sent_seq, n_xs=n_xs,
        hidden=HIDDEN, num_layers=1, dropout=DROPOUT,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    n_pos = float(y_tr.sum())
    pos_weight = torch.tensor([(len(y_tr) - n_pos) / max(n_pos, 1.0)], device=device)
    train_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_loss = nn.BCEWithLogitsLoss()

    Xt_t = torch.from_numpy(Xt_tr)
    Xs_t = torch.from_numpy(Xs_tr)
    Xxs_t = torch.from_numpy(Xxs_tr)
    y_t = torch.from_numpy(y_tr)
    yv = torch.from_numpy(y_va)
    n = len(Xt_t)
    best, best_state, bad = float("inf"), None, 0

    for _ in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            sel = perm[i:i+BATCH]
            opt.zero_grad()
            train_loss(
                model(Xt_t[sel].to(device), Xs_t[sel].to(device), Xxs_t[sel].to(device)),
                y_t[sel].to(device),
            ).backward()
            opt.step()
        vloss = eval_loss(batched_logits(model, Xt_va, Xs_va, Xxs_va, device), yv).item()
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
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== early-fusion h{args.horizon} | ensemble={args.seeds} | device={device} ===")
    print(f"    LSTM input: {len(TECH_FEATURES)} tech + {len(SENT_FEATURES)} sentiment per step")
    print(f"    Late injection: {len(XS_FEATURES)} XS momentum ranks")

    panel = pd.read_parquet(PANEL_PATH)
    Xt, Xs, Xxs, y, dates, tickers = build_samples(panel, args.horizon)
    print(f"samples: {len(y):,}  | overall up-rate={y.mean():.4f}")

    idx = np.arange(len(y))
    pooled = []
    t0 = time.time()

    for fold in walk_forward_folds(idx, idx, dates, n_folds=N_FOLDS):
        kf, tr, va, te = fold["fold"], fold["Xtr"], fold["Xva"], fold["Xte"]

        tech_scaler = StandardScaler().fit(Xt[tr].reshape(-1, Xt.shape[-1]))
        sent_scaler = StandardScaler().fit(Xs[tr].reshape(-1, Xs.shape[-1]))
        xs_scaler = StandardScaler().fit(Xxs[tr])

        def sc_tech(a):
            s = a.shape
            return tech_scaler.transform(a.reshape(-1, s[-1])).reshape(s).astype(np.float32)

        def sc_sent(a):
            s = a.shape
            return sent_scaler.transform(a.reshape(-1, s[-1])).reshape(s).astype(np.float32)

        Xt_tr, Xt_va, Xt_te = sc_tech(Xt[tr]), sc_tech(Xt[va]), sc_tech(Xt[te])
        Xs_tr, Xs_va, Xs_te = sc_sent(Xs[tr]), sc_sent(Xs[va]), sc_sent(Xs[te])
        Xxs_tr = xs_scaler.transform(Xxs[tr]).astype(np.float32)
        Xxs_va = xs_scaler.transform(Xxs[va]).astype(np.float32)
        Xxs_te = xs_scaler.transform(Xxs[te]).astype(np.float32)

        probs = np.zeros(len(te), dtype=np.float64)
        for s in range(args.seeds):
            model = train_one(Xt_tr, Xs_tr, Xxs_tr, y[tr],
                              Xt_va, Xs_va, Xxs_va, y[va], device, seed=100 + s)
            logits = batched_logits(model, Xt_te, Xs_te, Xxs_te, device)
            probs += torch.sigmoid(logits).numpy()
        probs /= args.seeds

        acc = (((probs > 0.5).astype(int)) == y[te].astype(int)).mean()
        print(f"  fold {kf}: train={len(tr):,} test={len(te):,}  blanket_acc={acc:.4f}")
        pooled.append(pd.DataFrame({"ticker": tickers[te], "date": dates[te], "fold": kf,
                                    "y_true": y[te].astype(int), "p_up": probs}))

    pooled_df = pd.concat(pooled, ignore_index=True)
    pooled_df["horizon"] = args.horizon
    tag = f"direction_early_h{args.horizon}"
    pooled_df.to_parquet(RUNS_DIR / f"preds_{tag}.parquet", index=False)

    blanket = (((pooled_df["p_up"] > 0.5).astype(int)) == pooled_df["y_true"]).mean()
    curve = selective_report(pooled_df["y_true"].values, pooled_df["p_up"].values)
    print(f"\n[{tag}] blanket accuracy: {blanket:.4f}")
    print("[selective prediction] accuracy vs coverage:")
    print(curve.to_string(index=False))
    curve.to_csv(RUNS_DIR / f"selective_{tag}.csv", index=False)

    manifest = {
        "tag": tag, "horizon": args.horizon, "ensemble_seeds": args.seeds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(y)), "blanket_accuracy": float(blanket),
        "selective_curve": curve.to_dict(orient="records"),
        "model": {"window": WINDOW, "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR,
                  "fusion": "early", "n_tech": len(TECH_FEATURES), "n_sent_seq": len(SENT_FEATURES)},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (RUNS_DIR / f"manifest_{tag}.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[saved] runs/preds_{tag}.parquet, selective_{tag}.csv, manifest_{tag}.json")


if __name__ == "__main__":
    main()
