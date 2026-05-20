"""Phase 5.3 — pooled multi-ticker training with walk-forward cross-validation.

One script trains every ablation condition (E0, E3..E7) at both horizons. A
single model is pooled across all 13 basket tickers.

  Technical indicators  -> the LSTM sequence input (identical every condition)
  Sentiment-quality vec -> late-injected after the LSTM (width set by condition)

Two prediction targets (--target):
  direction  -> 1 if the stock's horizon-day return beats the equal-weighted
                basket median return. Part 1 of the thesis (rigorous null).
  volatility -> 1 if realized volatility over the next horizon days exceeds the
                stock's own trailing median. Part 2 — narrative inconsistency as
                a RISK signal, the target where the divergence features should
                actually carry predictive content.

Walk-forward CV (config.N_FOLDS expanding-window folds) — never shuffled. A
single fixed split puts train/val/test in different market regimes (measured
up-rates 54.7 / 48.4 / 57.9 %), which buries feature signal under base-rate
noise. Walk-forward evaluates across many regimes; test predictions are pooled
across folds so McNemar (Phase 8.2) compares conditions on identical items.

Per run it writes a reproducibility manifest (Phase 5.4) and the pooled
test-set predictions per (ticker, date, fold).

Usage:
    python train_pooled.py --condition E7 --horizon 5 --seed 42
    python train_pooled.py --condition E0 --horizon 1
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import config
from config import HP, TECH_FEATURES, N_FOLDS
from features import shrink_series
from splits import TRAIN_START, TEST_END
from thesis_model import SentimentFusionLSTM
from train_price import fetch_long_panel, walk_forward_folds

ARTIFACTS = Path(__file__).parent / "artifacts"
RUNS_DIR = ARTIFACTS / "runs"
TICKER_LIST = Path(__file__).parent / "ticker_list.json"
FEATURE_DATASET = ARTIFACTS / "feature_dataset.parquet"
PANEL_CACHE = ARTIFACTS / "panel_thesis_8y.parquet"
PANEL_YEARS = 8


def load_basket() -> list[str]:
    return [r["ticker"].upper() for r in json.loads(TICKER_LIST.read_text())["basket"]]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_panel(refresh: bool = False) -> pd.DataFrame:
    """Price panel (technical indicators) joined with daily sentiment features.
    Indexed by (date, ticker). k-independent."""
    panel = fetch_long_panel(load_basket(), PANEL_YEARS, cache_path=PANEL_CACHE, refresh=refresh)
    sent = pd.read_parquet(FEATURE_DATASET)
    sent["date"] = pd.to_datetime(sent["date"]).dt.normalize()

    flat = panel.reset_index()
    flat["date"] = pd.to_datetime(flat["date"]).dt.normalize()
    merged = flat.merge(sent, on=["ticker", "date"], how="left")
    for c in ["headline_sent", "summary_sent", "signed_div", "abs_div",
              "cwd", "news_count", "prior"]:
        merged[c] = merged[c].fillna(0.0)
    return merged.set_index(["date", "ticker"]).sort_index()


def materialize_sentiment(panel: pd.DataFrame, k: float) -> pd.DataFrame:
    """Add the two k-dependent derived sentiment columns."""
    out = panel.copy()
    out["shrunk_sent"] = shrink_series(out["summary_sent"], out["news_count"], k, out["prior"])
    out["news_count_log"] = np.log1p(out["news_count"])
    return out


def add_relative_label(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add `y_rel`: 1 if a stock's horizon-day forward return exceeds the
    equal-weighted basket median return that day, else 0.

    ~50/50 by construction in every market regime — this removes market-wide
    drift, so the model cannot win by memorizing the base rate and must extract
    firm-specific signal (exactly what company news sentiment carries)."""
    out = panel.copy()
    fwd_parts = []
    for _ticker, sub in out.groupby(level="ticker", sort=False):
        close = sub.sort_index(level="date")["close"]
        fwd_parts.append(close.shift(-horizon) / close - 1.0)
    out["fwd_ret"] = pd.concat(fwd_parts)
    basket_median = out.groupby(level="date")["fwd_ret"].transform("median")
    out["y_rel"] = (out["fwd_ret"] > basket_median).astype(np.float32)
    return out


def add_volatility_label(panel: pd.DataFrame, horizon: int, lookback: int = 252) -> pd.DataFrame:
    """Add `y_vol`: 1 if realized volatility over the next `horizon` days exceeds
    the stock's own trailing median realized volatility, else 0.

    Realized vol of an h-day window = RMS of its daily returns. The threshold is
    a trailing rolling median (data up to day t only — no lookahead), so the
    label is ~50/50 over time and asks: 'will the coming period be more volatile
    than this stock's recent norm?' Volatility clusters, so this is genuinely
    predictable — leaving room for the narrative-consistency features to help."""
    out = panel.copy()
    parts = []
    for _ticker, sub in out.groupby(level="ticker", sort=False):
        close = sub.sort_index(level="date")["close"]
        r = close.pct_change()
        rv = r.pow(2).rolling(horizon).mean().pow(0.5)          # h-day RV ending at t
        future_rv = rv.shift(-horizon)                          # RV over [t+1, t+h]
        threshold = rv.rolling(lookback, min_periods=3 * horizon).median()
        parts.append((future_rv > threshold).astype(np.float32))
    out["y_vol"] = pd.concat(parts)
    return out


def build_all_samples(panel: pd.DataFrame, condition: str, horizon: int,
                      window: int, target: str):
    """Slide windows per ticker over the experiment window. Returns one set of
    arrays (Xt, Xs, y, dates, tickers) sorted by date — folds are carved later."""
    sent_feats = config.SENTIMENT_FEATURE_SETS[condition]
    lo, hi = pd.Timestamp(TRAIN_START), pd.Timestamp(TEST_END)
    if target == "direction":
        panel = add_relative_label(panel, horizon)
        label_col = "y_rel"
    else:
        panel = add_volatility_label(panel, horizon)
        label_col = "y_vol"
    Xt, Xs, y, dates, tickers = [], [], [], [], []

    for ticker, sub in panel.groupby(level="ticker", sort=False):
        sub = sub.droplevel("ticker").sort_index()
        sub = sub.dropna(subset=TECH_FEATURES + ["close", label_col])
        if len(sub) < window + horizon + 5:
            continue
        tech = sub[TECH_FEATURES].values.astype(np.float32)
        sentv = (sub[sent_feats].values.astype(np.float32)
                 if sent_feats else np.zeros((len(sub), 0), np.float32))
        close = sub["close"].values
        ylabel = sub[label_col].values.astype(np.float32)
        idx = sub.index.values

        for t in range(window - 1, len(sub) - horizon):
            d = pd.Timestamp(idx[t])
            if not (lo <= d <= hi):
                continue
            if close[t] <= 0:
                continue
            Xt.append(tech[t - window + 1 : t + 1])
            Xs.append(sentv[t])
            y.append(ylabel[t])
            dates.append(idx[t])
            tickers.append(ticker)

    Xt = np.asarray(Xt, dtype=np.float32)
    Xs = np.asarray(Xs, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    dates = np.asarray(dates)
    tickers = np.asarray(tickers, dtype=object)
    order = np.argsort(dates, kind="stable")
    return Xt[order], Xs[order], y[order], dates[order], tickers[order]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _batched_logits(model, Xt, Xs, device, n_sent, batch=4096):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch):
            xt = torch.from_numpy(Xt[i:i + batch]).to(device)
            xs = torch.from_numpy(Xs[i:i + batch]).to(device) if n_sent > 0 else None
            preds.append(model(xt, xs).cpu())
    return torch.cat(preds) if preds else torch.zeros(0)


def train_fold(Xt_tr, Xs_tr, y_tr, Xt_va, Xs_va, y_va, n_sent, device, seed):
    """Train one fold's model with early stopping on validation loss.

    The loss is class-balanced via pos_weight = n_neg / n_pos (computed from
    this fold's train labels) so the model cannot minimize loss by collapsing
    to the majority class — it is forced to use the features. Early stopping
    watches the UNWEIGHTED validation loss so the stopping signal stays a clean,
    comparable measure of generalization."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SentimentFusionLSTM(
        n_tech=len(TECH_FEATURES), n_sentiment=n_sent,
        hidden=HP.hidden, num_layers=HP.num_layers, dropout=HP.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=HP.lr, weight_decay=HP.weight_decay)

    n_pos = float(y_tr.sum())
    n_neg = float(len(y_tr) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    train_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_loss_fn = nn.BCEWithLogitsLoss()

    Xt_t, Xs_t, y_t = torch.from_numpy(Xt_tr), torch.from_numpy(Xs_tr), torch.from_numpy(y_tr)
    n = len(Xt_t)
    yv = torch.from_numpy(y_va)
    best_loss, best_state, bad = float("inf"), None, 0

    for ep in range(HP.max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, HP.batch_size):
            sel = perm[i:i + HP.batch_size]
            xt = Xt_t[sel].to(device)
            xs = Xs_t[sel].to(device) if n_sent > 0 else None
            opt.zero_grad()
            train_loss_fn(model(xt, xs), y_t[sel].to(device)).backward()
            opt.step()
        vloss = eval_loss_fn(_batched_logits(model, Xt_va, Xs_va, device, n_sent), yv).item()
        if vloss < best_loss - 1e-4:
            best_loss = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= HP.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


def metrics_from(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = float((y_pred == y_true).mean()) if len(y_true) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"n": int(len(y_true)), "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["direction", "volatility"], default="volatility")
    ap.add_argument("--condition", required=True, choices=config.CONDITIONS)
    ap.add_argument("--horizon", type=int, required=True, choices=config.HORIZONS)
    ap.add_argument("--seed", type=int, default=HP.seed)
    ap.add_argument("--k", type=float, default=config.CHOSEN_K, help="shrinkage strength (E7 only)")
    ap.add_argument("--refresh-panel", action="store_true")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{args.target}_{args.condition}_h{args.horizon}_s{args.seed}"
    if args.condition == "E7":
        tag += f"_k{args.k:g}"
    print(f"=== run {tag} | device={device} ===")

    panel = materialize_sentiment(build_panel(refresh=args.refresh_panel), args.k)
    Xt, Xs, y, dates, tickers = build_all_samples(
        panel, args.condition, args.horizon, HP.window, args.target)
    n_sent = len(config.SENTIMENT_FEATURE_SETS[args.condition])
    print(f"samples: {len(y)}  | n_sent={n_sent}  | overall up-rate={y.mean():.4f}")

    idx = np.arange(len(y))
    fold_rows, pooled = [], []
    t0 = time.time()

    for fold in walk_forward_folds(idx, idx, dates, n_folds=N_FOLDS):
        k_fold = fold["fold"]
        tr, va, te = fold["Xtr"], fold["Xva"], fold["Xte"]

        # Standardize on this fold's TRAIN only.
        tech_scaler = StandardScaler().fit(Xt[tr].reshape(-1, Xt.shape[-1]))
        def scale_tech(a):
            s = a.shape
            return tech_scaler.transform(a.reshape(-1, s[-1])).reshape(s).astype(np.float32)
        Xt_tr, Xt_va, Xt_te = scale_tech(Xt[tr]), scale_tech(Xt[va]), scale_tech(Xt[te])
        if n_sent > 0:
            sent_scaler = StandardScaler().fit(Xs[tr])
            Xs_tr, Xs_va, Xs_te = (sent_scaler.transform(Xs[tr]).astype(np.float32),
                                   sent_scaler.transform(Xs[va]).astype(np.float32),
                                   sent_scaler.transform(Xs[te]).astype(np.float32))
        else:
            # E0 has no sentiment vector — Xs is zero-width, passed through as-is.
            Xs_tr, Xs_va, Xs_te = Xs[tr], Xs[va], Xs[te]

        model, best_val = train_fold(Xt_tr, Xs_tr, y[tr], Xt_va, Xs_va, y[va],
                                     n_sent, device, args.seed + k_fold)
        logits = _batched_logits(model, Xt_te, Xs_te, device, n_sent)
        p_up = torch.sigmoid(logits).numpy()
        y_pred = (p_up > 0.5).astype(int)
        m = metrics_from(y[te], y_pred)
        print(f"  fold {k_fold}: train={len(tr)} test={len(te)}  "
              f"val_loss={best_val:.4f}  test_acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")
        fold_rows.append({"fold": k_fold, "best_val_loss": best_val, **m})
        pooled.append(pd.DataFrame({
            "ticker": tickers[te], "date": dates[te], "fold": k_fold,
            "y_true": y[te].astype(int), "y_pred": y_pred, "p_up": p_up,
        }))

    pooled_df = pd.concat(pooled, ignore_index=True)
    pooled_df["target"] = args.target
    pooled_df["condition"] = args.condition
    pooled_df["horizon"] = args.horizon
    pooled_df["seed"] = args.seed
    pooled_m = metrics_from(pooled_df["y_true"].values, pooled_df["y_pred"].values)
    fold_accs = [r["accuracy"] for r in fold_rows]
    elapsed = time.time() - t0

    print(f"\n[{tag}] POOLED test  acc={pooled_m['accuracy']:.4f}  f1={pooled_m['f1']:.4f}  "
          f"prec={pooled_m['precision']:.4f}  rec={pooled_m['recall']:.4f}")
    print(f"[{tag}] per-fold acc  mean={np.mean(fold_accs):.4f}  std={np.std(fold_accs):.4f}")

    pooled_df.to_parquet(RUNS_DIR / f"preds_{tag}.parquet", index=False)
    manifest = {
        "tag": tag, "target": args.target, "condition": args.condition,
        "condition_desc": config.CONDITION_DESC[args.condition],
        "horizon": args.horizon, "seed": args.seed, "k": args.k,
        "git_hash": git_hash(), "timestamp": datetime.now(timezone.utc).isoformat(),
        "cv": {"scheme": "walk-forward", "n_folds": N_FOLDS},
        "feature_set": {"tech": TECH_FEATURES,
                        "sentiment": config.SENTIMENT_FEATURE_SETS[args.condition]},
        "hyperparameters": vars(HP),
        "n_samples": int(len(y)), "overall_up_rate": float(y.mean()),
        "fold_metrics": fold_rows,
        "fold_acc_mean": float(np.mean(fold_accs)), "fold_acc_std": float(np.std(fold_accs)),
        "pooled_test_metrics": pooled_m,
        "elapsed_sec": round(elapsed, 1),
    }
    (RUNS_DIR / f"manifest_{tag}.json").write_text(json.dumps(manifest, indent=2))
    print(f"[saved] runs/preds_{tag}.parquet, runs/manifest_{tag}.json")


if __name__ == "__main__":
    main()
