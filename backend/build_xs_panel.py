"""Step 1 of the direction-prediction plan — cross-sectional momentum features.

Cross-sectional momentum (Jegadeesh & Titman, 1993) is one of the few robust,
decades-documented return anomalies: stocks that have outperformed their peers
tend to keep outperforming over the next 1-12 months. That is a genuine,
non-leaky edge — exactly what an honest directional model should be built on.

This script enriches a price panel with:
  - longer-horizon momentum     mom_1m / mom_3m / mom_6m / mom_12_1
  - per-date cross-sectional    xs_<feature> = percentile rank in [0,1] of each
    percentile ranks            stock against the whole universe that day

`mom_12_1` is the classic 12-minus-1-month momentum (return from t-252 to t-21,
skipping the most recent month to avoid the short-term reversal effect).

Usage:
    python build_xs_panel.py --in artifacts/panel_sp500_10y.parquet \\
                             --out artifacts/panel_sp500_xs.parquet
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).parent / "artifacts"

# Cross-sectional rank features the direction model consumes (the injected
# vector). All are percentile ranks in [0,1] within the universe on each date.
XS_RANK_BASE = ["mom_1m", "mom_3m", "mom_6m", "mom_12_1", "ret_5d", "vol_20d", "rsi_14"]
XS_FEATURES = [f"xs_{c}" for c in XS_RANK_BASE]

# FinBERT sentiment features joined from historical_sentiment.parquet
SENT_FEATURES = ["sentiment_mean", "sentiment_std", "log_news_count", "sentiment_5d_mean", "sentiment_surge"]

# Combined injected vector: momentum ranks + sentiment
INJ_FEATURES = XS_FEATURES + SENT_FEATURES


def add_momentum(panel: pd.DataFrame) -> pd.DataFrame:
    """Add longer-horizon per-ticker momentum measures."""
    out = panel.copy()
    by_ticker = out.groupby(level="ticker")["close"]
    out["mom_1m"] = by_ticker.pct_change(20)
    out["mom_3m"] = by_ticker.pct_change(60)
    out["mom_6m"] = by_ticker.pct_change(120)
    # 12-1 momentum: return from t-252 to t-21 (skip the most recent month).
    c21 = out.groupby(level="ticker")["close"].shift(21)
    c252 = out.groupby(level="ticker")["close"].shift(252)
    out["mom_12_1"] = c21 / c252 - 1.0
    return out


def add_xs_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """Add per-date cross-sectional percentile ranks (the momentum signals)."""
    out = panel.copy()
    for col in XS_RANK_BASE:
        if col not in out.columns:
            raise KeyError(f"panel is missing required column {col!r}")
        out[f"xs_{col}"] = out.groupby(level="date")[col].rank(pct=True, method="average")
    return out


def join_sentiment(panel: pd.DataFrame, sentiment_path: Path) -> pd.DataFrame:
    """Join FinBERT daily sentiment onto the panel and compute derived features."""
    sent = pd.read_parquet(sentiment_path)
    sent["date"] = pd.to_datetime(sent["date"]).dt.normalize()
    sent = sent.rename(columns={"news_count": "raw_news_count"})
    sent["log_news_count"] = np.log1p(sent["raw_news_count"])

    out = panel.reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.merge(
        sent[["ticker", "date", "sentiment_mean", "sentiment_std", "log_news_count"]],
        on=["ticker", "date"], how="left",
    )
    # Fill missing sentiment days with neutral values
    out["sentiment_mean"] = out["sentiment_mean"].fillna(0.0)
    out["sentiment_std"] = out["sentiment_std"].fillna(0.0)
    out["log_news_count"] = out["log_news_count"].fillna(0.0)

    # 5-day rolling sentiment mean (per ticker)
    out = out.sort_values(["ticker", "date"])
    out["sentiment_5d_mean"] = (
        out.groupby("ticker")["sentiment_mean"]
        .transform(lambda s: s.rolling(5, min_periods=1).mean())
    )
    # Sentiment surge: today vs 5d mean
    out["sentiment_surge"] = out["sentiment_mean"] - out["sentiment_5d_mean"]

    out = out.set_index(["date", "ticker"])
    return out


def build(panel: pd.DataFrame, sentiment_path: Path | None = None) -> pd.DataFrame:
    out = add_xs_ranks(add_momentum(panel))
    if sentiment_path is not None and sentiment_path.exists():
        out = join_sentiment(out, sentiment_path)
    return out.dropna(subset=XS_FEATURES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=str(ARTIFACTS / "panel_sp500_10y.parquet"))
    ap.add_argument("--out", dest="out_path", default=str(ARTIFACTS / "panel_sp500_xs.parquet"))
    ap.add_argument("--sentiment", dest="sentiment_path",
                    default=str(ARTIFACTS / "historical_sentiment.parquet"))
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"Missing {in_path}")

    panel = pd.read_parquet(in_path)
    print(f"[xs] loaded {len(panel):,} rows, {panel.index.get_level_values('ticker').nunique()} tickers")

    sentiment_path = Path(args.sentiment_path)
    if sentiment_path.exists():
        print(f"[xs] joining sentiment from {sentiment_path}")
    else:
        print(f"[xs] sentiment file not found at {sentiment_path}, skipping")
        sentiment_path = None

    enriched = build(panel, sentiment_path)
    enriched.to_parquet(args.out_path)
    print(f"[xs] enriched panel: {len(enriched):,} rows after momentum warm-up")
    print(f"[xs] cross-sectional features: {XS_FEATURES}")
    for c in XS_FEATURES:
        s = enriched[c]
        print(f"     {c:14s} min={s.min():.3f} mean={s.mean():.3f} max={s.max():.3f}")
    if sentiment_path is not None:
        print(f"[xs] sentiment features: {SENT_FEATURES}")
        cov = (enriched["log_news_count"] > 0).mean()
        print(f"     news coverage: {cov:.1%}")
    print(f"[saved] {args.out_path}")


if __name__ == "__main__":
    main()
