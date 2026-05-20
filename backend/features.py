"""Phase 3 + 4 — feature engineering for the sentiment-quality-aware model.

From the per-article FNSPID dual scores (artifacts/fnspid_dual_scored.parquet)
this builds the daily (ticker, date) feature table that the ablation conditions
consume:

  headline_sent  mean headline sentiment that day
  summary_sent   mean summary sentiment that day
  signed_div     mean signed divergence    d  = s - h
  abs_div        mean absolute divergence |d| = |s - h|
  cwd            mean confidence-weighted divergence  min(c_h, c_s) * (s - h)
  news_count     number of articles that day
  prior          rolling 30-day prior of summary_sent (coverage-density shrinkage)

`shrunk_sent` is deliberately NOT stored — it is derived at train time via
`shrink()` so the shrinkage strength k can be tuned (Phase 4.4) without
rebuilding the dataset.

Narrative-consistency definitions (Phase 3.1), per article i:
    d_i   = s_i - h_i                       signed divergence
    a_i   = |s_i - h_i|                     absolute divergence
    cwd_i = min(c_h, c_s) * (s_i - h_i)     confidence-weighted divergence

The min-confidence form for cwd is used deliberately: a divergence is only
trustworthy if BOTH the headline and the summary were scored confidently.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).parent / "artifacts"
DUAL_SCORED_PATH = ARTIFACTS / "fnspid_dual_scored.parquet"
FEATURE_DATASET_PATH = ARTIFACTS / "feature_dataset.parquet"

# The seven daily sentiment fields (Phase 3.2). `prior` is the eighth column —
# the ingredient for shrinkage — and shrunk_sent is computed from it at runtime.
DAILY_SENTIMENT_FIELDS = [
    "headline_sent", "summary_sent", "signed_div", "abs_div", "cwd", "news_count",
]

PRIOR_WINDOW_DAYS = 30   # rolling prior look-back / cold-start length


# ---------------------------------------------------------------------------
# Phase 4.1 — coverage-density shrinkage (pure function; see test_features.py)
# ---------------------------------------------------------------------------

def shrink(raw: float, n: float, k: float, prior: float) -> float:
    """Bayesian-style shrinkage of a daily sentiment estimate toward a prior.

        shrink(raw, n, k, prior) = (n * raw + k * prior) / (n + k)

    raw   : the day's observed sentiment (mean over n articles)
    n     : number of articles that day (the evidence weight)
    k     : shrinkage strength — the prior is worth k pseudo-articles
    prior : the value to shrink toward (recent rolling tone, or 0 at cold-start)

    With n = 0 the result is exactly `prior` (no evidence → pure prior). As n
    grows the result converges to `raw` (lots of evidence → trust the day).
    """
    denom = n + k
    if denom == 0:
        return prior
    return (n * raw + k * prior) / denom


def shrink_series(raw: pd.Series, n: pd.Series, k: float, prior: pd.Series) -> pd.Series:
    """Vectorized `shrink` over aligned pandas Series — used at train time."""
    denom = n + k
    out = (n * raw + k * prior) / denom
    # denom is k>0 wherever k>0, so this is only a guard for the k=0 edge case.
    return out.where(denom != 0, prior)


# ---------------------------------------------------------------------------
# Phase 3.1 — per-article divergence features
# ---------------------------------------------------------------------------

def add_divergence(articles: pd.DataFrame) -> pd.DataFrame:
    """Add per-article signed / absolute / confidence-weighted divergence."""
    out = articles.copy()
    d = out["summary_score"] - out["headline_score"]
    min_conf = out[["headline_confidence", "summary_confidence"]].min(axis=1)
    out["d"] = d
    out["abs_d"] = d.abs()
    out["cwd"] = min_conf * d
    return out


# ---------------------------------------------------------------------------
# Phase 3.2 — daily aggregation per (ticker, date)
# ---------------------------------------------------------------------------

def aggregate_daily(articles: pd.DataFrame) -> pd.DataFrame:
    """Mean-aggregate per-article scores to one row per (ticker, date).
    Only days that actually have articles appear here."""
    art = add_divergence(articles)
    daily = (
        art.groupby(["ticker", "date"])
        .agg(
            headline_sent=("headline_score", "mean"),
            summary_sent=("summary_score", "mean"),
            signed_div=("d", "mean"),
            abs_div=("abs_d", "mean"),
            cwd=("cwd", "mean"),
            news_count=("headline_score", "size"),
        )
        .reset_index()
    )
    return daily


# ---------------------------------------------------------------------------
# Phase 4.2 + 4.3 — rolling 30-day prior with cold-start handling
# ---------------------------------------------------------------------------

def _prior_for_ticker(daily_t: pd.DataFrame) -> pd.Series:
    """Compute the shrinkage prior for one ticker's continuous daily frame.

    prior = rolling PRIOR_WINDOW_DAYS mean of summary_sent over news-bearing
            days, computed strictly from PAST days (closed='left' excludes the
            current day — no lookahead).
    Cold-start: 0 for every date within the first PRIOR_WINDOW_DAYS days of the
            ticker's news history (not enough history to form a prior).
    """
    # summary_sent is NaN on no-news days so the rolling mean ignores them
    # instead of being dragged toward zero.
    s = daily_t["summary_sent"].where(daily_t["news_count"] > 0)
    prior = s.rolling(f"{PRIOR_WINDOW_DAYS}D", closed="left").mean().fillna(0.0)

    first_news = s.first_valid_index()
    if first_news is not None:
        cold_end = first_news + pd.Timedelta(days=PRIOR_WINDOW_DAYS)
        prior.loc[prior.index < cold_end] = 0.0
    return prior


def build_feature_dataset(articles: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: per-article scores -> daily (ticker, date) feature table.

    The output has one row per calendar day per ticker (from each ticker's
    first to last news date). No-news days carry zeros and news_count = 0; the
    `prior` column is populated for every day so coverage-density shrinkage is
    well-defined even when n = 0.
    """
    daily = aggregate_daily(articles)
    out_parts = []
    for ticker, g in daily.groupby("ticker", sort=True):
        g = g.set_index("date").sort_index()
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(full_idx)
        g["ticker"] = ticker
        for col in DAILY_SENTIMENT_FIELDS:
            g[col] = g[col].fillna(0.0)
        g["news_count"] = g["news_count"].astype(int)
        g["prior"] = _prior_for_ticker(g)
        g.index.name = "date"
        out_parts.append(g.reset_index())
    out = pd.concat(out_parts, ignore_index=True)
    cols = ["ticker", "date"] + DAILY_SENTIMENT_FIELDS + ["prior"]
    return out[cols].sort_values(["ticker", "date"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=str(DUAL_SCORED_PATH))
    ap.add_argument("--out", dest="out_path", default=str(FEATURE_DATASET_PATH))
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"Missing {in_path} — run score_fnspid_dual.py first.")

    articles = pd.read_parquet(in_path)
    print(f"[features] loaded {len(articles):,} scored articles")

    ds = build_feature_dataset(articles)
    ds.to_parquet(args.out_path, index=False)

    n_news = (ds["news_count"] > 0).sum()
    print(f"[features] {len(ds):,} (ticker, date) rows across {ds['ticker'].nunique()} tickers")
    print(f"           {n_news:,} news-bearing days ({n_news/len(ds):.1%} of rows)")
    print(f"           date range {ds['date'].min().date()} -> {ds['date'].max().date()}")
    print(f"[saved] {args.out_path}")


if __name__ == "__main__":
    main()
