"""Phase 1.6 — chronological train/val/test splits for the thesis experiments.

Chronological splits ONLY — never random-shuffle a time series, it leaks future
information via feature autocorrelation.

The plan's original splits (train to 2024-09, test 2025-04+) assumed live news
coverage. The actual historical news corpus is FNSPID, which ends ~Jan 2024, so
the splits are shifted earlier to live entirely inside FNSPID's coverage.

    |------------ train -----------|--- val ---|--- test ---|
    2020-01-01            2023-05-31          2023-09-30   2023-12-31

Per-ticker FNSPID coverage is ragged (e.g. AAPL news starts 2020-03, MSFT
2022-04). Pooled multi-ticker training accommodates this — a ticker-day with no
news simply gets zero/empty sentiment features (news_count = 0), which the
coverage-density shrinkage cold-start logic is designed to handle.
"""
from __future__ import annotations
import pandas as pd

TRAIN_START = "2020-01-01"
TRAIN_END   = "2023-05-31"
VAL_START   = "2023-06-01"
VAL_END     = "2023-09-30"
TEST_START  = "2023-10-01"
TEST_END    = "2023-12-31"

SPLIT_BOUNDS = {
    "train": (TRAIN_START, TRAIN_END),
    "val":   (VAL_START, VAL_END),
    "test":  (TEST_START, TEST_END),
}


def assign_split(date) -> str | None:
    """Return 'train' | 'val' | 'test' for a date, or None if outside all ranges."""
    ts = pd.Timestamp(date).normalize()
    for name, (lo, hi) in SPLIT_BOUNDS.items():
        if pd.Timestamp(lo) <= ts <= pd.Timestamp(hi):
            return name
    return None


def add_split_column(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add a 'split' column to a frame with a date column. Rows outside every
    split window get split=None and should be dropped before training."""
    out = df.copy()
    out["split"] = out[date_col].map(assign_split)
    return out
