"""Phase 5.1 + 6.3 — frozen experiment configuration.

Two things live here, and BOTH are frozen before any ablation is run:

  1. FEATURE_SETS — the exact feature list for each ablation condition. One
     training script consumes this dict so every condition is defined in one
     place and the paper's condition-table can be generated from it verbatim.

  2. HP — the LSTM hyperparameters. Locked once, identical across every
     condition, so the ONLY thing varying across the ablation is the feature
     set. Re-tuning per condition would contaminate the comparison.

Conditions (E1, E2 intentionally unused — the plan compares E0 vs the
progressive narrative-consistency stack E3..E7):

  E0  technical indicators only ............... market-only floor
  E3  + headline & summary sentiment ......... standard dual-sentiment baseline
  E4  + signed divergence       d = s - h
  E5  + absolute divergence    |d|
  E6  + confidence-weighted divergence  cwd
  E7  + coverage-density shrunk sentiment + news_count ... full method
"""
from __future__ import annotations
from dataclasses import dataclass

# --- shared technical block (always fed to the LSTM, identical every condition) ---
# Scale-invariant indicators only — see app/model.py FEATURES for the rationale.
TECH_FEATURES = [
    "ret_1d", "ret_5d", "ret_20d",
    "rsi_14",
    "macd_norm", "macd_sig_norm",
    "ema12_dev", "ema26_dev",
    "vol_20d", "volume_rel", "bb_pct",
]

# --- sentiment-quality features, late-injected as a vector after the LSTM ---
# Each condition is a strict superset of the previous one (E3 ⊂ E4 ⊂ ... ⊂ E7),
# which is what makes the ablation a clean progressive decomposition.
SENTIMENT_FEATURE_SETS = {
    "E0": [],
    "E3": ["headline_sent", "summary_sent"],
    "E4": ["headline_sent", "summary_sent", "signed_div"],
    "E5": ["headline_sent", "summary_sent", "signed_div", "abs_div"],
    "E6": ["headline_sent", "summary_sent", "signed_div", "abs_div", "cwd"],
    "E7": ["headline_sent", "summary_sent", "signed_div", "abs_div", "cwd",
           "shrunk_sent", "news_count_log"],
}

CONDITIONS = list(SENTIMENT_FEATURE_SETS.keys())

CONDITION_DESC = {
    "E0": "technical indicators only (market-only floor)",
    "E3": "tech + headline & summary sentiment (standard dual-sentiment baseline)",
    "E4": "E3 + signed divergence d = s - h",
    "E5": "E4 + absolute divergence |d|",
    "E6": "E5 + confidence-weighted divergence cwd",
    "E7": "E6 + coverage-density shrunk sentiment + news_count (full method)",
}

# Two of the sentiment columns are DERIVED, not read straight from
# feature_dataset.parquet, and the training script must materialize them:
#   shrunk_sent     = shrink(summary_sent, news_count, k, prior)   [Phase 4]
#   news_count_log  = log1p(news_count)   (raw counts are heavily right-skewed)
DERIVED_SENTIMENT_FEATURES = {"shrunk_sent", "news_count_log"}


def feature_columns(condition: str) -> tuple[list[str], list[str]]:
    """Return (tech_features, sentiment_features) for an ablation condition."""
    if condition not in SENTIMENT_FEATURE_SETS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    return list(TECH_FEATURES), list(SENTIMENT_FEATURE_SETS[condition])


# --- prediction horizons ---
HORIZONS = [1, 5, 20]  # trading-day directional horizons

# --- Phase 4.4 — coverage-density shrinkage strength ---
# Tuned on the validation split for E7 only; the winner is locked before any
# test-set evaluation and written back to CHOSEN_K.
SHRINKAGE_K_GRID = [1, 3, 5, 10, 20]
CHOSEN_K = 5  # placeholder until k_tuning.csv locks the validation winner


# --- walk-forward cross-validation ---
# A single fixed split puts train/val/test in different market regimes, which
# swamps feature signal under base-rate noise (measured: 54.7 / 48.4 / 57.9 %
# up-rates). Walk-forward CV evaluates across many regimes; McNemar then runs on
# test predictions pooled over all folds.
N_FOLDS = 5

# --- Phase 6.3 — FROZEN LSTM hyperparameters ---
# Locked from the one-time tune_hparams.py search (hparam_search.csv). Heavily
# regularized: 11k pooled windows is a small-data regime, so a small hidden
# size + strong dropout/weight-decay is what keeps the model from memorizing.
@dataclass(frozen=True)
class Hyperparameters:
    window: int = 20            # LSTM look-back, trading days
    hidden: int = 16            # LSTM hidden units (small — small dataset)
    num_layers: int = 1
    dropout: float = 0.5
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 128
    max_epochs: int = 80
    patience: int = 12          # early-stopping patience on validation loss
    seed: int = 42


HP = Hyperparameters()
