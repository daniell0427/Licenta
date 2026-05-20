"""Phase 6.2 — McNemar paired significance test for ablation comparisons.

Every claimed improvement in the paper (e.g. "E7 beats E3") must be backed by a
paired test on the SAME (ticker, date) test items — accuracy point estimates
alone are not evidence. McNemar's test looks only at the discordant pairs: cases
where one model was right and the other wrong.

  - exact binomial test when discordant pairs are few (more reliable)
  - chi-square with continuity correction when there are many

No statsmodels dependency — the test is implemented directly on scipy primitives
(scipy ships with scikit-learn, already in requirements.txt).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

EXACT_THRESHOLD = 25  # use the exact binomial test when n01 + n10 < this


@dataclass
class McNemarResult:
    n01: int          # model A wrong, model B right
    n10: int          # model A right, model B wrong
    statistic: float
    pvalue: float
    method: str
    acc_a: float
    acc_b: float

    @property
    def discordant(self) -> int:
        return self.n01 + self.n10

    def __str__(self) -> str:
        return (f"McNemar({self.method}): accA={self.acc_a:.4f} accB={self.acc_b:.4f} "
                f"n01={self.n01} n10={self.n10} stat={self.statistic:.3f} "
                f"p={self.pvalue:.4g}")


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> McNemarResult:
    """Paired McNemar test from two boolean correctness arrays (aligned per item).

    a_correct[i] / b_correct[i] = was model A / B correct on test item i.
    """
    a_correct = np.asarray(a_correct, dtype=bool)
    b_correct = np.asarray(b_correct, dtype=bool)
    if a_correct.shape != b_correct.shape:
        raise ValueError("a_correct and b_correct must be aligned and equal length")

    n01 = int((~a_correct & b_correct).sum())
    n10 = int((a_correct & ~b_correct).sum())
    acc_a = float(a_correct.mean()) if len(a_correct) else float("nan")
    acc_b = float(b_correct.mean()) if len(b_correct) else float("nan")
    disc = n01 + n10

    if disc == 0:
        return McNemarResult(n01, n10, 0.0, 1.0, "no-discordant", acc_a, acc_b)

    if disc < EXACT_THRESHOLD:
        k = min(n01, n10)
        pvalue = stats.binomtest(k, disc, 0.5, alternative="two-sided").pvalue
        return McNemarResult(n01, n10, float(k), float(pvalue), "exact", acc_a, acc_b)

    # chi-square with Edwards' continuity correction, df = 1
    chi2 = (abs(n01 - n10) - 1) ** 2 / disc
    pvalue = float(stats.chi2.sf(chi2, df=1))
    return McNemarResult(n01, n10, float(chi2), pvalue, "chi2-cc", acc_a, acc_b)


def mcnemar_from_preds(
    preds_a: pd.DataFrame, preds_b: pd.DataFrame, key=("ticker", "date")
) -> McNemarResult:
    """Run McNemar from two prediction frames (as written by train_pooled.py).

    Each frame must have the key columns plus `y_true` and `y_pred`. The frames
    are inner-joined on the key so exactly the same test items are compared.
    """
    key = list(key)
    a = preds_a[key + ["y_true", "y_pred"]].rename(columns={"y_true": "yt_a", "y_pred": "yp_a"})
    b = preds_b[key + ["y_true", "y_pred"]].rename(columns={"y_true": "yt_b", "y_pred": "yp_b"})
    m = a.merge(b, on=key, how="inner")
    if m.empty:
        raise ValueError("no overlapping (ticker, date) items between the two runs")
    if not (m["yt_a"] == m["yt_b"]).all():
        raise ValueError("ground-truth labels differ between runs — misaligned data")
    return mcnemar(m["yp_a"] == m["yt_a"], m["yp_b"] == m["yt_b"])


def bonferroni(pvalues: list[float], alpha: float = 0.05) -> pd.DataFrame:
    """Bonferroni multiple-comparison correction. Returns a frame with the raw
    p-value, the corrected threshold, and a significant flag per comparison."""
    m = len(pvalues)
    threshold = alpha / m if m else alpha
    return pd.DataFrame({
        "pvalue": pvalues,
        "alpha_corrected": threshold,
        "significant": [p < threshold for p in pvalues],
    })


if __name__ == "__main__":
    # Tiny self-check with hand-constructable numbers.
    rng = np.random.default_rng(0)
    a = rng.random(400) < 0.52
    b = a.copy()
    flip = rng.random(400) < 0.10          # B fixes 10% of items
    b[flip & ~a] = True
    r = mcnemar(a, b)
    print(r)
    print(bonferroni([r.pvalue, 0.2, 0.001]))
