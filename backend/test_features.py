"""Unit tests for the coverage-density shrinkage function (Phase 4.1).

Run with either:
    python test_features.py        # plain asserts, no pytest needed
    pytest test_features.py
"""
import math

import numpy as np
import pandas as pd

from features import shrink, shrink_series, build_feature_dataset


def test_n0_returns_prior():
    # No articles -> no evidence -> result must be exactly the prior.
    assert shrink(raw=0.7, n=0, k=5, prior=0.2) == 0.2
    assert shrink(raw=-0.9, n=0, k=10, prior=0.0) == 0.0


def test_n1_hand_computed():
    # (1*1.0 + 5*0.0) / (1 + 5) = 1/6
    assert math.isclose(shrink(raw=1.0, n=1, k=5, prior=0.0), 1 / 6, rel_tol=1e-12)
    # (1*0.6 + 3*0.2) / (1 + 3) = 1.2/4 = 0.3
    assert math.isclose(shrink(raw=0.6, n=1, k=3, prior=0.2), 0.3, rel_tol=1e-12)


def test_n10_hand_computed():
    # (10*1.0 + 5*0.0) / (10 + 5) = 10/15
    assert math.isclose(shrink(raw=1.0, n=10, k=5, prior=0.0), 10 / 15, rel_tol=1e-12)
    # (10*0.8 + 5*0.2) / (10 + 5) = 9/15 = 0.6
    assert math.isclose(shrink(raw=0.8, n=10, k=5, prior=0.2), 0.6, rel_tol=1e-12)


def test_large_n_converges_to_raw():
    # With overwhelming evidence the prior barely matters.
    assert math.isclose(shrink(raw=0.5, n=10_000, k=5, prior=-0.9), 0.5, abs_tol=1e-3)


def test_k0_is_plain_mean():
    # k=0 means "no prior" -> result is just raw (when there is any evidence).
    assert math.isclose(shrink(raw=0.42, n=7, k=0, prior=0.9), 0.42, rel_tol=1e-12)
    # k=0 AND n=0 is undefined -> fall back to prior.
    assert shrink(raw=0.42, n=0, k=0, prior=0.9) == 0.9


def test_monotonic_in_n():
    # More articles -> the estimate moves monotonically from prior toward raw.
    prior, raw, k = 0.0, 1.0, 5
    vals = [shrink(raw, n, k, prior) for n in range(0, 20)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert vals[0] == prior
    assert vals[-1] < raw


def test_shrink_series_matches_scalar():
    raw = pd.Series([1.0, 0.8, 0.0, -0.5])
    n = pd.Series([1, 10, 0, 4])
    prior = pd.Series([0.0, 0.2, 0.3, -0.1])
    vec = shrink_series(raw, n, k=5, prior=prior)
    for i in range(len(raw)):
        scalar = shrink(raw[i], n[i], 5, prior[i])
        assert math.isclose(vec[i], scalar, rel_tol=1e-12)


def test_build_feature_dataset_shapes():
    # Two tickers, a few articles, including a divergent pair.
    arts = pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA", "BBB"],
        "date": pd.to_datetime(["2022-01-03", "2022-01-03", "2022-01-05", "2022-01-04"]),
        "headline_score": [0.5, -0.5, 0.2, 0.1],
        "headline_confidence": [0.9, 0.8, 0.7, 0.6],
        "summary_score": [0.9, -0.1, 0.2, -0.4],
        "summary_confidence": [0.8, 0.7, 0.9, 0.95],
    })
    ds = build_feature_dataset(arts)
    # Continuous daily reindex per ticker -> no gaps.
    for t, g in ds.groupby("ticker"):
        assert (g["date"].diff().dropna() == pd.Timedelta(days=1)).all()
    # Day with two AAA articles on 2022-01-03: news_count == 2.
    row = ds[(ds["ticker"] == "AAA") & (ds["date"] == "2022-01-03")].iloc[0]
    assert row["news_count"] == 2
    # signed_div = mean((0.9-0.5), (-0.1-(-0.5))) = mean(0.4, 0.4) = 0.4
    assert math.isclose(row["signed_div"], 0.4, rel_tol=1e-9)
    # No-news gap day carries zeros.
    gap = ds[(ds["ticker"] == "AAA") & (ds["date"] == "2022-01-04")].iloc[0]
    assert gap["news_count"] == 0 and gap["abs_div"] == 0.0


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
