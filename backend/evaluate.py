"""Phase 8 — evaluation: metrics, McNemar significance, strata, k-sensitivity.

Reads every run under artifacts/runs/ and writes the paper's results tables to
artifacts/results/:

  results_main.csv         accuracy / precision / recall / F1 per condition x horizon
  significance_matrix.csv  McNemar p-values for the key pairwise comparisons
  results_by_sector.csv    per-GICS-sector accuracy breakdown
  results_strata.csv       pre-registered strata (high-divergence, low-coverage)
  k_sensitivity.csv        E7 accuracy vs shrinkage k  (only if a k-sweep exists)

Robust to missing runs — produces whatever the available runs support.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from stats import mcnemar, bonferroni

ARTIFACTS = Path(__file__).parent / "artifacts"
RUNS_DIR = ARTIFACTS / "runs"
RESULTS_DIR = ARTIFACTS / "results"
FEATURE_DATASET = ARTIFACTS / "feature_dataset.parquet"
TICKER_LIST = Path(__file__).parent / "ticker_list.json"

# Pairwise comparisons that back the paper's claims (Phase 8.2).
KEY_PAIRS = [("E0", "E3"), ("E3", "E4"), ("E3", "E7"), ("E4", "E7"), ("E6", "E7")]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true, y_pred = y_true.astype(int), y_pred.astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = float((y_pred == y_true).mean()) if len(y_true) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"n": len(y_true), "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _tag(target, condition, horizon, seed, k):
    tag = f"{target}_{condition}_h{horizon}_s{seed}"
    if condition == "E7":
        tag += f"_k{k:g}"
    return tag


def load_preds(target, condition, horizon, seed=42, k=config.CHOSEN_K):
    p = RUNS_DIR / f"preds_{_tag(target, condition, horizon, seed, k)}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _correct(preds: pd.DataFrame) -> pd.DataFrame:
    out = preds[["ticker", "date"]].copy()
    out["correct"] = (preds["y_pred"] == preds["y_true"]).values
    return out


# --- Phase 8.1 — main metrics table -----------------------------------------

def results_main() -> pd.DataFrame:
    rows = []
    for mf in sorted(RUNS_DIR.glob("manifest_*.json")):
        m = json.loads(mf.read_text())
        tm = m["pooled_test_metrics"]
        rows.append({
            "target": m.get("target", "?"), "condition": m["condition"],
            "horizon": m["horizon"], "seed": m["seed"], "k": m.get("k"),
            "n": tm["n"], "accuracy": tm["accuracy"], "precision": tm["precision"],
            "recall": tm["recall"], "f1": tm["f1"],
            "fold_acc_mean": m.get("fold_acc_mean"), "fold_acc_std": m.get("fold_acc_std"),
        })
    df = pd.DataFrame(rows).sort_values(["target", "horizon", "condition", "seed"])
    return df.reset_index(drop=True)


# --- Phase 8.2 — McNemar significance matrix --------------------------------

def significance_matrix(target: str, horizon: int, seed: int = 42) -> pd.DataFrame:
    rows = []
    for a, b in KEY_PAIRS:
        pa, pb = load_preds(target, a, horizon, seed), load_preds(target, b, horizon, seed)
        if pa is None or pb is None:
            continue
        ca, cb = _correct(pa), _correct(pb)
        m = ca.merge(cb, on=["ticker", "date"], suffixes=("_a", "_b"))
        r = mcnemar(m["correct_a"].values, m["correct_b"].values)
        rows.append({"comparison": f"{a} vs {b}", "acc_a": r.acc_a, "acc_b": r.acc_b,
                     "delta": r.acc_b - r.acc_a, "n01": r.n01, "n10": r.n10,
                     "method": r.method, "statistic": r.statistic, "pvalue": r.pvalue})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    corr = bonferroni(df["pvalue"].tolist())
    df["alpha_corrected"] = corr["alpha_corrected"].values
    df["significant"] = corr["significant"].values
    return df


# --- Phase 8.3 — per-sector breakdown ---------------------------------------

def _sector_map() -> dict:
    data = json.loads(TICKER_LIST.read_text())
    return {r["ticker"].upper(): r["sector"] for r in data["basket"]}


def per_sector(target: str, horizon: int, seed: int = 42) -> pd.DataFrame:
    sectors = _sector_map()
    rows = []
    for cond in config.CONDITIONS:
        p = load_preds(target, cond, horizon, seed)
        if p is None:
            continue
        p = p.copy()
        p["sector"] = p["ticker"].str.upper().map(sectors)
        for sector, g in p.groupby("sector"):
            m = _metrics(g["y_true"].values, g["y_pred"].values)
            rows.append({"condition": cond, "sector": sector,
                         "n": m["n"], "accuracy": m["accuracy"], "f1": m["f1"]})
    return pd.DataFrame(rows)


# --- Phase 8.4 + 8.5 — pre-registered strata (E3 vs E7) ---------------------

def strata(target: str, horizon: int, seed: int = 42) -> pd.DataFrame:
    pe3, pe7 = load_preds(target, "E3", horizon, seed), load_preds(target, "E7", horizon, seed)
    if pe3 is None or pe7 is None:
        return pd.DataFrame()

    feat = pd.read_parquet(FEATURE_DATASET)[["ticker", "date", "abs_div", "news_count"]]
    feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()

    def tag_items(p):
        p = p[["ticker", "date", "y_true", "y_pred"]].copy()
        p["date"] = pd.to_datetime(p["date"]).dt.normalize()
        return p.merge(feat, on=["ticker", "date"], how="left")

    e3, e7 = tag_items(pe3), tag_items(pe7)
    merged = e3.merge(e7, on=["ticker", "date"], suffixes=("_e3", "_e7"))
    merged = merged.dropna(subset=["abs_div_e3", "news_count_e3"])

    div_cut = merged["abs_div_e3"].quantile(0.75)
    strata_defs = {
        "overall":         merged.index == merged.index,
        "high_divergence": merged["abs_div_e3"] >= div_cut,
        "rest_divergence": merged["abs_div_e3"] < div_cut,
        "low_coverage":    merged["news_count_e3"] <= 2,
        "med_coverage":    merged["news_count_e3"].between(3, 5),
        "high_coverage":   merged["news_count_e3"] > 5,
    }
    rows = []
    for name, mask in strata_defs.items():
        sub = merged[mask]
        if len(sub) < 10:
            continue
        r = mcnemar((sub["y_pred_e3"] == sub["y_true_e3"]).values,
                    (sub["y_pred_e7"] == sub["y_true_e7"]).values)
        rows.append({"stratum": name, "n": len(sub), "acc_E3": r.acc_a,
                     "acc_E7": r.acc_b, "delta": r.acc_b - r.acc_a,
                     "method": r.method, "pvalue": r.pvalue})
    return pd.DataFrame(rows)


# --- Phase 8.6 — shrinkage k-sensitivity ------------------------------------

def k_sensitivity(target: str, horizon: int = 5, seed: int = 42) -> pd.DataFrame:
    rows = []
    for k in config.SHRINKAGE_K_GRID:
        p = load_preds(target, "E7", horizon, seed, k=k)
        if p is None:
            continue
        m = _metrics(p["y_true"].values, p["y_pred"].values)
        rows.append({"k": k, "n": m["n"], "accuracy": m["accuracy"], "f1": m["f1"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="volatility", choices=["direction", "volatility"])
    ap.add_argument("--seed", type=int, default=config.HP.seed)
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    main_df = results_main()
    main_df.to_csv(RESULTS_DIR / "results_main.csv", index=False)
    print(f"[results_main] {len(main_df)} runs")
    sub = main_df[main_df["target"] == args.target]
    if not sub.empty:
        print(sub[["condition", "horizon", "seed", "accuracy", "f1",
                   "fold_acc_std"]].to_string(index=False))

    horizons = sorted(main_df.loc[main_df["target"] == args.target, "horizon"].unique())
    sig_all, strata_all, sector_all = [], [], []
    for h in horizons:
        sm = significance_matrix(args.target, h, args.seed)
        if not sm.empty:
            sm.insert(0, "horizon", h)
            sig_all.append(sm)
        st = strata(args.target, h, args.seed)
        if not st.empty:
            st.insert(0, "horizon", h)
            strata_all.append(st)
        sec = per_sector(args.target, h, args.seed)
        if not sec.empty:
            sec.insert(0, "horizon", h)
            sector_all.append(sec)

    if sig_all:
        sig = pd.concat(sig_all, ignore_index=True)
        sig.to_csv(RESULTS_DIR / "significance_matrix.csv", index=False)
        print(f"\n[significance_matrix]\n{sig.to_string(index=False)}")
    if strata_all:
        pd.concat(strata_all, ignore_index=True).to_csv(RESULTS_DIR / "results_strata.csv", index=False)
        print(f"\n[results_strata] saved")
    if sector_all:
        pd.concat(sector_all, ignore_index=True).to_csv(RESULTS_DIR / "results_by_sector.csv", index=False)
        print(f"[results_by_sector] saved")

    ks = k_sensitivity(args.target)
    if not ks.empty:
        ks.to_csv(RESULTS_DIR / "k_sensitivity.csv", index=False)
        print(f"\n[k_sensitivity]\n{ks.to_string(index=False)}")
    else:
        print("\n[k_sensitivity] no k-sweep runs found — run: "
              "python run_ablation.py --conditions E7 --horizons 5 --k-sweep")

    print(f"\n[done] tables written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
