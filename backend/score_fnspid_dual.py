"""Score FNSPID articles with FinBERT — headline AND summary scored SEPARATELY.

Phase 2 (thesis pipeline). Unlike score_fnspid.py — which scored only
`Article_title` and aggregated straight to (ticker, date) — this keeps ONE ROW
PER ARTICLE with four scores, the raw material for the narrative-consistency
divergence features:

    headline_score, headline_confidence, summary_score, summary_confidence

  - headline = FNSPID `Article_title`
  - summary  = FNSPID `Lsa_summary` (LSA extractive summary of the article body)

If an article has no summary, the headline scores are reused for the summary so
the divergence features (d, |d|, cwd) collapse to 0 for that article.

Filters the 94.6M-row corpus to the thesis ticker basket (ticker_list.json).
Streams in chunks; resumes from artifacts/_fnspid_dual_progress.parquet.

Usage:
    python score_fnspid_dual.py
    python score_fnspid_dual.py --summary-col Textrank_summary --limit-rows 200000
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from app import sentiment

DEFAULT_CSV = Path.home() / ".cache/huggingface/hub/datasets--Zihan1004--FNSPID" \
              "/snapshots/bf9189c41527198897d1af3e17b1a0095279fc45/Stock_news/nasdaq_exteral_data.csv"
ARTIFACTS = Path(__file__).parent / "artifacts"
TICKER_LIST = Path(__file__).parent / "ticker_list.json"
PROGRESS_PATH = ARTIFACTS / "_fnspid_dual_progress.parquet"
OUT_PATH = ARTIFACTS / "fnspid_dual_scored.parquet"

OUT_COLS = ["ticker", "date", "headline_score", "headline_confidence",
            "summary_score", "summary_confidence", "_chunk_id"]


def load_basket() -> set:
    if not TICKER_LIST.exists():
        raise SystemExit(f"Missing {TICKER_LIST} — run Phase 1.5 first.")
    data = json.loads(TICKER_LIST.read_text())
    return {row["ticker"].upper() for row in data["basket"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default=str(DEFAULT_CSV))
    ap.add_argument("--summary-col", default="Lsa_summary",
                    choices=["Lsa_summary", "Luhn_summary", "Textrank_summary", "Lexrank_summary"],
                    help="which FNSPID extractive summary column to treat as the article summary")
    ap.add_argument("--chunksize", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--save-every", type=int, default=10, help="checkpoint every N kept chunks")
    ap.add_argument("--limit-rows", type=int, default=0, help="cap kept rows for a smoke test")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"FNSPID CSV not found at {csv_path}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    basket = load_basket()
    print(f"[universe] {len(basket)} thesis tickers: {sorted(basket)}")
    print(f"[summary]  using FNSPID column '{args.summary_col}' as the article summary")

    # Resume support.
    if PROGRESS_PATH.exists():
        scored = pd.read_parquet(PROGRESS_PATH)
        skip_chunks = int(scored["_chunk_id"].max()) + 1 if not scored.empty else 0
        print(f"[resume] {len(scored):,} prior rows; skipping first {skip_chunks} chunks")
    else:
        scored = pd.DataFrame(columns=OUT_COLS)
        skip_chunks = 0

    new_rows: list[pd.DataFrame] = []
    kept = 0
    t_start = time.time()

    reader = pd.read_csv(
        csv_path,
        chunksize=args.chunksize,
        usecols=["Date", "Article_title", "Stock_symbol", args.summary_col],
        dtype=str,
        on_bad_lines="skip",
        low_memory=False,
    )

    for chunk_id, chunk in enumerate(reader):
        if chunk_id < skip_chunks:
            continue
        chunk = chunk.dropna(subset=["Stock_symbol", "Date", "Article_title"])
        chunk["Stock_symbol"] = chunk["Stock_symbol"].str.upper().str.strip()
        chunk = chunk[chunk["Stock_symbol"].isin(basket)]
        if chunk.empty:
            continue

        # FNSPID has mixed date formats — parse loosely, drop unparseable.
        dt = pd.to_datetime(chunk["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        chunk = chunk.assign(date=dt).dropna(subset=["date"])
        if chunk.empty:
            continue

        headlines = chunk["Article_title"].astype(str).tolist()
        summaries = chunk[args.summary_col].fillna("").astype(str).tolist()

        head_sc = sentiment.score_texts_signed_conf(headlines, batch_size=args.batch)
        summ_sc = sentiment.score_texts_signed_conf(summaries, batch_size=args.batch)

        h_signed = np.array([s for s, _ in head_sc], dtype=np.float32)
        h_conf   = np.array([c for _, c in head_sc], dtype=np.float32)
        s_signed = np.array([s for s, _ in summ_sc], dtype=np.float32)
        s_conf   = np.array([c for _, c in summ_sc], dtype=np.float32)

        # Fallback: blank summary → reuse headline scores (divergence = 0).
        blank = np.array([not t.strip() for t in summaries])
        s_signed[blank] = h_signed[blank]
        s_conf[blank] = h_conf[blank]

        new_rows.append(pd.DataFrame({
            "ticker": chunk["Stock_symbol"].values,
            "date": chunk["date"].dt.normalize().values,
            "headline_score": h_signed,
            "headline_confidence": h_conf,
            "summary_score": s_signed,
            "summary_confidence": s_conf,
            "_chunk_id": chunk_id,
        }))
        kept += len(chunk)

        if len(new_rows) % args.save_every == 0:
            checkpoint = pd.concat([scored] + new_rows, ignore_index=True)
            checkpoint.to_parquet(PROGRESS_PATH, index=False)
            rate = kept / max(time.time() - t_start, 1e-9)
            print(f"[chunk {chunk_id+1}] kept {kept:,} basket articles  ({rate:.0f}/s)")

        if args.limit_rows and kept >= args.limit_rows:
            print(f"[limit] hit {args.limit_rows:,} kept rows — stopping")
            break

    full = pd.concat([scored] + new_rows, ignore_index=True) if new_rows else scored
    full.to_parquet(PROGRESS_PATH, index=False)

    out = full.drop(columns=["_chunk_id"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\n[done] {len(out):,} scored articles across {out['ticker'].nunique()} tickers")
    print(f"       date range {out['date'].min()} -> {out['date'].max()}")
    print(f"[saved] {OUT_PATH}")
    print("\nNext: Phase 3 — build divergence features from this per-article table.")


if __name__ == "__main__":
    main()
