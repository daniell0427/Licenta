"""Score FNSPID headlines with FinBERT and aggregate to (ticker, date) sentiment.

Usage:
    python score_fnspid.py
    python score_fnspid.py --csv-path /custom/path.csv --batch 256

Streams the 23 GB FNSPID CSV in chunks, filters to S&P 500 tickers (the universe
we train on), scores Article_title with the fine-tuned FinBERT on GPU, and
writes artifacts/historical_sentiment.parquet keyed by (ticker, date).

Columns produced per (ticker, date):
  - sentiment_mean: avg signed sentiment (P(positive) - P(negative))
  - sentiment_std:  std of signed sentiment that day (sentiment volatility)
  - news_count:     number of articles for this ticker on this date

Resumes from artifacts/_fnspid_progress.parquet if the script is interrupted.
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEFAULT_CSV = Path.home() / ".cache/huggingface/hub/datasets--Zihan1004--FNSPID" \
              "/snapshots/bf9189c41527198897d1af3e17b1a0095279fc45/Stock_news/nasdaq_exteral_data.csv"
SP500_PATH = Path(__file__).parent / "sp500.txt"
ARTIFACTS = Path(__file__).parent / "artifacts"
PROGRESS_PATH = ARTIFACTS / "_fnspid_progress.parquet"
OUT_PATH = ARTIFACTS / "historical_sentiment.parquet"


def load_sp500() -> set:
    if not SP500_PATH.exists():
        raise SystemExit(f"Missing {SP500_PATH}. Run train_price.py once with --tickers SP500 first.")
    return {t.strip().upper() for t in SP500_PATH.read_text().splitlines() if t.strip()}


def load_finbert():
    """Use the fine-tuned model if present, else the public ProsusAI/finbert."""
    ft_dir = ARTIFACTS / "finbert-ft"
    model_id = str(ft_dir) if ft_dir.exists() else "ProsusAI/finbert"
    print(f"[finbert] loading {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if device == "cuda":
        model = model.half()  # fp16 → 2x throughput on Blackwell
    # Map label names → indices once. FinBERT uses {positive, negative, neutral}
    # but capitalization varies between checkpoints.
    id2label = {i: v.lower() for i, v in model.config.id2label.items()}
    pos_idx = next(i for i, l in id2label.items() if l == "positive")
    neg_idx = next(i for i, l in id2label.items() if l == "negative")
    return tok, model, device, pos_idx, neg_idx


@torch.no_grad()
def score_batch(texts, tok, model, device, pos_idx, neg_idx, max_len=96):
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    if device == "cuda":
        # logits in fp16 are fine for argmax/softmax precision here
        logits = model(**enc).logits
    else:
        logits = model(**enc).logits
    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    signed = probs[:, pos_idx] - probs[:, neg_idx]
    return signed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default=str(DEFAULT_CSV))
    ap.add_argument("--chunksize", type=int, default=50_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit-rows", type=int, default=0,
                    help="cap rows for a quick smoke test (0=full corpus)")
    ap.add_argument("--save-every", type=int, default=20,
                    help="checkpoint progress every N chunks")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found at {csv_path}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    sp500 = load_sp500()
    print(f"[universe] {len(sp500)} S&P 500 tickers")

    tok, model, device, pos_idx, neg_idx = load_finbert()
    print(f"[finbert] device={device}  pos_idx={pos_idx}  neg_idx={neg_idx}")

    # Resume support — collect already-processed (ticker, date, signed) rows.
    if PROGRESS_PATH.exists():
        scored = pd.read_parquet(PROGRESS_PATH)
        skip_chunks = scored["_chunk_id"].max() + 1 if not scored.empty else 0
        print(f"[resume] loaded {len(scored):,} prior scored rows; skipping first {skip_chunks} chunks")
    else:
        scored = pd.DataFrame(columns=["ticker", "date", "signed", "_chunk_id"])
        skip_chunks = 0

    new_rows: list[pd.DataFrame] = []
    seen_rows = 0
    t_start = time.time()

    reader = pd.read_csv(
        csv_path,
        chunksize=args.chunksize,
        dtype=str,
        on_bad_lines="skip",
        low_memory=False,
    )

    for chunk_id, chunk in enumerate(reader):
        if chunk_id < skip_chunks:
            continue
        # Drop rows missing essentials
        chunk = chunk.dropna(subset=["Stock_symbol", "Date", "Article_title"])
        chunk["Stock_symbol"] = chunk["Stock_symbol"].str.upper().str.strip()
        chunk = chunk[chunk["Stock_symbol"].isin(sp500)]
        if chunk.empty:
            continue

        # Parse dates loosely — FNSPID has multiple date formats
        chunk["date"] = pd.to_datetime(chunk["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        chunk = chunk.dropna(subset=["date"])
        chunk["date"] = chunk["date"].dt.date.astype(str)

        titles = chunk["Article_title"].astype(str).tolist()
        signed = np.empty(len(titles), dtype=np.float32)
        for i in range(0, len(titles), args.batch):
            signed[i : i + args.batch] = score_batch(
                titles[i : i + args.batch], tok, model, device, pos_idx, neg_idx
            )

        new_rows.append(pd.DataFrame({
            "ticker": chunk["Stock_symbol"].values,
            "date": chunk["date"].values,
            "signed": signed,
            "_chunk_id": chunk_id,
        }))
        seen_rows += len(chunk)

        if (chunk_id + 1) % args.save_every == 0:
            checkpoint = pd.concat([scored] + new_rows, ignore_index=True) if scored.size else pd.concat(new_rows, ignore_index=True)
            checkpoint.to_parquet(PROGRESS_PATH, index=False)
            elapsed = time.time() - t_start
            rate = seen_rows / elapsed if elapsed else 0
            print(f"[chunk {chunk_id+1}] kept {seen_rows:,} S&P rows  ({rate:.0f}/s)")

        if args.limit_rows and seen_rows >= args.limit_rows:
            print(f"[limit] hit {args.limit_rows:,} rows — stopping")
            break

    # Final concat + aggregation
    if new_rows:
        full = pd.concat([scored] + new_rows, ignore_index=True) if scored.size else pd.concat(new_rows, ignore_index=True)
        full.to_parquet(PROGRESS_PATH, index=False)
    else:
        full = scored

    print(f"\n[done] scored {len(full):,} headlines across {full['ticker'].nunique()} tickers")
    print("Aggregating to (ticker, date)...")
    agg = (
        full.groupby(["ticker", "date"], as_index=False)
            .agg(sentiment_mean=("signed", "mean"),
                 sentiment_std=("signed", "std"),
                 news_count=("signed", "size"))
    )
    agg["sentiment_std"] = agg["sentiment_std"].fillna(0.0)
    agg["date"] = pd.to_datetime(agg["date"])
    agg.to_parquet(OUT_PATH, index=False)
    print(f"[saved] {OUT_PATH}  ({len(agg):,} (ticker, date) rows)")
    print("\nNext: pass historical_sentiment.parquet into train_price.py to use as features.")


if __name__ == "__main__":
    main()
