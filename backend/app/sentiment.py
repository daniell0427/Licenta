"""FinBERT sentiment scoring.

Phase 2 refactor: the monolithic per-article scoring path is split into a
single-text helper (`score_text`) plus a batch helper (`score_texts_signed_conf`),
which makes it possible to score a headline and an article summary separately —
the foundation of the narrative-consistency features.

Public API:
  score_text(text)                  -> (signed, confidence)        single text
  score_texts_signed_conf(texts)    -> [(signed, confidence), ...]  batched bulk
  score_headline_and_summary(art)   -> dict of 4 dual-score fields  per article
  score_texts(texts)                -> [{label, confidence, signed}]  (legacy)
  aggregate_daily(items)            -> {date: avg_signed}            (legacy)

`signed`     = P(positive) - P(negative), in [-1, 1].
`confidence` = max class probability, in [~0.33, 1] (how peaked the softmax is).
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# If FINBERT_PATH points at a fine-tuned checkpoint (or the default artifact
# directory exists), load that. Otherwise fall back to the public FinBERT.
_DEFAULT_FT = Path(__file__).resolve().parent.parent / "artifacts" / "finbert-ft"
MODEL_NAME = os.environ.get("FINBERT_PATH") or (str(_DEFAULT_FT) if _DEFAULT_FT.exists() else "ProsusAI/finbert")
LABELS = ["positive", "negative", "neutral"]


@lru_cache(maxsize=1)
def _load():
    print(f"[sentiment] loading {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # Resolve class indices once — FinBERT checkpoints vary in label order/case.
    id2label = {i: v.lower() for i, v in model.config.id2label.items()}
    pos_idx = next(i for i, l in id2label.items() if l == "positive")
    neg_idx = next(i for i, l in id2label.items() if l == "negative")
    neu_idx = next(i for i, l in id2label.items() if l == "neutral")
    return tokenizer, model, device, pos_idx, neg_idx, neu_idx


def _probs(texts: List[str], max_length: int = 256) -> np.ndarray:
    """Run FinBERT on a batch of texts → softmax probabilities, shape (N, 3)."""
    tokenizer, model, device, *_ = _load()
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return probs


def _to_signed_conf(probs: np.ndarray) -> List[Tuple[float, float]]:
    """(N, 3) prob array → [(signed, confidence), ...]."""
    _, _, _, pos_idx, neg_idx, _ = _load()
    return [(float(p[pos_idx] - p[neg_idx]), float(p.max())) for p in probs]


# ---------- single-text helper (Phase 2.3) ----------

def score_text(text: str) -> Tuple[float, float]:
    """Score a single text → (signed, confidence). Empty/blank text → (0.0, 0.0)."""
    if not text or not text.strip():
        return 0.0, 0.0
    return _to_signed_conf(_probs([text]))[0]


def score_texts_signed_conf(
    texts: List[str], batch_size: int = 256, max_length: int = 256
) -> List[Tuple[float, float]]:
    """Batched (signed, confidence) for bulk scoring (e.g. FNSPID corpus).
    Empty/blank strings score to (0.0, 0.0) without a forward pass."""
    out: List[Tuple[float, float]] = [(0.0, 0.0)] * len(texts)
    nonempty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    for j in range(0, len(nonempty), batch_size):
        chunk = nonempty[j : j + batch_size]
        probs = _probs([t for _, t in chunk], max_length=max_length)
        for (idx, _), sc in zip(chunk, _to_signed_conf(probs)):
            out[idx] = sc
    return out


# ---------- dual headline/summary scorer (Phase 2.4) ----------

def score_headline_and_summary(article: Dict) -> Dict:
    """Score an article's headline and summary separately.

    Returns the four dual-score fields consumed by the divergence features:
      headline_score, headline_confidence, summary_score, summary_confidence

    Fallback: if the summary is missing or empty, the headline scores are reused
    for the summary, so signed/absolute/confidence-weighted divergence all
    collapse to 0 for that article. This is the documented behaviour for
    headline-only sources (see methodology section).
    """
    title = (article.get("title") or "").strip()
    summary = (article.get("summary") or "").strip()
    h_signed, h_conf = score_text(title)
    if summary and summary != title:
        s_signed, s_conf = score_text(summary)
    else:
        s_signed, s_conf = h_signed, h_conf  # fallback → divergence = 0
    return {
        "headline_score": h_signed,
        "headline_confidence": h_conf,
        "summary_score": s_signed,
        "summary_confidence": s_conf,
    }


# ---------- legacy API (kept for existing callers) ----------

def score_texts(texts: List[str]) -> List[Dict]:
    """Backward-compatible entry point used by news_repo / main / background.
    Returns [{label, confidence, signed}, ...]."""
    if not texts:
        return []
    _, model, _, pos_idx, neg_idx, _ = _load()
    probs = _probs(texts)
    out = []
    for p in probs:
        idx = int(p.argmax())
        out.append({
            "label": model.config.id2label[idx].lower(),
            "confidence": float(p.max()),
            "signed": float(p[pos_idx] - p[neg_idx]),
        })
    return out


def aggregate_daily(items: List[Dict]) -> Dict:
    """items: list of {date, signed}. Returns {date: avg_signed}."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for it in items:
        buckets[it["date"]].append(it["signed"])
    return {d: sum(v) / len(v) for d, v in buckets.items()}
