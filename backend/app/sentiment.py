from functools import lru_cache
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "ProsusAI/finbert"
LABELS = ["positive", "negative", "neutral"]


@lru_cache(maxsize=1)
def _load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


def score_texts(texts: List[str]) -> List[Dict]:
    if not texts:
        return []
    tokenizer, model, device = _load()
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    out = []
    for p in probs:
        idx = int(p.argmax())
        # FinBERT label order from config: positive, negative, neutral
        label = model.config.id2label[idx].lower()
        score = float(p[idx])
        # Signed score: positive minus negative, useful as a feature
        signed = float(p[0] - p[1]) if model.config.id2label[0].lower() == "positive" else None
        if signed is None:
            # Resolve dynamically
            id2 = {v.lower(): k for k, v in model.config.id2label.items()}
            signed = float(p[id2["positive"]] - p[id2["negative"]])
        out.append({"label": label, "confidence": score, "signed": signed})
    return out


def aggregate_daily(items: List[Dict]) -> Dict:
    """items: list of {date, signed}. Returns {date: avg_signed}."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for it in items:
        buckets[it["date"]].append(it["signed"])
    return {d: sum(v) / len(v) for d, v in buckets.items()}
