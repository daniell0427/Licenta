"""Fine-tune FinBERT on Financial PhraseBank + Twitter Financial News Sentiment.

Usage:
    python train_sentiment.py --out artifacts/finbert-ft

Datasets are pulled from HuggingFace Hub on first run. The fine-tuned model
is saved to --out and loaded by app/sentiment.py if FINBERT_PATH env var
points at it (or if the directory exists at the default location).

Why both datasets:
- PhraseBank (~4.8k) is professionally labeled financial news — high signal,
  but formal newswire tone.
- Twitter Financial (~12k) is noisier social-media language — closer to what
  shows up in the news feed of a real product.
Combining them produces a model that handles both registers.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, ClassLabel
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    EarlyStoppingCallback,
)


BASE_MODEL = "ProsusAI/finbert"
LABELS = ["positive", "negative", "neutral"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def load_phrasebank():
    # 'sentences_50agree' = labels where >=50% of annotators agreed (largest split)
    ds = load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
    # Schema: { sentence: str, label: ClassLabel(neutral, negative, positive) }
    # Remap to our label order.
    pb_labels = ds["train"].features["label"].names  # e.g. ['negative','neutral','positive']
    def remap(ex):
        return {
            "text": ex["sentence"],
            "label": LABEL2ID[pb_labels[ex["label"]]],
        }
    ds = ds["train"].map(remap, remove_columns=ds["train"].column_names)
    return ds


def load_twitter_fin():
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    # Schema: { text: str, label: int(0=Bearish, 1=Bullish, 2=Neutral) }
    twitter_labels = {0: "negative", 1: "positive", 2: "neutral"}
    def remap(ex):
        return {"text": ex["text"], "label": LABEL2ID[twitter_labels[ex["label"]]]}
    train = ds["train"].map(remap, remove_columns=ds["train"].column_names)
    val = ds["validation"].map(remap, remove_columns=ds["validation"].column_names)
    return train, val


def build_dataset(seed: int):
    print("Loading Financial PhraseBank...")
    pb = load_phrasebank()
    print(f"  {len(pb)} examples")

    print("Loading Twitter Financial News Sentiment...")
    tw_train, tw_val = load_twitter_fin()
    print(f"  {len(tw_train)} train, {len(tw_val)} val")

    # PhraseBank has no canonical split — make our own 85/15
    pb = pb.shuffle(seed=seed)
    n_val_pb = len(pb) // 7  # ~14%
    pb_val = pb.select(range(n_val_pb))
    pb_train = pb.select(range(n_val_pb, len(pb)))

    train = concatenate_datasets([pb_train, tw_train]).shuffle(seed=seed)
    val = concatenate_datasets([pb_val, tw_val]).shuffle(seed=seed)
    print(f"\nCombined: {len(train)} train / {len(val)} val")

    # Class distribution sanity check
    from collections import Counter
    print(f"  train labels: {Counter(train['label'])}")
    print(f"  val labels:   {Counter(val['label'])}")
    return train, val


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/finbert-ft")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_ds, val_ds = build_dataset(args.seed)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tok, batched=True, remove_columns=["text"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting training...")
    trainer.train()

    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # Save in HF format so app/sentiment.py can load via from_pretrained
    print(f"\nSaving final model to {out_dir}/")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Save metrics for reproducibility
    import json
    with open(out_dir / "training_metrics.json", "w") as f:
        json.dump({k: float(v) if isinstance(v, (int, float)) else str(v)
                   for k, v in metrics.items()}, f, indent=2)

    print("Done. To use this model in the API, set:")
    print(f"  export FINBERT_PATH={out_dir.resolve()}")


if __name__ == "__main__":
    main()
