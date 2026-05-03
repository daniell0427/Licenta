"""Lightweight LSTM + sentiment fusion model trained on-the-fly per ticker.

This is a prototype: we fit a small model in seconds on the available history
so the API can return a directional prediction without any pre-trained weights.
For the thesis, this module will be replaced with a properly trained,
fine-tuned model.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "close", "volume", "rsi_14", "macd", "macd_signal",
    "ema_12", "ema_26", "ret_1d",
]


@dataclass
class TrainResult:
    train_acc: float
    val_acc: float
    epochs: int
    n_samples: int


class FusionLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, with_sentiment: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True, num_layers=1)
        sent_dim = 1 if with_sentiment else 0
        self.head = nn.Sequential(
            nn.Linear(hidden + sent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # up / down
        )
        self.with_sentiment = with_sentiment

    def forward(self, x, sentiment=None):
        _, (h, _) = self.lstm(x)
        h = h.squeeze(0)
        if self.with_sentiment and sentiment is not None:
            h = torch.cat([h, sentiment], dim=-1)
        return self.head(h)


def _build_dataset(
    df: pd.DataFrame,
    sentiment_by_date: Optional[Dict[str, float]],
    window: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = df.dropna().copy()
    if len(df) < window + horizon + 10:
        return np.empty((0,)), np.empty((0,)), np.empty((0,))

    df["sentiment"] = 0.0
    if sentiment_by_date:
        idx_dates = df.index.strftime("%Y-%m-%d")
        df["sentiment"] = [sentiment_by_date.get(d, 0.0) for d in idx_dates]

    feat = df[FEATURES].values.astype(np.float32)
    sent = df["sentiment"].values.astype(np.float32)
    close = df["close"].values

    X, S, y = [], [], []
    for i in range(len(df) - window - horizon):
        X.append(feat[i : i + window])
        S.append(sent[i + window - 1])  # most recent day in window
        future = close[i + window + horizon - 1]
        present = close[i + window - 1]
        y.append(1 if future > present else 0)
    return np.asarray(X), np.asarray(S), np.asarray(y)


def train_and_predict(
    df: pd.DataFrame,
    sentiment_by_date: Optional[Dict[str, float]],
    horizon: int,
    window: int = 20,
    epochs: int = 30,
) -> Dict:
    X, S, y = _build_dataset(df, sentiment_by_date, window, horizon)
    if len(X) == 0:
        return {"error": "not enough data"}

    # Scale features per-feature on the training portion
    n = len(X)
    split = max(1, int(n * 0.8))
    flat_train = X[:split].reshape(-1, X.shape[-1])
    scaler = StandardScaler().fit(flat_train)
    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)

    Xt = torch.from_numpy(X_scaled[:split])
    St = torch.from_numpy(S[:split]).unsqueeze(-1)
    yt = torch.from_numpy(y[:split]).long()
    Xv = torch.from_numpy(X_scaled[split:])
    Sv = torch.from_numpy(S[split:]).unsqueeze(-1)
    yv = torch.from_numpy(y[split:]).long()

    model = FusionLSTM(n_features=X.shape[-1], with_sentiment=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xt, St)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_acc = (model(Xt, St).argmax(-1) == yt).float().mean().item()
        val_acc = (model(Xv, Sv).argmax(-1) == yv).float().mean().item() if len(Xv) else float("nan")

        # Predict for the most recent window
        last_window = X_scaled[-1:]
        last_sent = np.array([[S[-1]]], dtype=np.float32)
        logits = model(torch.from_numpy(last_window), torch.from_numpy(last_sent))
        probs = torch.softmax(logits, dim=-1).numpy()[0]
        direction = "up" if probs.argmax() == 1 else "down"
        confidence = float(probs.max())

    return {
        "horizon_days": horizon,
        "direction": direction,
        "confidence": confidence,
        "prob_up": float(probs[1]),
        "prob_down": float(probs[0]),
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "n_samples": int(n),
    }
