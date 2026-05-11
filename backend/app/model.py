"""Lightweight LSTM + sentiment fusion model trained on-the-fly per ticker.

This is a prototype: we fit a small model in seconds on the available history
so the API can return a directional prediction without any pre-trained weights.
For the thesis, this module will be replaced with a properly trained,
fine-tuned model.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
_PRICE_MODEL_PATH = Path(os.environ.get("PRICE_MODEL_PATH", _ARTIFACTS_DIR / "price_model.pt"))
_PRICE_SCALER_PATH = Path(os.environ.get("PRICE_SCALER_PATH", _ARTIFACTS_DIR / "price_scaler.pkl"))
_LOADED_MODEL: Optional[Tuple["FusionLSTM", StandardScaler, dict]] = None


# Scale-invariant features only — the model needs to see patterns that look
# the same for AAPL at $200 and F at $12, and the same in 2010 and 2024.
# Absolute levels (close, volume, ema_*, macd raw) are kept in market.add_indicators
# for charts and backwards compat but are NOT features for training.
FEATURES = [
    "ret_1d", "ret_5d", "ret_20d",          # momentum at 1/5/20 day horizons
    "rsi_14",                                 # bounded 0–100
    "macd_norm", "macd_sig_norm",             # MACD divided by price
    "ema12_dev", "ema26_dev",                 # EMA % deviation from price
    "vol_20d",                                # realized 20-day volatility
    "volume_rel",                             # volume / 20d avg volume
    "bb_pct",                                 # position within Bollinger band
]


@dataclass
class TrainResult:
    train_acc: float
    val_acc: float
    epochs: int
    n_samples: int


class FusionLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32,
                 with_sentiment: bool = True, dropout: float = 0.3,
                 num_layers: int = 2):
        super().__init__()
        # PyTorch nn.LSTM only applies its dropout BETWEEN layers, so we add
        # an explicit dropout after the LSTM output for the last-layer regularization.
        self.lstm = nn.LSTM(
            n_features, hidden, batch_first=True,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.post_dropout = nn.Dropout(dropout)
        sent_dim = 1 if with_sentiment else 0
        self.head = nn.Sequential(
            nn.Linear(hidden + sent_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),  # regression: predict return as a fraction
        )
        self.with_sentiment = with_sentiment
        self.num_layers = num_layers

    def forward(self, x, sentiment=None):
        _, (h, _) = self.lstm(x)
        # h is (num_layers, batch, hidden) — take the top layer's hidden state
        h = h[-1]
        h = self.post_dropout(h)
        if self.with_sentiment and sentiment is not None:
            h = torch.cat([h, sentiment], dim=-1)
        return self.head(h).squeeze(-1)  # (batch,) scalar return prediction


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


def _load_pretrained() -> Optional[Tuple["FusionLSTM", StandardScaler, dict]]:
    """Load the artifacts produced by train_price.py, cached after first load.
    Returns None if the files don't exist (caller falls back to on-the-fly training)."""
    global _LOADED_MODEL
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL
    if not (_PRICE_MODEL_PATH.exists() and _PRICE_SCALER_PATH.exists()):
        return None
    blob = torch.load(_PRICE_MODEL_PATH, map_location="cpu", weights_only=False)
    model = FusionLSTM(
        n_features=blob["n_features"],
        hidden=blob["hidden"],
        with_sentiment=blob.get("with_sentiment", False),
        dropout=blob.get("dropout", 0.3),
        num_layers=blob.get("num_layers", 2),
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    with open(_PRICE_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    meta = {k: blob[k] for k in ("window", "horizon", "features", "with_sentiment") if k in blob}
    _LOADED_MODEL = (model, scaler, meta)
    print(f"[model] loaded pretrained from {_PRICE_MODEL_PATH}")
    return _LOADED_MODEL


def predict_pretrained(
    df: pd.DataFrame,
    sentiment_by_date: Optional[Dict[str, float]],
) -> Optional[Dict]:
    """Predict using the saved checkpoint. Returns None if no checkpoint exists."""
    loaded = _load_pretrained()
    if loaded is None:
        return None
    model, scaler, meta = loaded
    window = meta.get("window", 20)
    features = meta.get("features", FEATURES)
    horizon = meta.get("horizon", 5)

    df = df.dropna()
    if len(df) < window:
        return None
    last = df[features].values[-window:].astype(np.float32)
    flat = scaler.transform(last)
    x = torch.from_numpy(flat).unsqueeze(0)  # (1, window, n_features)

    sent = None
    if meta.get("with_sentiment") and sentiment_by_date:
        recent_date = df.index[-1].strftime("%Y-%m-%d")
        sent = torch.tensor([[sentiment_by_date.get(recent_date, 0.0)]], dtype=torch.float32)

    with torch.no_grad():
        pred = model(x, sent)  # scalar: predicted return as fraction
        predicted_return = float(pred.item())

    predicted_pct = round(predicted_return * 100, 2)
    return {
        "horizon_days": horizon,
        "predicted_return_pct": predicted_pct,
        "direction": "up" if predicted_return > 0 else "down",
        "source": "pretrained",
    }


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

    # y from _build_dataset is still binary (0/1) — convert to signed returns
    # by centering: treat 1 as +target_return, 0 as -target_return.
    # For on-the-fly training we use the actual close returns instead.
    df_clean = df.dropna().copy()
    close = df_clean["close"].values
    y_ret = np.empty(len(X), dtype=np.float32)
    for i in range(len(X)):
        present = close[i + window - 1]
        future = close[i + window + horizon - 1]
        y_ret[i] = (future - present) / present

    n = len(X)
    split = max(1, int(n * 0.8))
    flat_train = X[:split].reshape(-1, X.shape[-1])
    scaler = StandardScaler().fit(flat_train)
    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)

    Xt = torch.from_numpy(X_scaled[:split])
    St = torch.from_numpy(S[:split]).unsqueeze(-1)
    yt = torch.from_numpy(y_ret[:split])
    Xv = torch.from_numpy(X_scaled[split:])
    Sv = torch.from_numpy(S[split:]).unsqueeze(-1)
    yv = torch.from_numpy(y_ret[split:])

    model = FusionLSTM(n_features=X.shape[-1], with_sentiment=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.HuberLoss(delta=0.02)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        preds = model(Xt, St)
        loss = loss_fn(preds, yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_preds = model(Xt, St)
        val_preds = model(Xv, Sv) if len(Xv) else torch.zeros(0)
        train_dir_acc = ((train_preds > 0) == (yt > 0)).float().mean().item()
        val_dir_acc = ((val_preds > 0) == (yv > 0)).float().mean().item() if len(Xv) else float("nan")

        last_window = X_scaled[-1:]
        last_sent = np.array([[S[-1]]], dtype=np.float32)
        pred = model(torch.from_numpy(last_window), torch.from_numpy(last_sent))
        predicted_return = float(pred.item())

    predicted_pct = round(predicted_return * 100, 2)
    return {
        "horizon_days": horizon,
        "predicted_return_pct": predicted_pct,
        "direction": "up" if predicted_return > 0 else "down",
        "train_dir_acc": float(train_dir_acc),
        "val_dir_acc": float(val_dir_acc),
        "n_samples": int(n),
    }
