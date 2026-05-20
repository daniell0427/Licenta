"""Phase 5.2 — the sentiment-quality-aware directional classifier.

A dedicated thesis model, kept separate from app/model.py so the webapp's
regression FusionLSTM stays untouched. It preserves the original SentiTrade
late-injection design — the LSTM sees ONLY the technical-indicator sequence,
and the sentiment-quality features are concatenated to the LSTM's final hidden
state as a static per-prediction-day vector.

That design is what makes the ablation clean: conditions E0..E7 differ ONLY in
the width of the injected sentiment vector (0 for E0, up to 7 for E7); the
recurrent backbone is byte-for-byte identical across conditions.

Binary directional classifier: outputs a single logit per sample, where
sigmoid(logit) = P(price moves up over the horizon). Train with
nn.BCEWithLogitsLoss against 0/1 direction labels.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SentimentFusionLSTM(nn.Module):
    def __init__(self, n_tech: int, n_sentiment: int, hidden: int = 64,
                 num_layers: int = 1, dropout: float = 0.4):
        """
        n_tech      : number of technical features per timestep (LSTM input width)
        n_sentiment : width of the late-injected sentiment vector (0 for E0)
        """
        super().__init__()
        # nn.LSTM only applies dropout BETWEEN stacked layers, so for a
        # single-layer LSTM we add an explicit dropout on the output instead.
        self.lstm = nn.LSTM(
            n_tech, hidden, batch_first=True,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.post_dropout = nn.Dropout(dropout)
        self.n_sentiment = n_sentiment
        self.head = nn.Sequential(
            nn.Linear(hidden + n_sentiment, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),  # single logit -> binary direction
        )

    def forward(self, tech_seq: torch.Tensor, sentiment: torch.Tensor | None = None):
        """
        tech_seq  : (batch, window, n_tech)
        sentiment : (batch, n_sentiment) — required unless n_sentiment == 0
        returns   : (batch,) logits; sigmoid(logit) = P(up)
        """
        _, (h, _) = self.lstm(tech_seq)
        h = h[-1]                       # top layer's final hidden state
        h = self.post_dropout(h)
        if self.n_sentiment > 0:
            if sentiment is None:
                raise ValueError("this condition needs a sentiment vector but got None")
            h = torch.cat([h, sentiment], dim=-1)
        return self.head(h).squeeze(-1)
