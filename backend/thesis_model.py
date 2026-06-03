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


class EarlyFusionLSTM(nn.Module):
    """Early fusion variant: sentiment is concatenated into the LSTM input at every
    timestep so the recurrent backbone sees how sentiment evolves over the window.
    XS momentum ranks are still injected late (they are point-in-time, not sequences).

    LSTM input per timestep : [n_tech technical features | n_sent_seq sentiment features]
    Late injection           : n_xs cross-sectional momentum ranks
    """
    def __init__(self, n_tech: int, n_sent_seq: int, n_xs: int, hidden: int = 32,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            n_tech + n_sent_seq, hidden, batch_first=True,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.post_dropout = nn.Dropout(dropout)
        self.n_xs = n_xs
        self.head = nn.Sequential(
            nn.Linear(hidden + n_xs, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, tech_seq: torch.Tensor, sent_seq: torch.Tensor,
                xs: torch.Tensor | None = None):
        """
        tech_seq : (batch, window, n_tech)
        sent_seq : (batch, window, n_sent_seq)  — daily sentiment over the window
        xs       : (batch, n_xs)                — today's cross-sectional ranks
        returns  : (batch,) logits
        """
        x = torch.cat([tech_seq, sent_seq], dim=-1)
        _, (h, _) = self.lstm(x)
        h = h[-1]
        h = self.post_dropout(h)
        if self.n_xs > 0 and xs is not None:
            h = torch.cat([h, xs], dim=-1)
        return self.head(h).squeeze(-1)
