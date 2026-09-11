from __future__ import annotations

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """
    Temporal world model.

    Input:
        (batch_size, sequence_length, num_features)

    Outputs:
        risk:
            (batch_size, 3)

        stage:
            (batch_size, num_stages)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 3,
        num_stages: int = 2,
    ):
        super().__init__()

        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.dropout = nn.Dropout(dropout)

        # Risk forecasting head
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon),
        )

        # Future stage classification head
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_stages),
        )

    def forward(self, x):

        # x:
        # (batch, sequence_length, features)

        lstm_output, _ = self.lstm(x)

        # Last historical time step
        last_hidden = lstm_output[:, -1, :]

        last_hidden = self.dropout(
            last_hidden
        )

        risk_logits = self.risk_head(
            last_hidden
        )

        stage_logits = self.stage_head(
            last_hidden
        )

        return risk_logits, stage_logits