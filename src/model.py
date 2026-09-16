from __future__ import annotations

import torch
import torch.nn as nn


class RiskLSTM(nn.Module):
    """LSTM encoder with a configurable multi-horizon malicious-risk head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 3,
        sequence_length: int = 10,
    ):
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = float(dropout)
        self.horizon = horizon
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        head_size = max(1, hidden_size // 2)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_size, head_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Model input must be a torch.Tensor.")
        if x.ndim != 3:
            raise ValueError(
                "Model input must have shape (batch, sequence_length, features)."
            )
        if x.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, received {x.shape[1]}."
            )
        if x.shape[2] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, received {x.shape[2]}."
            )
        if x.shape[1] <= 0 or x.shape[0] <= 0:
            raise ValueError("Model input batch/sequence cannot be empty.")
        if not torch.isfinite(x).all():
            raise ValueError("Model input contains NaN or Inf.")

        lstm_output, _ = self.lstm(x)
        last_hidden = self.dropout(lstm_output[:, -1, :])
        logits = self.risk_head(last_hidden)

        if logits.shape != (x.shape[0], self.horizon):
            raise RuntimeError(
                f"Malformed model output: expected {(x.shape[0], self.horizon)}, "
                f"received {tuple(logits.shape)}."
            )
        if not torch.isfinite(logits).all():
            raise RuntimeError("Model produced NaN or Inf logits.")

        return logits
