"""Feed-forward multi-layer perceptron (MLP) module for Transformer block."""

from __future__ import annotations

import torch
import torch.nn as nn

from forgelm.model.config import ModelConfig


class FeedForward(nn.Module):
    """Two-layer feed-forward network with GELU activation and dropout.

    Projects from d_model -> d_ff -> d_model.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
