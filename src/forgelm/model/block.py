"""Transformer decoder block combining causal self-attention and MLP."""

from __future__ import annotations

import torch
import torch.nn as nn

from forgelm.model.attention import CausalSelfAttention
from forgelm.model.config import ModelConfig
from forgelm.model.mlp import FeedForward


class TransformerBlock(nn.Module):
    """Single Transformer decoder block using modern pre-LayerNorm architecture.

    Formulation:
        x = x + Attention(LayerNorm1(x))
        x = x + MLP(LayerNorm2(x))
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through one Transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Pre-LN Self-Attention with residual connection
        x = x + self.attn(self.ln_1(x))

        # Pre-LN Feed-Forward with residual connection
        x = x + self.mlp(self.ln_2(x))

        return x
