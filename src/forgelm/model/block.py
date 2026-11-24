"""Transformer decoder block combining causal self-attention and MLP with KV cache support."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from forgelm.model.attention import CausalSelfAttention
from forgelm.model.config import ModelConfig
from forgelm.model.mlp import FeedForward

if TYPE_CHECKING:
    from forgelm.inference.kv_cache import KVCache


class TransformerBlock(nn.Module):
    """Single Transformer decoder block using modern pre-LayerNorm architecture.

    Formulation:
        x = x + Attention(LayerNorm1(x), kv_cache, layer_idx)
        x = x + MLP(LayerNorm2(x))
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        """Forward pass through one Transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            kv_cache: Optional key-value cache.
            layer_idx: Layer index for cache addressing.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Pre-LN Self-Attention with residual connection and optional KV caching
        x = x + self.attn(self.ln_1(x), kv_cache=kv_cache, layer_idx=layer_idx)

        # Pre-LN Feed-Forward with residual connection
        x = x + self.mlp(self.ln_2(x))

        return x
