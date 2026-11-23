"""Multi-head causal self-attention module."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from forgelm.model.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with lower-triangular causal masking.

    Ensures tokens can only attend to previous and current positions,
    preserving the autoregressive property.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.dropout_p = config.dropout

        # Key, Query, Value projections in a single batch linear layer
        self.c_attn = nn.Linear(self.d_model, 3 * self.d_model, bias=config.bias)

        # Output projection
        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular matrix of shape (1, 1, max_seq_len, max_seq_len)
        # 1 means attend, 0 means mask out (set to -inf)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        B, T, C = x.size()

        # Calculate Query, Key, Values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape to (B, n_heads, T, d_head) and transpose for matrix multiplication
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        # (B, n_heads, T, d_head) @ (B, n_heads, d_head, T) -> (B, n_heads, T, T)
        scale = 1.0 / math.sqrt(self.d_head)
        att = (q @ k.transpose(-2, -1)) * scale

        # Apply causal mask: mask positions where future tokens exist
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        # Softmax over the key dimension
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values: (B, n_heads, T, T) @ (B, n_heads, T, d_head) -> (B, n_heads, T, d_head)
        y = att @ v

        # Concatenate head outputs: (B, n_heads, T, d_head) -> (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection with residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y
