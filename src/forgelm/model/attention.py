"""Multi-head causal self-attention module with optional KV Cache support."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from forgelm.model.config import ModelConfig

if TYPE_CHECKING:
    from forgelm.inference.kv_cache import KVCache


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with lower-triangular causal masking and KV cache.

    Ensures tokens can only attend to previous and current positions,
    preserving the autoregressive property. Supports KV Cache for O(N) generation.
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
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        """Forward pass for causal self-attention with optional KV cache.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            kv_cache: Optional KVCache instance for accelerated generation.
            layer_idx: Integer layer index (required if kv_cache is provided).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        B, T, C = x.size()

        # Calculate Query, Key, Values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Update and retrieve historical keys/values if KV cache is active
        if kv_cache is not None and layer_idx is not None:
            k, v = kv_cache.update(k, v, layer_idx=layer_idx)

        total_keys_len = k.size(2)

        # Scaled dot-product attention: (B, n_heads, T, d_head) @ (B, n_heads, d_head, total_keys)
        scale = 1.0 / math.sqrt(self.d_head)
        att = (q @ k.transpose(-2, -1)) * scale  # (B, n_heads, T, total_keys)

        # Apply causal mask
        if T == 1:
            # Single new token attends to all historical tokens up to current position
            pass
        elif T == total_keys_len:
            # Standard training or full prompt prefill: apply lower-triangular mask
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        else:
            # Chunked query with historical cache: position i attends to <= past_len + i
            past_len = total_keys_len - T
            col_indices = torch.arange(total_keys_len, device=x.device).view(1, 1, 1, total_keys_len)
            row_indices = (torch.arange(T, device=x.device) + past_len).view(1, 1, T, 1)
            mask = col_indices <= row_indices
            att = att.masked_fill(~mask, float("-inf"))

        # Softmax over the key dimension
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values: (B, n_heads, T, total_keys) @ (B, n_heads, total_keys, d_head)
        y = att @ v  # (B, n_heads, T, d_head)

        # Concatenate head outputs: (B, n_heads, T, d_head) -> (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection with residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y
