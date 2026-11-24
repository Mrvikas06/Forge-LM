"""Key-Value (KV) Cache management for accelerated autoregressive generation.

Eliminates redundant matrix multiplications across generation steps,
reducing computational complexity from O(N^2) to O(N).
"""

from __future__ import annotations

import torch


class KVCache:
    """Manages layer-wise key and value projection tensors during inference.

    Stores past keys and values of shape (batch_size, n_heads, seq_len, d_head)
    so that only the single newest token needs to be projected at each decoding step.
    """

    def __init__(self) -> None:
        """Initialize empty KV Cache."""
        # layer_idx -> (cached_keys, cached_values)
        self.cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append newly computed keys and values to the cache for a given layer.

        Args:
            key: New key tensor of shape (batch_size, n_heads, new_seq_len, d_head).
            value: New value tensor of shape (batch_size, n_heads, new_seq_len, d_head).
            layer_idx: Integer layer index identifying the Transformer block.

        Returns:
            Tuple of (full_keys, full_values) including all historical tokens.
        """
        if layer_idx not in self.cache:
            # First forward pass (prefill phase)
            self.cache[layer_idx] = (key, value)
            return key, value

        prev_key, prev_value = self.cache[layer_idx]

        # Concatenate along the sequence length dimension (dim=2)
        full_key = torch.cat([prev_key, key], dim=2)
        full_value = torch.cat([prev_value, value], dim=2)

        self.cache[layer_idx] = (full_key, full_value)
        return full_key, full_value

    def get_seq_len(self, layer_idx: int = 0) -> int:
        """Return the sequence length currently accumulated in the cache.

        Args:
            layer_idx: Layer to inspect (defaults to first layer 0).

        Returns:
            Integer number of cached tokens, or 0 if cache is empty.
        """
        if layer_idx not in self.cache:
            return 0
        return self.cache[layer_idx][0].size(2)

    def reset(self) -> None:
        """Clear all cached tensors to release memory."""
        self.cache.clear()

    def memory_bytes(self) -> int:
        """Calculate total memory consumed by cached tensors in bytes.

        Returns:
            Integer total byte footprint.
        """
        total = 0
        for k, v in self.cache.values():
            total += k.element_size() * k.nelement()
            total += v.element_size() * v.nelement()
        return total

    def __len__(self) -> int:
        """Return the number of layers with active caches."""
        return len(self.cache)
