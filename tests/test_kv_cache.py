"""Tests for Key-Value (KV) Cache management."""

import pytest
import torch

from forgelm.inference.kv_cache import KVCache


def test_kv_cache_initial_state():
    """Verify newly initialized KVCache is empty."""
    cache = KVCache()
    assert len(cache) == 0
    assert cache.get_seq_len(layer_idx=0) == 0
    assert cache.memory_bytes() == 0


def test_kv_cache_update_and_concatenation():
    """Verify cache concatenates sequential token representations along sequence dimension."""
    cache = KVCache()
    b, n_heads, d_head = 2, 4, 16

    # Step 1: Initial prompt of 5 tokens
    k1 = torch.randn(b, n_heads, 5, d_head)
    v1 = torch.randn(b, n_heads, 5, d_head)
    full_k1, full_v1 = cache.update(k1, v1, layer_idx=0)

    assert full_k1.shape == (b, n_heads, 5, d_head)
    assert full_v1.shape == (b, n_heads, 5, d_head)
    assert cache.get_seq_len(layer_idx=0) == 5

    # Step 2: Append 1 new token
    k2 = torch.randn(b, n_heads, 1, d_head)
    v2 = torch.randn(b, n_heads, 1, d_head)
    full_k2, full_v2 = cache.update(k2, v2, layer_idx=0)

    assert full_k2.shape == (b, n_heads, 6, d_head)
    assert full_v2.shape == (b, n_heads, 6, d_head)
    assert cache.get_seq_len(layer_idx=0) == 6

    # Verify tensor contents match concatenation exactly
    assert torch.equal(full_k2[:, :, :5, :], k1)
    assert torch.equal(full_k2[:, :, 5:, :], k2)


def test_kv_cache_multi_layer():
    """Verify different layers maintain isolated key-value states."""
    cache = KVCache()
    b, n_heads, d_head = 1, 2, 8

    k_layer0 = torch.randn(b, n_heads, 3, d_head)
    v_layer0 = torch.randn(b, n_heads, 3, d_head)
    cache.update(k_layer0, v_layer0, layer_idx=0)

    k_layer1 = torch.randn(b, n_heads, 4, d_head)
    v_layer1 = torch.randn(b, n_heads, 4, d_head)
    cache.update(k_layer1, v_layer1, layer_idx=1)

    assert len(cache) == 2
    assert cache.get_seq_len(layer_idx=0) == 3
    assert cache.get_seq_len(layer_idx=1) == 4


def test_kv_cache_reset_and_memory():
    """Verify reset clears state and memory_bytes returns valid positive count."""
    cache = KVCache()
    k = torch.randn(1, 2, 10, 8)
    v = torch.randn(1, 2, 10, 8)
    cache.update(k, v, layer_idx=0)

    mem_before = cache.memory_bytes()
    assert mem_before > 0

    cache.reset()
    assert len(cache) == 0
    assert cache.get_seq_len(0) == 0
    assert cache.memory_bytes() == 0
