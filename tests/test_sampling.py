"""Tests for text generation sampling algorithms."""

import pytest
import torch
import torch.nn.functional as F

from forgelm.inference.sampling import (
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_next_token,
)


def test_apply_temperature():
    """Verify temperature scaling changes probability distribution sharpness."""
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    # Lower temperature (< 1.0) sharpens probabilities
    low_temp = apply_temperature(logits, temperature=0.5)
    probs_low = F.softmax(low_temp, dim=-1)

    # Higher temperature (> 1.0) flattens probabilities
    high_temp = apply_temperature(logits, temperature=2.0)
    probs_high = F.softmax(high_temp, dim=-1)

    assert probs_low[0, 2] > probs_high[0, 2]


def test_apply_top_k():
    """Verify top-k sets all non-top-k logits to negative infinity."""
    logits = torch.tensor([[1.0, 5.0, 2.0, 4.0, 3.0]])  # ranks: 5.0, 4.0, 3.0, 2.0, 1.0
    k = 2
    filtered = apply_top_k(logits, k=k)

    # Only tokens with values 5.0 and 4.0 should remain finite
    finite_count = torch.isfinite(filtered).sum().item()
    assert finite_count == k
    assert torch.isfinite(filtered[0, 1])  # 5.0
    assert torch.isfinite(filtered[0, 3])  # 4.0
    assert torch.isneginf(filtered[0, 0])  # 1.0 masked out


def test_apply_top_p():
    """Verify nucleus sampling retains smallest set exceeding cumulative probability p."""
    # Logits where token 0 has ~70% prob, token 1 ~20%, token 2 ~10%
    logits = torch.tensor([[5.0, 2.0, 0.0]])
    filtered = apply_top_p(logits, p=0.6)

    # First token alone exceeds 0.6, so only token 0 should remain
    assert torch.isfinite(filtered[0, 0])
    assert torch.isneginf(filtered[0, 2])


def test_apply_repetition_penalty():
    """Verify repetition penalty reduces probability of previously generated tokens."""
    logits = torch.tensor([[4.0, 4.0]])
    penalized = apply_repetition_penalty(logits, generated_tokens=[0], penalty=2.0)

    # Token 0 should be penalized (reduced score) while token 1 remains unchanged
    assert penalized[0, 0] < penalized[0, 1]
    assert penalized[0, 1] == logits[0, 1]


def test_sample_next_token_greedy():
    """Verify temperature=0.0 deterministically selects the argmax token."""
    logits = torch.tensor([[1.0, 10.0, 2.0]])
    token = sample_next_token(logits, temperature=0.0)

    assert token.shape == (1, 1)
    assert token.item() == 1


def test_sample_next_token_shape():
    """Verify sample_next_token returns correct batch dimension."""
    logits = torch.randn(3, 100)
    tokens = sample_next_token(logits, temperature=0.8, top_k=20, top_p=0.9)

    assert tokens.shape == (3, 1)
    assert (tokens >= 0).all() and (tokens < 100).all()
