"""Tests for evaluation metrics functions."""

import math
import pytest
import torch

from forgelm.evaluation.metrics import (
    calculate_accuracy,
    calculate_bpc,
    calculate_perplexity,
    calculate_topk_accuracy,
)


def test_calculate_perplexity():
    """Verify perplexity computation and boundary handling."""
    # Loss 0.0 -> PPL 1.0
    assert abs(calculate_perplexity(0.0) - 1.0) < 1e-5

    # Loss ln(10) -> PPL 10.0
    assert abs(calculate_perplexity(math.log(10.0)) - 10.0) < 1e-4

    # High loss is clamped to avoid overflow
    assert calculate_perplexity(100.0) == math.exp(20.0)

    # NaN / Inf handling
    assert calculate_perplexity(float("nan")) == float("inf")
    assert calculate_perplexity(float("inf")) == float("inf")


def test_calculate_accuracy():
    """Verify top-1 accuracy calculations."""
    # Logits where argmax matches targets perfectly
    logits = torch.tensor([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    targets = torch.tensor([0, 1, 2])
    assert calculate_accuracy(logits, targets) == 1.0

    # Completely wrong predictions
    wrong_targets = torch.tensor([1, 2, 0])
    assert calculate_accuracy(logits, wrong_targets) == 0.0

    # 2 out of 3 correct
    partial_targets = torch.tensor([0, 1, 0])
    assert abs(calculate_accuracy(logits, partial_targets) - (2 / 3)) < 1e-5


def test_calculate_accuracy_ignore_index():
    """Verify ignore_index masks tokens out of accuracy."""
    logits = torch.tensor([
        [10.0, 0.0],
        [0.0, 10.0],
    ])
    targets = torch.tensor([0, 0])  # First is correct, second is wrong
    assert calculate_accuracy(logits, targets, ignore_index=0) == 0.0  # target 0 is ignored


def test_calculate_topk_accuracy():
    """Verify top-k prediction accuracy."""
    # Vocab size 5
    logits = torch.tensor([
        [5.0, 4.0, 3.0, 2.0, 1.0],  # rank order: 0, 1, 2, 3, 4
    ])

    # Top-1 match
    assert calculate_topk_accuracy(logits, torch.tensor([0]), k=1) == 1.0
    # Inside top-3 (token 2 is 3rd best)
    assert calculate_topk_accuracy(logits, torch.tensor([2]), k=3) == 1.0
    # Outside top-3
    assert calculate_topk_accuracy(logits, torch.tensor([3]), k=3) == 0.0
    # Inside top-5
    assert calculate_topk_accuracy(logits, torch.tensor([3]), k=5) == 1.0


def test_calculate_bpc():
    """Verify bits-per-character/token math."""
    # 1 nat = 1 / ln(2) bits = ~1.442695 bits
    assert abs(calculate_bpc(math.log(2.0)) - 1.0) < 1e-5
    assert calculate_bpc(float("nan")) == float("inf")
