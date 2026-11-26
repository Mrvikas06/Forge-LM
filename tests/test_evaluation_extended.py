"""Extended tests for evaluation metrics and diversity calculations."""

import math
import pytest
import torch

from forgelm.evaluation.metrics import (
    calculate_accuracy,
    calculate_bpc,
    calculate_distinct_ngrams,
    calculate_perplexity,
    calculate_repetition_rate,
    calculate_topk_accuracy,
)


def test_calculate_perplexity_edge_cases():
    """Verify perplexity bounds and non-finite protections."""
    assert calculate_perplexity(0.0) == 1.0
    assert pytest.approx(calculate_perplexity(1.0), rel=1e-3) == math.e
    assert calculate_perplexity(float("nan")) == float("inf")
    assert calculate_perplexity(float("inf")) == float("inf")
    # Loss capping at 20.0 prevents overflow
    assert calculate_perplexity(50.0) == math.exp(20.0)


def test_calculate_bpc():
    """Verify BPC translates nats to bits with log2 scaling."""
    loss = 1.0  # 1 nat
    # 1 nat / ln(2) = 1.442695 bits
    assert pytest.approx(calculate_bpc(loss), rel=1e-3) == 1.442695
    assert calculate_bpc(float("nan")) == float("inf")


def test_calculate_distinct_ngrams():
    """Verify lexical diversity Distinct-N calculation."""
    # Completely repetitive sequence
    repeated = ["the", "cat", "the", "cat", "the", "cat"]
    # Bigrams: ("the", "cat"), ("cat", "the"), ("the", "cat"), ("cat", "the"), ("the", "cat") -> 2 unique out of 5
    distinct_2 = calculate_distinct_ngrams(repeated, n=2)
    assert pytest.approx(distinct_2, rel=1e-2) == 2 / 5

    # Completely unique sequence
    unique_words = ["apple", "banana", "cherry", "date", "elderberry"]
    assert calculate_distinct_ngrams(unique_words, n=2) == 1.0

    # Sequence shorter than n
    assert calculate_distinct_ngrams(["single"], n=2) == 1.0
    assert calculate_distinct_ngrams([], n=2) == 0.0


def test_calculate_repetition_rate():
    """Verify lookback window repetition frequency."""
    # Fully repetitive sequence: [1, 1, 1, 1]
    repetitive_seq = [1, 1, 1, 1, 1]
    assert calculate_repetition_rate(repetitive_seq, window_size=2) == 1.0

    # Completely non-repeating sequence
    clean_seq = [1, 2, 3, 4, 5, 6, 7, 8]
    assert calculate_repetition_rate(clean_seq, window_size=4) == 0.0

    # Short sequence
    assert calculate_repetition_rate([42], window_size=5) == 0.0


def test_topk_accuracy_matching():
    """Verify top-k accuracy correctly flags target in top-5."""
    # 3 tokens, vocab 10
    logits = torch.zeros((3, 10))
    # For token 0, let highest 5 logits be 0, 1, 2, 3, 4
    logits[0, :5] = 10.0
    # For token 1, let highest 5 logits be 5, 6, 7, 8, 9
    logits[1, 5:] = 10.0
    # For token 2, let highest 5 logits be 0, 1, 2, 3, 4
    logits[2, :5] = 10.0

    targets = torch.tensor([2, 9, 8])  # targets: 2 (in top 5), 9 (in top 5), 8 (NOT in top 5 for index 2)
    acc = calculate_topk_accuracy(logits, targets, k=5)
    # 2 out of 3 match
    assert pytest.approx(acc, rel=1e-3) == 2 / 3
