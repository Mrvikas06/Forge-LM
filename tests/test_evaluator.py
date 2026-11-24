"""Tests for Evaluator engine."""

import pytest
import torch
from torch.utils.data import TensorDataset

from forgelm.evaluation.evaluator import Evaluator
from forgelm.model.config import ModelConfig
from forgelm.model.transformer import ForgeTransformer


@pytest.fixture
def test_model() -> ForgeTransformer:
    config = ModelConfig(
        vocab_size=32,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
    )
    return ForgeTransformer(config)


@pytest.fixture
def test_dataset() -> TensorDataset:
    b, t = 8, 16
    x = torch.randint(1, 32, (b, t))
    y = torch.randint(1, 32, (b, t))
    return TensorDataset(x, y)


def test_evaluator_metrics_output(test_model: ForgeTransformer, test_dataset: TensorDataset):
    """Verify Evaluator computes all required metric keys properly."""
    evaluator = Evaluator(test_model, device="cpu", mixed_precision=False)
    metrics = evaluator.evaluate(test_dataset, batch_size=4, ignore_index=-1)

    expected_keys = [
        "loss",
        "perplexity",
        "top1_accuracy",
        "top5_accuracy",
        "bpc",
        "total_tokens",
        "duration_sec",
        "tok_per_sec",
    ]
    for key in expected_keys:
        assert key in metrics

    assert metrics["loss"] > 0.0
    assert metrics["perplexity"] >= 1.0
    assert 0.0 <= metrics["top1_accuracy"] <= 1.0
    assert 0.0 <= metrics["top5_accuracy"] <= 1.0
    assert metrics["top5_accuracy"] >= metrics["top1_accuracy"]
    assert metrics["total_tokens"] == 8 * 16


def test_evaluator_max_batches(test_model: ForgeTransformer, test_dataset: TensorDataset):
    """Verify max_batches limit stops evaluation early."""
    evaluator = Evaluator(test_model, device="cpu", mixed_precision=False)
    metrics = evaluator.evaluate(test_dataset, batch_size=2, max_batches=2, ignore_index=-1)

    # 2 batches * 2 batch_size * 16 tokens = 64 tokens
    assert metrics["total_tokens"] == 64


def test_evaluator_empty_dataset(test_model: ForgeTransformer):
    """Verify handling of empty dataset without crashing."""
    evaluator = Evaluator(test_model, device="cpu", mixed_precision=False)
    empty_dataset = TensorDataset(torch.empty(0, 16, dtype=torch.long), torch.empty(0, 16, dtype=torch.long))
    metrics = evaluator.evaluate(empty_dataset, batch_size=2)

    assert metrics["total_tokens"] == 0
    assert metrics["top1_accuracy"] == 0.0
