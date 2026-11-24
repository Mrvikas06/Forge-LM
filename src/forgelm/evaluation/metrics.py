"""Evaluation metrics for language model assessment."""

from __future__ import annotations

import math
from typing import Sequence

import torch


def calculate_perplexity(loss: float, max_loss: float = 20.0) -> float:
    """Calculate perplexity from cross-entropy loss.

    Formula:
        Perplexity = exp(loss)

    Args:
        loss: Cross-entropy loss value.
        max_loss: Cap on loss to avoid floating point overflow during exp().

    Returns:
        Perplexity value (>= 1.0).
    """
    if math.isnan(loss) or math.isinf(loss):
        return float("inf")
    clamped_loss = min(loss, max_loss)
    return math.exp(clamped_loss)


def calculate_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Calculate top-1 next-token prediction accuracy.

    Args:
        logits: Model predictions of shape (batch_size, seq_len, vocab_size) or (N, vocab_size).
        targets: Target token IDs of shape (batch_size, seq_len) or (N,).
        ignore_index: Token ID to ignore during accuracy calculation (e.g., pad token).

    Returns:
        Accuracy float between 0.0 and 1.0.
    """
    if logits.ndim == 3:
        logits = logits.view(-1, logits.size(-1))
    if targets.ndim > 1:
        targets = targets.view(-1)

    # Filter out ignored indices
    valid_mask = targets != ignore_index
    if not valid_mask.any():
        return 0.0

    preds = torch.argmax(logits, dim=-1)
    correct = (preds[valid_mask] == targets[valid_mask]).sum().item()
    total = valid_mask.sum().item()

    return correct / total if total > 0 else 0.0


def calculate_topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
    ignore_index: int = -100,
) -> float:
    """Calculate top-k next-token prediction accuracy.

    Args:
        logits: Model predictions of shape (batch_size, seq_len, vocab_size) or (N, vocab_size).
        targets: Target token IDs of shape (batch_size, seq_len) or (N,).
        k: Number of highest probability candidates to consider.
        ignore_index: Token ID to ignore during accuracy calculation.

    Returns:
        Top-k accuracy float between 0.0 and 1.0.
    """
    if logits.ndim == 3:
        logits = logits.view(-1, logits.size(-1))
    if targets.ndim > 1:
        targets = targets.view(-1)

    valid_mask = targets != ignore_index
    if not valid_mask.any():
        return 0.0

    # Get top-k predictions along vocabulary dimension
    _, topk_preds = torch.topk(logits, k=k, dim=-1)  # (N, k)

    # Check if target is in top-k predictions
    target_expanded = targets.unsqueeze(-1)  # (N, 1)
    correct_matrix = (topk_preds == target_expanded)[valid_mask]  # (num_valid, k)
    correct = correct_matrix.any(dim=-1).sum().item()
    total = valid_mask.sum().item()

    return correct / total if total > 0 else 0.0


def calculate_bpc(loss: float) -> float:
    """Convert natural-log cross-entropy loss to bits-per-token (or bits-per-character).

    Formula:
        BPC = loss / ln(2) = loss * log2(e)

    Args:
        loss: Cross-entropy loss in natural log (nats).

    Returns:
        Loss in bits.
    """
    if math.isnan(loss) or math.isinf(loss):
        return float("inf")
    return loss / math.log(2.0)
