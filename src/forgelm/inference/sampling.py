"""Sampling algorithms for autoregressive text generation.

Provides temperature scaling, top-k filtering, nucleus (top-p) sampling,
and repetition penalty mechanisms.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: Sequence[int] | torch.Tensor,
    penalty: float = 1.0,
) -> torch.Tensor:
    """Penalize logits of previously generated tokens to prevent repetition loops.

    Args:
        logits: Unnormalized log probabilities of shape (batch_size, vocab_size).
        generated_tokens: Sequence or tensor of token IDs already generated.
        penalty: Penalty multiplier (1.0 = no penalty, >1.0 = discourage repetition).

    Returns:
        Tensor of penalized logits.
    """
    if penalty == 1.0 or not len(generated_tokens):
        return logits

    logits = logits.clone()

    if isinstance(generated_tokens, torch.Tensor):
        unique_tokens = torch.unique(generated_tokens).tolist()
    else:
        unique_tokens = list(set(generated_tokens))

    for token_id in unique_tokens:
        # If score is positive, divide by penalty; if negative, multiply by penalty
        score = logits[:, token_id]
        logits[:, token_id] = torch.where(score > 0, score / penalty, score * penalty)

    return logits


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Scale logits by temperature parameter.

    Args:
        logits: Unnormalized logits of shape (batch_size, vocab_size).
        temperature: Temperature value (> 0.0). Lower sharpens, higher flattens.

    Returns:
        Temperature-scaled logits.
    """
    if temperature <= 0.0:
        return logits
    return logits / max(temperature, 1e-5)


def apply_top_k(logits: torch.Tensor, k: int = 0) -> torch.Tensor:
    """Filter logits keeping only the top-k highest probability tokens.

    Args:
        logits: Unnormalized logits of shape (batch_size, vocab_size).
        k: Number of highest probability tokens to keep. If <= 0, no filtering.

    Returns:
        Logits with non-top-k values set to -infinity.
    """
    if k <= 0 or k >= logits.size(-1):
        return logits

    logits = logits.clone()
    top_values, _ = torch.topk(logits, k, dim=-1)
    min_values = top_values[:, -1].unsqueeze(-1)
    logits[logits < min_values] = float("-inf")
    return logits


def apply_top_p(logits: torch.Tensor, p: float = 0.0) -> torch.Tensor:
    """Apply nucleus (top-p) sampling to filter candidate tokens.

    Retains the smallest set of tokens whose cumulative probability exceeds p.

    Args:
        logits: Unnormalized logits of shape (batch_size, vocab_size).
        p: Cumulative probability threshold (0.0 to 1.0). If <= 0.0 or >= 1.0, no filtering.

    Returns:
        Logits with non-nucleus values set to -infinity.
    """
    if p <= 0.0 or p >= 1.0:
        return logits

    logits = logits.clone()
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask out tokens with cumulative probability above threshold p
    # Shift right to always keep at least the first token exceeding p
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    # Scatter mask back to original logit positions
    mask = sorted_mask.scatter(dim=-1, index=sorted_indices, src=sorted_mask)
    logits[mask] = float("-inf")
    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    generated_tokens: Sequence[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Select the next token ID from logits using full sampling pipeline.

    Args:
        logits: Unnormalized predictions of shape (batch_size, vocab_size).
        temperature: Temperature scaling factor (0.0 = greedy argmax).
        top_k: Top-k candidate truncation.
        top_p: Nucleus cumulative probability threshold.
        repetition_penalty: Repetition penalty coefficient.
        generated_tokens: Prior tokens in the sequence for repetition penalty.

    Returns:
        Tensor of next token IDs of shape (batch_size, 1).
    """
    # 1. Apply repetition penalty
    if generated_tokens is not None and repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated_tokens, penalty=repetition_penalty)

    # 2. Greedy decoding when temperature is zero
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 3. Apply temperature scaling
    logits = apply_temperature(logits, temperature=temperature)

    # 4. Apply top-k filtering
    if top_k > 0:
        logits = apply_top_k(logits, k=top_k)

    # 5. Apply top-p (nucleus) filtering
    if 0.0 < top_p < 1.0:
        logits = apply_top_p(logits, p=top_p)

    # 6. Sample from filtered probability distribution
    probs = F.softmax(logits, dim=-1)

    # Handle numerical edge cases where all probabilities became 0
    if torch.isnan(probs).any() or probs.sum() == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
