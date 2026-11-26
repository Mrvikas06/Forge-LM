"""Optimizer setup and learning rate scheduling for pretraining."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from forgelm.utils.logging import get_logger

logger = get_logger(__name__)


def configure_optimizers(
    model: nn.Module,
    learning_rate: float = 5.0e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    """Configure AdamW optimizer with selective weight decay.

    Splits model parameters into two groups:
    - 2D parameters (weights of linear layers and embeddings) will be decayed.
    - 1D parameters (biases and LayerNorm scale/shift) will NOT be decayed.

    Args:
        model: PyTorch model.
        learning_rate: Peak learning rate.
        weight_decay: Weight decay coefficient for 2D parameters.
        betas: Adam beta coefficients (default: (0.9, 0.95) for language models).
        eps: Epsilon value for numerical stability.

    Returns:
        Configured AdamW optimizer instance.
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases and 1D parameters (e.g. LayerNorm scale) do not get weight decay
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    num_decay = sum(p.numel() for p in decay_params)
    num_nodecay = sum(p.numel() for p in no_decay_params)
    logger.info(
        f"Optimizer configuration: {len(decay_params)} decay tensors ({num_decay:,} params), "
        f"{len(no_decay_params)} no-decay tensors ({num_nodecay:,} params)"
    )

    optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)
    return optimizer


def calculate_cosine_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """Calculate the learning rate at a given step using cosine decay with warmup."""
    if step < warmup_steps:
        factor = float(step + 1) / float(max(1, warmup_steps))
    elif step > total_steps:
        factor = min_lr_ratio
    else:
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        factor = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return peak_lr * factor


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a learning rate scheduler with linear warmup followed by cosine annealing.

    Formula:
        step < warmup_steps:
            lr = lr_max * (step / warmup_steps)
        warmup_steps <= step <= max_steps:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
        step > max_steps:
            lr = lr_min

    Args:
        optimizer: PyTorch optimizer.
        warmup_steps: Number of initial warmup steps.
        max_steps: Total number of training steps.
        min_lr_ratio: Minimum learning rate as a fraction of peak learning rate.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step + 1) / float(max(1, warmup_steps))

        if current_step > max_steps:
            # Post-schedule constant minimum
            return min_lr_ratio

        # Cosine decay down to min_lr_ratio
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get the current learning rate from the first parameter group."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0
