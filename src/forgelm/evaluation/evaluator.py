"""Evaluator engine for comprehensive language model assessment."""

from __future__ import annotations

import time
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from forgelm.evaluation.metrics import (
    calculate_accuracy,
    calculate_bpc,
    calculate_perplexity,
    calculate_topk_accuracy,
)
from forgelm.utils.logging import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Evaluates causal language models across validation and benchmark datasets."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str | None = None,
        mixed_precision: bool = True,
    ) -> None:
        """Initialize Evaluator.

        Args:
            model: PyTorch model.
            device: Target evaluation device.
            mixed_precision: Whether to use AMP on CUDA.
        """
        self.model = model

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model.to(self.device)
        self.mixed_precision = mixed_precision and self.device.type == "cuda"

    def evaluate(
        self,
        data: Dataset | DataLoader,
        batch_size: int = 16,
        max_batches: int | None = None,
        ignore_index: int = 0,
    ) -> dict[str, float]:
        """Run evaluation across dataset or DataLoader.

        Args:
            data: Dataset or DataLoader to evaluate.
            batch_size: Batch size to use if Dataset is passed.
            max_batches: Optional limit on number of batches.
            ignore_index: Token ID to ignore in loss and accuracy calculations (e.g. pad token).

        Returns:
            Dictionary with evaluation metrics:
                loss, perplexity, top1_accuracy, top5_accuracy, bpc,
                total_tokens, duration_sec, tok_per_sec.
        """
        if isinstance(data, DataLoader):
            loader = data
        else:
            loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        self.model.eval()

        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        total_tokens = 0
        num_batches = 0

        t0 = time.time()

        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break

                x = x.to(self.device)
                y = y.to(self.device)

                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.mixed_precision,
                ):
                    logits, loss = self.model(x, targets=y)

                if loss is not None and not torch.isnan(loss):
                    total_loss += loss.item()
                    top1 = calculate_accuracy(logits, y, ignore_index=ignore_index)
                    top5 = calculate_topk_accuracy(logits, y, k=5, ignore_index=ignore_index)

                    total_top1 += top1
                    total_top5 += top5
                    total_tokens += (y != ignore_index).sum().item()
                    num_batches += 1

        duration = time.time() - t0

        if num_batches == 0:
            return {
                "loss": float("nan"),
                "perplexity": float("nan"),
                "top1_accuracy": 0.0,
                "top5_accuracy": 0.0,
                "bpc": float("nan"),
                "total_tokens": 0,
                "duration_sec": duration,
                "tok_per_sec": 0.0,
            }

        avg_loss = total_loss / num_batches
        avg_top1 = total_top1 / num_batches
        avg_top5 = total_top5 / num_batches
        tok_per_sec = total_tokens / duration if duration > 0 else 0.0

        metrics = {
            "loss": avg_loss,
            "perplexity": calculate_perplexity(avg_loss),
            "top1_accuracy": avg_top1,
            "top5_accuracy": avg_top5,
            "bpc": calculate_bpc(avg_loss),
            "total_tokens": total_tokens,
            "duration_sec": duration,
            "tok_per_sec": tok_per_sec,
        }

        return metrics
