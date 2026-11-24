"""Evaluation module for ForgeLM."""

from forgelm.evaluation.evaluator import Evaluator
from forgelm.evaluation.metrics import (
    calculate_accuracy,
    calculate_bpc,
    calculate_perplexity,
    calculate_topk_accuracy,
)

__all__ = [
    "Evaluator",
    "calculate_perplexity",
    "calculate_accuracy",
    "calculate_topk_accuracy",
    "calculate_bpc",
]
