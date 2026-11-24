"""Inference and text generation subpackage for ForgeLM."""

from forgelm.inference.generator import TextGenerator
from forgelm.inference.sampling import (
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_next_token,
)

__all__ = [
    "TextGenerator",
    "sample_next_token",
    "apply_temperature",
    "apply_top_k",
    "apply_top_p",
    "apply_repetition_penalty",
]
