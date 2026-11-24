"""Training configuration dataclass for ForgeLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    """Hyperparameters and settings for pretraining ForgeLM.

    Attributes:
        batch_size: Micro-batch size per forward pass.
        gradient_accumulation_steps: Number of forward/backward steps before optimizer step.
        learning_rate: Maximum peak learning rate.
        min_lr_ratio: Minimum learning rate as a fraction of peak learning rate.
        weight_decay: AdamW weight decay coefficient.
        max_steps: Total number of training steps.
        warmup_steps: Number of steps for linear learning rate warmup.
        grad_clip: Maximum gradient norm for clipping.
        eval_interval: Frequency (in steps) to evaluate on validation set.
        eval_steps: Number of batches to evaluate during each evaluation cycle.
        checkpoint_interval: Frequency (in steps) to save model checkpoints.
        mixed_precision: Whether to use Automatic Mixed Precision (AMP) when CUDA is available.
        device: Target execution device ('auto', 'cuda', or 'cpu').
        checkpoint_dir: Directory where checkpoints are saved.
    """

    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5.0e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    max_steps: int = 10000
    warmup_steps: int = 500
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_steps: int = 50
    checkpoint_interval: int = 1000
    mixed_precision: bool = True
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self) -> None:
        """Validate configuration settings."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}"
            )
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size after gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Instantiated TrainingConfig.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
