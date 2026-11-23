"""Model configuration definition for ForgeLM Transformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Hyperparameters for ForgeLM Transformer decoder model.

    Attributes:
        vocab_size: Number of unique tokens in vocabulary.
        d_model: Hidden dimension (embedding dimension).
        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        d_ff: Feed-forward hidden dimension (defaults to 4 * d_model).
        max_seq_len: Maximum context window sequence length.
        dropout: Dropout probability.
        bias: Whether to include bias terms in linear layers.
        pad_token_id: Special token ID used for padding.
        bos_token_id: Special token ID for beginning of sequence.
        eos_token_id: Special token ID for end of sequence.
        unk_token_id: Special token ID for unknown tokens.
    """

    vocab_size: int = 4096
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 1536
    max_seq_len: int = 512
    dropout: float = 0.1
    bias: bool = False
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

    @property
    def d_head(self) -> int:
        """Dimension of each individual attention head."""
        return self.d_model // self.n_heads

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        """Load ModelConfig from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Instantiated ModelConfig.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        # Filter only fields that belong to ModelConfig
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
