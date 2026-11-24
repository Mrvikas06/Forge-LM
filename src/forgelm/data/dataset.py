"""Memory-mapped dataset and binary tokenization utilities for pretraining."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from forgelm.tokenizer.bpe import BPETokenizer
from forgelm.utils.logging import get_logger

logger = get_logger(__name__)


class PretrainingDataset(Dataset):
    """Memory-mapped binary token dataset for autoregressive language model training.

    Reads fixed-length sequence chunks from a raw uint16 binary token file.
    Tokens are mapped via numpy memmap, keeping memory usage constant and minimal
    even with multi-gigabyte corpora.
    """

    def __init__(self, bin_path: str | Path, seq_len: int = 512) -> None:
        """Initialize PretrainingDataset.

        Args:
            bin_path: Path to binary token file (.bin).
            seq_len: Number of tokens per input sequence.
        """
        self.bin_path = Path(bin_path)
        self.seq_len = seq_len

        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary token file not found: {self.bin_path}")

        # Map binary file as uint16 array
        self.data: np.memmap | None = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)

        # Number of non-overlapping chunks of length seq_len
        # We need seq_len + 1 tokens to form (x, y) input-target pairs
        self.num_samples = max(0, (self.total_tokens - 1) // self.seq_len)

        if self.num_samples == 0:
            logger.warning(
                f"Binary file {self.bin_path} has {self.total_tokens} tokens, "
                f"which is fewer than seq_len + 1 ({self.seq_len + 1})."
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch a single input-target sequence pair.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (x, y) where:
                x: Input token IDs of shape (seq_len,)
                y: Target token IDs of shape (seq_len,) shifted right by 1
        """
        if self.data is None:
            raise RuntimeError("Dataset file handle is closed.")

        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")

        start_idx = idx * self.seq_len
        # Slice seq_len + 1 tokens to construct input x and target y
        chunk = self.data[start_idx : start_idx + self.seq_len + 1].astype(np.int64)

        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        return x, y

    def close(self) -> None:
        """Explicitly release underlying memory-mapped file handle (essential for Windows)."""
        if hasattr(self, "data") and self.data is not None:
            # Delete reference to release open OS file lock
            del self.data
            self.data = None

    def __del__(self) -> None:
        self.close()


def tokenize_text_file(
    text_path: str | Path,
    tokenizer: BPETokenizer,
    output_bin_path: str | Path,
    max_lines: int | None = None,
    add_bos: bool = True,
    add_eos: bool = True,
) -> int:
    """Tokenize a text file line-by-line and save as a contiguous uint16 binary file.

    Args:
        text_path: Path to input text file.
        tokenizer: Trained BPETokenizer instance.
        output_bin_path: Destination binary path (.bin).
        max_lines: Optional limit on number of lines to process.
        add_bos: Whether to prepend BOS token to each line.
        add_eos: Whether to append EOS token to each line.

    Returns:
        Total number of tokens written.
    """
    text_path = Path(text_path)
    output_bin_path = Path(output_bin_path)
    output_bin_path.parent.mkdir(parents=True, exist_ok=True)

    if not text_path.exists():
        raise FileNotFoundError(f"Input text file not found: {text_path}")

    all_tokens: list[int] = []
    line_count = 0

    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                tokens = tokenizer.encode(stripped, add_bos=add_bos, add_eos=add_eos)
                all_tokens.extend(tokens)
                line_count += 1

            if max_lines and line_count >= max_lines:
                break

    # Convert to numpy uint16 array (supports vocab sizes up to 65,535)
    token_arr = np.array(all_tokens, dtype=np.uint16)

    # Write binary file
    with open(output_bin_path, "wb") as f_out:
        f_out.write(token_arr.tobytes())

    file_size_mb = output_bin_path.stat().st_size / (1024**2)
    logger.info(
        f"Tokenized {line_count:,} lines -> {len(token_arr):,} tokens "
        f"saved to {output_bin_path} ({file_size_mb:.2f} MB)"
    )

    return len(token_arr)
