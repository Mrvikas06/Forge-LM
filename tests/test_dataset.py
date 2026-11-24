"""Tests for memory-mapped dataset and binary tokenization."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from forgelm.data.dataset import PretrainingDataset, tokenize_text_file
from forgelm.tokenizer.bpe import BPETokenizer


@pytest.fixture
def dummy_tokenizer() -> BPETokenizer:
    tok = BPETokenizer(vocab_size=300)
    tok.train(["hello world", "test sample", "little girl"], min_frequency=1)
    return tok


def test_tokenize_text_file(dummy_tokenizer: BPETokenizer):
    """Verify tokenize_text_file creates valid uint16 binary array."""
    with tempfile.TemporaryDirectory() as tmpdir:
        txt_path = Path(tmpdir) / "sample.txt"
        bin_path = Path(tmpdir) / "sample.bin"

        txt_path.write_text("hello world\ntest sample\n", encoding="utf-8")

        num_tokens = tokenize_text_file(txt_path, dummy_tokenizer, bin_path)
        assert num_tokens > 0
        assert bin_path.exists()

        # Check raw bytes match uint16 length
        data = np.fromfile(bin_path, dtype=np.uint16)
        assert len(data) == num_tokens


def test_pretraining_dataset_slicing():
    """Verify input-target next-token shifting behavior in PretrainingDataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "tokens.bin"

        # Write sequential token IDs [0, 1, 2, ..., 32]
        seq_len = 8
        tokens = np.arange(33, dtype=np.uint16)
        tokens.tofile(bin_path)

        dataset = PretrainingDataset(bin_path, seq_len=seq_len)

        # 33 tokens with seq_len=8 gives (33 - 1) // 8 = 4 samples
        assert len(dataset) == 4

        # Verify sample 0: x should be 0..7, y should be 1..8
        x0, y0 = dataset[0]
        assert isinstance(x0, torch.Tensor)
        assert isinstance(y0, torch.Tensor)
        assert x0.shape == (seq_len,)
        assert y0.shape == (seq_len,)

        assert torch.equal(x0, torch.arange(0, 8, dtype=torch.int64))
        assert torch.equal(y0, torch.arange(1, 9, dtype=torch.int64))

        # Verify next-token property: y[t] == x[t + 1]
        assert torch.equal(x0[1:], y0[:-1])

        # Release file handle before tempdir cleanup on Windows
        dataset.close()


def test_pretraining_dataset_out_of_bounds():
    """Accessing out of bounds indices must raise IndexError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "tokens.bin"
        tokens = np.arange(20, dtype=np.uint16)
        tokens.tofile(bin_path)

        dataset = PretrainingDataset(bin_path, seq_len=10)
        assert len(dataset) == 1

        with pytest.raises(IndexError):
            _ = dataset[1]
        with pytest.raises(IndexError):
            _ = dataset[-1]

        # Release file handle before tempdir cleanup on Windows
        dataset.close()
