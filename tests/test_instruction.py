"""Tests for instruction formatting and response-only loss masking."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from forgelm.data.instruction import (
    ASSISTANT_TAG,
    USER_TAG,
    InstructionDataset,
    format_instruction,
)
from forgelm.tokenizer.bpe import BPETokenizer


@pytest.fixture
def sample_tokenizer() -> BPETokenizer:
    tok = BPETokenizer(vocab_size=300)
    tok.train(
        [
            f"{USER_TAG}\nTell me a story\n{ASSISTANT_TAG}\nOnce upon a time there was a puppy.",
            f"{USER_TAG}\nWho are you?\n{ASSISTANT_TAG}\nI am ForgeLM.",
        ],
        min_frequency=1,
    )
    return tok


def test_format_instruction_prompt_only():
    """Verify template formatting when only instruction is provided."""
    prompt = format_instruction("Explain gravity.")
    assert prompt.startswith(f"{USER_TAG}\n")
    assert "Explain gravity." in prompt
    assert prompt.endswith(f"\n{ASSISTANT_TAG}\n")


def test_format_instruction_with_response():
    """Verify template formatting when both instruction and response are provided."""
    full = format_instruction("Hi", "Hello!")
    assert f"{USER_TAG}\nHi\n{ASSISTANT_TAG}\nHello!" in full


def test_instruction_dataset_loss_masking(sample_tokenizer: BPETokenizer):
    """Verify prompt tokens are masked out with pad_token_id in target tensor."""
    records = [
        {"instruction": "Tell me a story", "response": "Once upon a time there was a puppy."},
    ]
    max_len = 32
    dataset = InstructionDataset(
        data=records,
        tokenizer=sample_tokenizer,
        max_seq_len=max_len,
        pad_token_id=0,
    )

    assert len(dataset) == 1
    x, y = dataset[0]

    assert x.shape == (max_len,)
    assert y.shape == (max_len,)

    # The prompt should be masked with 0 in y
    # Count how many prompt tokens exist
    prompt_str = format_instruction("Tell me a story")
    prompt_ids = sample_tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
    prompt_len = len(prompt_ids)

    # First (prompt_len - 1) tokens in y MUST be 0 (masked out)
    assert (y[: prompt_len - 1] == 0).all()

    # The response tokens in y must NOT be all 0 (active loss)
    response_targets = y[prompt_len - 1 : prompt_len + 3]
    assert (response_targets != 0).any()


def test_instruction_dataset_json_file(sample_tokenizer: BPETokenizer):
    """Verify loading dataset from a JSON file on disk."""
    records = [
        {"instruction": "Who are you?", "response": "I am ForgeLM."},
        {"instruction": "Hi", "response": "Hello!"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_instruct.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f)

        dataset = InstructionDataset(
            data=json_path,
            tokenizer=sample_tokenizer,
            max_seq_len=24,
            pad_token_id=0,
        )

        assert len(dataset) == 2
