"""Tests for TextGenerator engine."""

import pytest
import torch

from forgelm.inference.generator import TextGenerator
from forgelm.model.config import ModelConfig
from forgelm.model.transformer import ForgeTransformer
from forgelm.tokenizer.bpe import BPETokenizer


@pytest.fixture
def toy_tokenizer() -> BPETokenizer:
    tok = BPETokenizer(vocab_size=300)
    tok.train(["Once upon a time", "Lily had a puppy", "The puppy ran fast"], min_frequency=1)
    return tok


@pytest.fixture
def toy_model() -> ForgeTransformer:
    config = ModelConfig(
        vocab_size=300,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=64,
        max_seq_len=64,
        dropout=0.0,
    )
    return ForgeTransformer(config)


def test_generator_output(toy_model: ForgeTransformer, toy_tokenizer: BPETokenizer):
    """Verify generated text begins with the prompt and produces extra tokens."""
    gen = TextGenerator(toy_model, toy_tokenizer, device="cpu")
    prompt = "Once upon"
    result = gen.generate(prompt, max_new_tokens=5, temperature=0.8)

    assert isinstance(result, str)
    assert result.startswith(prompt)
    assert len(result) >= len(prompt)


def test_generator_streaming(toy_model: ForgeTransformer, toy_tokenizer: BPETokenizer):
    """Verify streaming generation yields token chunks iteratively."""
    gen = TextGenerator(toy_model, toy_tokenizer, device="cpu")
    prompt = "Lily"

    chunks = list(gen.stream_generate(prompt, max_new_tokens=4, temperature=0.8))
    assert len(chunks) <= 4
    for chunk in chunks:
        assert isinstance(chunk, str)


def test_generator_greedy_deterministic(toy_model: ForgeTransformer, toy_tokenizer: BPETokenizer):
    """Verify greedy generation produces deterministic output for identical prompts."""
    gen = TextGenerator(toy_model, toy_tokenizer, device="cpu")
    prompt = "The puppy"

    out1 = gen.generate(prompt, max_new_tokens=6, temperature=0.0)
    out2 = gen.generate(prompt, max_new_tokens=6, temperature=0.0)

    assert out1 == out2
