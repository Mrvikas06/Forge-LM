"""Comprehensive tests for the custom Byte-Pair Encoding (BPE) tokenizer."""

import tempfile
from pathlib import Path

import pytest

from forgelm.tokenizer.bpe import BPETokenizer
from forgelm.tokenizer.special_tokens import (
    BASE_VOCAB_SIZE,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
)


@pytest.fixture
def base_tokenizer() -> BPETokenizer:
    """Returns a tokenizer with only the base vocabulary (no merges)."""
    return BPETokenizer(vocab_size=BASE_VOCAB_SIZE)


@pytest.fixture
def trained_tokenizer() -> BPETokenizer:
    """Returns a tokenizer trained on a small repetitive toy corpus."""
    corpus = [
        "Once upon a time, there was a little girl named Lily.",
        "Lily loved to play with her little dog in the garden.",
        "One day, Lily found a little red ball under a tree.",
        "The dog ran fast and brought the little ball back to Lily.",
        "They were both very happy and played all afternoon.",
    ]
    tok = BPETokenizer(vocab_size=300)
    tok.train(corpus, min_frequency=2)
    return tok


def test_base_vocab_size(base_tokenizer: BPETokenizer):
    """Base vocabulary must have 260 tokens (4 special + 256 bytes)."""
    assert len(base_tokenizer) == BASE_VOCAB_SIZE
    assert base_tokenizer.pad_id == 0
    assert base_tokenizer.bos_id == 1
    assert base_tokenizer.eos_id == 2
    assert base_tokenizer.unk_id == 3


def test_invalid_vocab_size():
    """Initializing with vocab size less than 260 must raise ValueError."""
    with pytest.raises(ValueError):
        BPETokenizer(vocab_size=100)


def test_byte_fallback_roundtrip(base_tokenizer: BPETokenizer):
    """Even with zero merges, any UTF-8 string must decode back identically."""
    texts = [
        "Hello, World!",
        "123 + 456 = 579",
        "Line 1\nLine 2\tIndented",
        "Bonjour le monde! Ça va bien?",
        "Mathematical symbols: α + β = γ",
    ]
    for text in texts:
        tokens = base_tokenizer.encode(text)
        decoded = base_tokenizer.decode(tokens)
        assert decoded == text


def test_bpe_training_compresses_tokens(trained_tokenizer: BPETokenizer):
    """Training BPE merges must reduce the number of tokens compared to byte-level."""
    text = "Lily found a little ball and played with the little dog."
    base_tok = BPETokenizer(vocab_size=BASE_VOCAB_SIZE)

    tokens_base = base_tok.encode(text)
    tokens_trained = trained_tokenizer.encode(text)

    # Merged tokens must require fewer IDs than raw bytes
    assert len(tokens_trained) < len(tokens_base)

    # Roundtrip must be exact
    decoded = trained_tokenizer.decode(tokens_trained)
    assert decoded == text


def test_special_tokens_handling(trained_tokenizer: BPETokenizer):
    """Test BOS and EOS insertion and decoding behavior."""
    text = "Lily and her dog."
    tokens = trained_tokenizer.encode(text, add_bos=True, add_eos=True)

    assert tokens[0] == trained_tokenizer.bos_id
    assert tokens[-1] == trained_tokenizer.eos_id

    # Decoding with skip_special_tokens=True restores original string
    assert trained_tokenizer.decode(tokens, skip_special_tokens=True) == text

    # Decoding with skip_special_tokens=False retains <bos> and <eos>
    decoded_with_special = trained_tokenizer.decode(tokens, skip_special_tokens=False)
    assert decoded_with_special.startswith(BOS_TOKEN)
    assert decoded_with_special.endswith(EOS_TOKEN)


def test_empty_string_handling(trained_tokenizer: BPETokenizer):
    """Empty strings should encode to empty token lists (unless special tokens requested)."""
    assert trained_tokenizer.encode("") == []
    assert trained_tokenizer.decode([]) == ""

    tokens_with_bos = trained_tokenizer.encode("", add_bos=True, add_eos=True)
    assert tokens_with_bos == [trained_tokenizer.bos_id, trained_tokenizer.eos_id]


def test_serialization_roundtrip(trained_tokenizer: BPETokenizer):
    """Saving and loading a trained tokenizer must preserve exact behavior."""
    test_text = "Once upon a time, Lily found a ball."

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_tok.json"
        trained_tokenizer.save(save_path)

        assert save_path.exists()

        loaded_tok = BPETokenizer.load(save_path)

        assert len(loaded_tok) == len(trained_tokenizer)
        assert loaded_tok.vocab_size == trained_tokenizer.vocab_size
        assert loaded_tok.merges == trained_tokenizer.merges

        orig_encoded = trained_tokenizer.encode(test_text)
        loaded_encoded = loaded_tok.encode(test_text)
        assert orig_encoded == loaded_encoded

        assert loaded_tok.decode(loaded_encoded) == test_text
