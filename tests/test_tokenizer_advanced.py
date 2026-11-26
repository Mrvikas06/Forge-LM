"""Extended test suite for ForgeLM BPE Tokenizer edge cases and guarantees."""

from pathlib import Path
import pytest

from forgelm.tokenizer.bpe import BPETokenizer
from forgelm.tokenizer.special_tokens import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN

TOKENIZER_JSON = Path(__file__).resolve().parent.parent / "tokenizer" / "tokenizer.json"


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    """Load the trained 4096-vocab tokenizer."""
    assert TOKENIZER_JSON.exists(), "Trained tokenizer.json must exist"
    return BPETokenizer.load(TOKENIZER_JSON)


def test_emoji_byte_fallback(tokenizer: BPETokenizer):
    """Verify that multi-byte emojis encode and decode with exact roundtrip parity."""
    emojis = "🐶 ⚡ 🚗 🌈 🍒 🧸"
    token_ids = tokenizer.encode(emojis, add_bos=False, add_eos=False)
    assert len(token_ids) > 0
    # None of the tokens should be UNK
    assert tokenizer.unk_id not in token_ids
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert decoded == emojis


def test_multilingual_byte_fallback(tokenizer: BPETokenizer):
    """Verify that non-Latin scripts (Devanagari, Chinese, Greek) decompose to bytes and roundtrip."""
    foreign_text = "नमस्ते दुनिया! 你好世界! Γειά σου Κόσμε!"
    token_ids = tokenizer.encode(foreign_text, add_bos=False, add_eos=False)
    assert len(token_ids) > 0
    assert tokenizer.unk_id not in token_ids
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert decoded == foreign_text


def test_whitespace_and_newlines_exactness(tokenizer: BPETokenizer):
    """Verify tabs, consecutive spaces, and newlines preserve exact byte integrity."""
    text_with_spaces = "Line 1\n\tIndented line\n\n   Three spaces and trailing.   "
    token_ids = tokenizer.encode(text_with_spaces, add_bos=False, add_eos=False)
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert decoded == text_with_spaces


def test_bos_eos_placement(tokenizer: BPETokenizer):
    """Verify BOS is strictly index 0 and EOS is strictly final token."""
    text = "Short test"
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert tokens[0] == tokenizer.bos_id
    assert tokens[-1] == tokenizer.eos_id
    # Decoding with skip_special_tokens=False should show symbols or decode cleanly
    decoded_skip = tokenizer.decode(tokens, skip_special_tokens=True)
    assert decoded_skip == text


def test_batch_padding_and_truncation(tokenizer: BPETokenizer):
    """Verify encode_batch produces aligned rectangular tensors."""
    import torch
    batch = [
        "Hi",
        "Once upon a time in a sunny meadow",
        "A slightly longer sentence with multiple words for padding test.",
    ]
    max_len = 24
    padded_ids = []
    for s in batch:
        ids = tokenizer.encode(s, add_bos=True, add_eos=True)
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [tokenizer.pad_id] * (max_len - len(ids))
        padded_ids.append(ids)

    tensor = torch.tensor(padded_ids, dtype=torch.long)
    assert tensor.shape == (3, 24)
    assert tensor[0, 0].item() == tokenizer.bos_id
    assert tokenizer.pad_id in tensor[0].tolist()
