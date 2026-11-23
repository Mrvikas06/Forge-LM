"""Byte-Pair Encoding (BPE) tokenizer implemented from scratch.

Uses byte-level fallback so any UTF-8 string can be encoded and decoded
without out-of-vocabulary errors.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from forgelm.tokenizer.special_tokens import (
    BASE_VOCAB_SIZE,
    BOS_TOKEN,
    DEFAULT_SPECIAL_TOKENS,
    EOS_TOKEN,
    NUM_SPECIAL_TOKENS,
    PAD_TOKEN,
    UNK_TOKEN,
)

# Standard regex pattern for pre-tokenization (splits into words, punctuation, whitespace)
SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


def _byte_to_token_id(b: int) -> int:
    """Map a raw byte value (0..255) to its initial token ID."""
    return b + NUM_SPECIAL_TOKENS


def _token_id_to_byte(token_id: int) -> int:
    """Map an initial byte token ID back to its raw byte value (0..255)."""
    return token_id - NUM_SPECIAL_TOKENS


class BPETokenizer:
    """Byte-level Byte-Pair Encoding Tokenizer.

    Attributes:
        vocab_size: Target vocabulary size.
        special_tokens: Mapping from special token string to integer ID.
        merges: Mapping from token ID pair (id1, id2) to merged token ID.
        vocab: Mapping from token ID to raw byte sequence.
        inverse_vocab: Mapping from token byte sequence to token ID.
    """

    def __init__(self, vocab_size: int = 4096) -> None:
        """Initialize tokenizer with base vocabulary."""
        if vocab_size < BASE_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be at least {BASE_VOCAB_SIZE} (got {vocab_size})"
            )

        self.vocab_size = vocab_size
        self.special_tokens = dict(DEFAULT_SPECIAL_TOKENS)
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

        # Merges: (id1, id2) -> new_id
        self.merges: dict[tuple[int, int], int] = {}

        # Vocabulary: token_id -> bytes
        self.vocab: dict[int, bytes] = {}
        self._init_base_vocab()

    def _init_base_vocab(self) -> None:
        """Initialize the base vocabulary with special tokens and all 256 single bytes."""
        self.vocab.clear()
        # Special tokens are stored as empty or placeholder bytes
        for name, token_id in self.special_tokens.items():
            self.vocab[token_id] = name.encode("utf-8")

        # 256 raw bytes
        for b in range(256):
            token_id = _byte_to_token_id(b)
            self.vocab[token_id] = bytes([b])

    @property
    def pad_id(self) -> int:
        return self.special_tokens[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.special_tokens[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.special_tokens[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.special_tokens[UNK_TOKEN]

    def __len__(self) -> int:
        """Return the current number of tokens in vocabulary."""
        return len(self.vocab)

    def train(
        self,
        corpus: Iterable[str],
        vocab_size: int | None = None,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> None:
        """Train BPE merge rules on a corpus of text strings.

        Args:
            corpus: Iterable of text documents/lines.
            vocab_size: Target vocabulary size (defaults to self.vocab_size).
            min_frequency: Minimum pair frequency required to create a merge.
            verbose: If True, prints progress updates.
        """
        if vocab_size is not None:
            if vocab_size < BASE_VOCAB_SIZE:
                raise ValueError(f"vocab_size must be >= {BASE_VOCAB_SIZE}")
            self.vocab_size = vocab_size

        self._init_base_vocab()
        self.merges.clear()

        num_merges = self.vocab_size - len(self.vocab)
        if num_merges <= 0:
            return

        # Step 1: Pre-tokenize corpus and count word frequencies
        word_counts: Counter[str] = Counter()
        for doc in corpus:
            matches = SPLIT_PATTERN.findall(doc)
            word_counts.update(matches)

        # Step 2: Convert words to tuple of byte token IDs
        # word_dict: tuple[int, ...] -> frequency
        word_dict: dict[tuple[int, ...], int] = {}
        for word, count in word_counts.items():
            raw_bytes = word.encode("utf-8")
            token_tuple = tuple(_byte_to_token_id(b) for b in raw_bytes)
            word_dict[token_tuple] = count

        # Step 3: Iteratively find and merge the most frequent pair
        for step in range(num_merges):
            pair_counts: Counter[tuple[int, int]] = Counter()

            for tokens, freq in word_dict.items():
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            best_pair, highest_freq = pair_counts.most_common(1)[0]
            if highest_freq < min_frequency:
                break

            new_token_id = len(self.vocab)
            self.merges[best_pair] = new_token_id

            # Concatenate bytes of the merged pair to build vocab
            self.vocab[new_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Update word_dict with the merged token
            new_word_dict: dict[tuple[int, ...], int] = {}
            for tokens, freq in word_dict.items():
                new_tokens = self._merge_pair_in_tuple(tokens, best_pair, new_token_id)
                new_word_dict[new_tokens] = freq
            word_dict = new_word_dict

            if verbose and (step + 1) % 500 == 0:
                print(f"BPE merge {step + 1}/{num_merges}: {best_pair} -> {new_token_id} (freq={highest_freq})")

    @staticmethod
    def _merge_pair_in_tuple(
        tokens: tuple[int, ...], pair: tuple[int, int], new_id: int
    ) -> tuple[int, ...]:
        """Replace all occurrences of pair with new_id in a tuple of token IDs."""
        result: list[int] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return tuple(result)

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Encode a pre-tokenized text chunk into token IDs using learned merges."""
        raw_bytes = chunk.encode("utf-8")
        tokens = [_byte_to_token_id(b) for b in raw_bytes]

        if len(tokens) < 2:
            return tokens

        while len(tokens) >= 2:
            # Find all adjacent pairs and their merge priority
            pair_candidates = [
                ((tokens[i], tokens[i + 1]), self.merges[(tokens[i], tokens[i + 1])])
                for i in range(len(tokens) - 1)
                if (tokens[i], tokens[i + 1]) in self.merges
            ]

            if not pair_candidates:
                break

            # Merge the pair that was learned first (lowest new_token_id)
            best_pair, new_id = min(pair_candidates, key=lambda x: x[1])

            new_tokens: list[int] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode a text string into a list of integer token IDs.

        Args:
            text: Input string.
            add_bos: Whether to prepend the BOS token ID.
            add_eos: Whether to append the EOS token ID.

        Returns:
            List of integer token IDs.
        """
        if not text:
            tokens: list[int] = []
        else:
            chunks = SPLIT_PATTERN.findall(text)
            tokens = []
            for chunk in chunks:
                tokens.extend(self._encode_chunk(chunk))

        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]

        return tokens

    def decode(
        self,
        tokens: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode a sequence of integer token IDs back into a string.

        Args:
            tokens: Sequence of integer token IDs.
            skip_special_tokens: Whether to omit special tokens (<pad>, <bos>, etc.).

        Returns:
            Decoded text string.
        """
        byte_chunks: list[bytes] = []

        for token_id in tokens:
            if token_id in self.inverse_special_tokens:
                if not skip_special_tokens:
                    special_str = self.inverse_special_tokens[token_id]
                    byte_chunks.append(special_str.encode("utf-8"))
            elif token_id in self.vocab:
                byte_chunks.append(self.vocab[token_id])
            else:
                # Unknown ID fallback
                if not skip_special_tokens:
                    byte_chunks.append(UNK_TOKEN.encode("utf-8"))

        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """Serialize tokenizer vocabulary and merge rules to a JSON file.

        Args:
            path: Target file path (.json).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            # Merges stored as list of [id1, id2, new_id]
            "merges": [
                [pair[0], pair[1], new_id]
                for pair, new_id in self.merges.items()
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load a serialized tokenizer from a JSON file.

        Args:
            path: Path to serialized tokenizer JSON file.

        Returns:
            Configured BPETokenizer instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.special_tokens = data.get("special_tokens", dict(DEFAULT_SPECIAL_TOKENS))
        tokenizer.inverse_special_tokens = {v: k for k, v in tokenizer.special_tokens.items()}
        tokenizer._init_base_vocab()

        for id1, id2, new_id in data.get("merges", []):
            pair = (id1, id2)
            tokenizer.merges[pair] = new_id
            tokenizer.vocab[new_id] = tokenizer.vocab[id1] + tokenizer.vocab[id2]

        return tokenizer
