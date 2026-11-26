"""CLI tool to inspect, analyze, and test the ForgeLM BPE Tokenizer.

Usage:
    python scripts/inspect_tokenizer.py --text "Hello world, this is ForgeLM!"
    python scripts/inspect_tokenizer.py --vocab-summary
    python scripts/inspect_tokenizer.py --benchmark
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgelm.tokenizer.bpe import BPETokenizer
from forgelm.tokenizer.special_tokens import BASE_VOCAB_SIZE, NUM_BYTE_TOKENS, NUM_SPECIAL_TOKENS

DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer.json"


def inspect_text(tokenizer: BPETokenizer, text: str) -> None:
    """Analyze and display token-by-token breakdown for an input string."""
    encoded_raw = tokenizer.encode(text, add_bos=False, add_eos=False)
    encoded_full = tokenizer.encode(text, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(encoded_raw, skip_special_tokens=True)

    byte_len = len(text.encode("utf-8"))
    tok_len = len(encoded_raw)
    compression = byte_len / tok_len if tok_len > 0 else 1.0

    print("=" * 65)
    print("TOKENIZER SEGMENTATION ANALYSIS")
    print("=" * 65)
    print(f"Raw Input Text : {repr(text)}")
    print(f"UTF-8 Bytes    : {byte_len} bytes")
    print(f"Token Count    : {tok_len} tokens (excluding BOS/EOS)")
    print(f"Compression    : {compression:.2f}x bytes/token")
    print(f"Roundtrip Check: {'PASS (Exact Match)' if decoded == text else 'FAIL (Mismatch)'}")
    print("-" * 65)
    print(f"{'Idx':<5} | {'Token ID':<10} | {'Token Representation':<25} | {'Hex Bytes':<12}")
    print("-" * 65)

    for i, tok_id in enumerate(encoded_full):
        if tok_id == tokenizer.pad_id:
            rep = "<pad>"
            hex_str = "SPECIAL"
        elif tok_id == tokenizer.bos_id:
            rep = "<bos>"
            hex_str = "SPECIAL"
        elif tok_id == tokenizer.eos_id:
            rep = "<eos>"
            hex_str = "SPECIAL"
        elif tok_id == tokenizer.unk_id:
            rep = "<unk>"
            hex_str = "SPECIAL"
        else:
            rep = repr(tokenizer.decode([tok_id], skip_special_tokens=False))
            raw_bytes = tokenizer.vocab.get(tok_id, b"")
            hex_str = raw_bytes.hex() if isinstance(raw_bytes, (bytes, bytearray)) else ""

        print(f"{i:<5} | {tok_id:<10} | {rep:<25} | {hex_str:<12}")

    print("=" * 65)


def show_summary(tokenizer: BPETokenizer) -> None:
    """Display overall vocabulary distribution and merge statistics."""
    num_merges = len(tokenizer.merges)
    total_vocab = len(tokenizer.vocab)

    print("=" * 65)
    print("FORGELM TOKENIZER VOCABULARY SPECIFICATION")
    print("=" * 65)
    print(f"Total Vocabulary Size    : {total_vocab:,} tokens")
    print(f"Reserved Special Tokens  : {NUM_SPECIAL_TOKENS} tokens (IDs 0..3)")
    print(f"Raw Base UTF-8 Bytes     : {NUM_BYTE_TOKENS} tokens (IDs 4..259)")
    print(f"Learned BPE Merge Rules  : {num_merges:,} merges (IDs 260..{total_vocab - 1})")
    print(f"Zero OOV Guarantee       : Yes (Byte-fallback enabled)")
    print("-" * 65)
    print("First 5 BPE Merge Operations:")
    for i, (pair, new_id) in enumerate(list(tokenizer.merges.items())[:5]):
        left_tok = repr(tokenizer.decode([pair[0]], skip_special_tokens=False))
        right_tok = repr(tokenizer.decode([pair[1]], skip_special_tokens=False))
        result_tok = repr(tokenizer.decode([new_id], skip_special_tokens=False))
        print(f"  Merge #{i+1:02d}: ({left_tok} + {right_tok}) -> ID {new_id} ({result_tok})")
    print("=" * 65)


def run_benchmark(tokenizer: BPETokenizer) -> None:
    """Benchmark tokenization speed on standard text."""
    import time
    sample_corpus = (
        "Once upon a time in a sunny meadow, a curious little puppy named Barnaby chased a bright red ball. "
        "The sun shone warmly in the sky and the gentle breeze whispered through the tall green trees. "
    ) * 200

    t0 = time.perf_counter()
    tokens = tokenizer.encode(sample_corpus, add_bos=False, add_eos=False)
    t_enc = time.perf_counter() - t0

    t1 = time.perf_counter()
    _ = tokenizer.decode(tokens, skip_special_tokens=True)
    t_dec = time.perf_counter() - t1

    bytes_total = len(sample_corpus.encode("utf-8"))
    enc_speed_mb = (bytes_total / (1024 * 1024)) / t_enc
    dec_speed_mb = (bytes_total / (1024 * 1024)) / t_dec

    print("=" * 65)
    print("TOKENIZER THROUGHPUT BENCHMARK")
    print("=" * 65)
    print(f"Benchmark Corpus Size : {bytes_total:,} bytes ({len(tokens):,} tokens)")
    print(f"Encoding Throughput   : {enc_speed_mb:.2f} MB/s ({len(tokens)/t_enc:,.0f} tok/s)")
    print(f"Decoding Throughput   : {dec_speed_mb:.2f} MB/s ({len(tokens)/t_dec:,.0f} tok/s)")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and benchmark ForgeLM BPE Tokenizer.")
    parser.add_argument("--tokenizer", type=str, default=str(DEFAULT_TOKENIZER_PATH), help="Path to tokenizer.json")
    parser.add_argument("--text", type=str, default="Once upon a time, a little puppy found a red ball in the forest.", help="Sample text to analyze")
    parser.add_argument("--vocab-summary", action="store_true", help="Display vocabulary distribution statistics")
    parser.add_argument("--benchmark", action="store_true", help="Run tokenization throughput benchmark")
    args = parser.parse_args()

    tok_path = Path(args.tokenizer)
    if not tok_path.exists():
        print(f"Error: Tokenizer file not found at {tok_path}")
        sys.exit(1)

    tokenizer = BPETokenizer.load(tok_path)

    if args.vocab_summary:
        show_summary(tokenizer)
    elif args.benchmark:
        run_benchmark(tokenizer)
    else:
        show_summary(tokenizer)
        print()
        inspect_text(tokenizer, args.text)


if __name__ == "__main__":
    main()
