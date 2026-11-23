"""Train Byte-Pair Encoding (BPE) tokenizer on processed text data.

Usage:
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --vocab-size 4096 --max-lines 50000
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgelm.tokenizer.bpe import BPETokenizer
from forgelm.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "train.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "tokenizer" / "tokenizer.json"


def load_corpus(path: Path, max_lines: int | None = None) -> list[str]:
    """Load text lines from a file up to max_lines.

    Args:
        path: Path to text file.
        max_lines: Maximum number of lines to read.

    Returns:
        List of text lines.
    """
    if not path.exists():
        raise FileNotFoundError(f"Training corpus not found: {path}")

    lines: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
            if max_lines and i + 1 >= max_lines:
                break

    return lines


def main() -> None:
    """Run tokenizer training."""
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on text corpus.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to input text file (default: data/processed/train.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to save trained tokenizer JSON (default: tokenizer/tokenizer.json)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
        help="Target vocabulary size (default: 4096)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=50000,
        help="Maximum lines to read for training (default: 50000)",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="Minimum pair frequency to form a merge (default: 2)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    logger.info("Initializing BPE Tokenizer training")
    logger.info(f"Corpus path      : {input_path}")
    logger.info(f"Target vocab size: {args.vocab_size:,}")
    logger.info(f"Max lines        : {args.max_lines:,}")
    logger.info(f"Output path      : {output_path}")

    # Step 1: Read corpus
    t0 = time.time()
    corpus = load_corpus(input_path, max_lines=args.max_lines)
    logger.info(f"Loaded {len(corpus):,} lines ({time.time() - t0:.2f}s)")

    if not corpus:
        logger.error("Corpus is empty. Please run scripts/prepare_data.py first.")
        sys.exit(1)

    # Step 2: Train BPE tokenizer
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    logger.info("Training merge rules...")
    t_train = time.time()
    tokenizer.train(corpus, vocab_size=args.vocab_size, min_frequency=args.min_freq, verbose=True)
    train_duration = time.time() - t_train

    logger.info(f"Trained {len(tokenizer.merges):,} merges in {train_duration:.2f}s")
    logger.info(f"Final vocabulary size: {len(tokenizer):,}")

    # Step 3: Save tokenizer
    tokenizer.save(output_path)
    file_size_kb = output_path.stat().st_size / 1024
    logger.info(f"Saved tokenizer to {output_path} ({file_size_kb:.1f} KB)")

    # Step 4: Verification test
    sample_text = "Once upon a time, Lily found a little blue bird in the forest."
    tokens = tokenizer.encode(sample_text, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)

    logger.info("\n--- Verification Sample ---")
    logger.info(f"Sample text   : {sample_text}")
    logger.info(f"Encoded tokens: {tokens}")
    logger.info(f"Token count   : {len(tokens)} (raw bytes: {len(sample_text.encode('utf-8'))})")
    logger.info(f"Decoded text  : {decoded}")

    assert decoded == sample_text, "Decoded text does not match original sample!"
    logger.info("Verification passed successfully.")


if __name__ == "__main__":
    main()
