"""CLI script to tokenize processed text files into binary token datasets.

Usage:
    python scripts/tokenize_dataset.py
    python scripts/tokenize_dataset.py --max-lines 50000
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgelm.data.dataset import tokenize_text_file
from forgelm.tokenizer.bpe import BPETokenizer
from forgelm.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "tokenized"


def main() -> None:
    """Run dataset tokenization."""
    parser = argparse.ArgumentParser(description="Tokenize processed text into binary arrays.")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=str(DEFAULT_TOKENIZER_PATH),
        help="Path to trained tokenizer JSON file",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(DEFAULT_PROCESSED_DIR),
        help="Directory containing train.txt and val.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save train.bin and val.bin",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Optional maximum lines to tokenize per split (for debugging)",
    )
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)

    if not tokenizer_path.exists():
        logger.error(f"Tokenizer not found at {tokenizer_path}. Please run train_tokenizer.py first.")
        sys.exit(1)

    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = BPETokenizer.load(tokenizer_path)
    logger.info(f"Loaded tokenizer with vocabulary size: {len(tokenizer):,}")

    splits = [
        ("train.txt", "train.bin"),
        ("val.txt", "val.bin"),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    start_total = time.time()

    for txt_name, bin_name in splits:
        txt_path = processed_dir / txt_name
        bin_path = output_dir / bin_name

        if not txt_path.exists():
            logger.warning(f"Skipping {txt_name}: file does not exist at {txt_path}")
            continue

        logger.info(f"Tokenizing {txt_name} -> {bin_name}...")
        t0 = time.time()
        num_tokens = tokenize_text_file(
            text_path=txt_path,
            tokenizer=tokenizer,
            output_bin_path=bin_path,
            max_lines=args.max_lines,
        )
        elapsed = time.time() - t0
        rate = num_tokens / elapsed if elapsed > 0 else 0
        logger.info(f"Completed {bin_name}: {num_tokens:,} tokens in {elapsed:.2f}s ({rate:,.0f} tok/s)")

    logger.info(f"All splits tokenized in {time.time() - start_total:.2f}s")


if __name__ == "__main__":
    main()
