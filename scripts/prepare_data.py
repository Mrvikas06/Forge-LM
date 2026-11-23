"""Data processing pipeline for ForgeLM.

Reads raw TinyStories CSV files, cleans the text, and writes processed
plain-text files ready for tokenizer training.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --max-samples 100000
"""

import argparse
import csv
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgelm.data.clean import clean_text
from forgelm.utils.logging import get_logger


def process_split(
    input_path: Path,
    output_path: Path,
    max_samples: int | None = None,
) -> dict:
    """Process a single CSV split into a cleaned text file.

    Each story is written as one line in the output file.

    Args:
        input_path: Path to raw CSV file.
        output_path: Path to write processed text file.
        max_samples: Maximum number of samples to process (None = all).

    Returns:
        Dictionary with processing statistics.
    """
    csv.field_size_limit(sys.maxsize)

    stats = {"total": 0, "kept": 0, "skipped": 0}

    with (
        open(input_path, "r", encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        reader = csv.DictReader(fin)

        for row in tqdm(reader, desc=f"Processing {input_path.name}"):
            stats["total"] += 1

            cleaned = clean_text(row.get("text", ""))

            if cleaned:
                fout.write(cleaned + "\n")
                stats["kept"] += 1
            else:
                stats["skipped"] += 1

            if max_samples and stats["total"] >= max_samples:
                break

    return stats


def main() -> None:
    """Run the data processing pipeline."""
    parser = argparse.ArgumentParser(description="Process TinyStories dataset.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of samples per split (default: process all).",
    )
    args = parser.parse_args()

    logger.info("Starting data processing pipeline")
    logger.info(f"Raw data directory : {RAW_DIR}")
    logger.info(f"Output directory   : {PROCESSED_DIR}")

    if args.max_samples:
        logger.info(f"Max samples/split  : {args.max_samples:,}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train.csv", "train.txt"),
        ("validation.csv", "val.txt"),
    ]

    start_time = time.time()

    for csv_name, txt_name in splits:
        input_path = RAW_DIR / csv_name
        output_path = PROCESSED_DIR / txt_name

        if not input_path.exists():
            logger.warning(f"Skipping {csv_name}: file not found")
            continue

        logger.info(f"Processing {csv_name} -> {txt_name}")
        stats = process_split(input_path, output_path, args.max_samples)

        logger.info(
            f"  {csv_name}: {stats['kept']:,} kept, "
            f"{stats['skipped']:,} skipped, "
            f"{stats['total']:,} total"
        )

    elapsed = time.time() - start_time
    logger.info(f"Data processing complete in {elapsed:.1f}s")

    # Print output file sizes
    for _, txt_name in splits:
        out = PROCESSED_DIR / txt_name
        if out.exists():
            size_mb = out.stat().st_size / 1024**2
            logger.info(f"  {txt_name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
