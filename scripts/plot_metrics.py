"""CLI tool to plot training and validation loss curves from experiment logs.

Usage:
    python scripts/plot_metrics.py
    python scripts/plot_metrics.py --run-dir experiments/runs/pretrain_20261125_120000
    python scripts/plot_metrics.py --output-path experiments/plots/latest_loss.png
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgelm.experiments.visualizer import plot_loss_curves
from forgelm.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_RUNS_DIR = PROJECT_ROOT / "experiments" / "runs"


def find_latest_run(runs_dir: Path) -> Path | None:
    """Find the most recently modified run directory containing metrics.jsonl."""
    if not runs_dir.exists():
        return None
    candidates = [
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "metrics.jsonl").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    """Plot metrics CLI."""
    parser = argparse.ArgumentParser(description="Generate loss and metric curves from ForgeLM runs.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to specific run directory. Defaults to latest run in experiments/runs.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional destination path for generated plot image.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Moving average smoothing window for training loss (default: 10).",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_path = Path(args.run_dir)
    else:
        run_path = find_latest_run(DEFAULT_RUNS_DIR)

    if not run_path or not run_path.exists():
        logger.error(
            f"No valid run directory found at {run_path or DEFAULT_RUNS_DIR}. "
            "Run training first with 'python scripts/train.py'."
        )
        sys.exit(1)

    metrics_file = run_path / "metrics.jsonl"
    if not metrics_file.exists():
        logger.error(f"Metrics file not found: {metrics_file}")
        sys.exit(1)

    logger.info(f"Generating loss curves for run: {run_path.name}")
    saved_plot = plot_loss_curves(
        metrics_path=metrics_file,
        output_path=args.output_path,
        window_size=args.window_size,
    )
    logger.info(f"Plot saved successfully to: {saved_plot}")

    # Display summary if present
    summary_file = run_path / "summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
            print("\n" + "=" * 50)
            print(f" Run Summary: {summary.get('run_name')}")
            print(f" Total Steps     : {summary.get('total_steps')}")
            print(f" Final Train Loss: {summary.get('final_train_loss')}")
            print(f" Best Val Loss   : {summary.get('best_val_loss')}")
            print(f" Best Perplexity : {summary.get('best_perplexity')}")
            print(f" Duration        : {summary.get('duration_seconds')}s")
            print("=" * 50 + "\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
