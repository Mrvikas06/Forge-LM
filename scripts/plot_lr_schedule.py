"""CLI utility to calculate, verify, and plot ForgeLM learning rate schedules.

Usage:
    python scripts/plot_lr_schedule.py --max-steps 1000 --warmup-steps 100 --peak-lr 3e-4
    python scripts/plot_lr_schedule.py --export-plot docs/lr_schedule.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgelm.training.optim import get_cosine_schedule_with_warmup


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and plot ForgeLM learning rate schedule.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Linear warmup steps")
    parser.add_argument("--peak-lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--export-plot", type=str, default=None, help="Optional image path to save plot")
    args = parser.parse_args()

    steps = list(range(0, args.max_steps + 1))
    lrs = [
        get_cosine_schedule_with_warmup(
            current_step=s,
            total_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            peak_lr=args.peak_lr,
            min_lr=args.min_lr,
        )
        for s in steps
    ]

    print("=" * 60)
    print("FORGELM LEARNING RATE SCHEDULE PROFILE")
    print("=" * 60)
    print(f"Total Steps       : {args.max_steps:,}")
    print(f"Warmup Steps      : {args.warmup_steps:,} ({(args.warmup_steps/args.max_steps)*100:.1f}%)")
    print(f"Peak Learning Rate: {args.peak_lr:.2e}")
    print(f"Min Learning Rate : {args.min_lr:.2e}")
    print("-" * 60)
    print(f"{'Step':<10} | {'Phase':<15} | {'Learning Rate':<15} | {'% of Peak':<10}")
    print("-" * 60)

    sample_checkpoints = [
        0,
        args.warmup_steps // 2,
        args.warmup_steps,
        args.warmup_steps + (args.max_steps - args.warmup_steps) // 4,
        args.warmup_steps + (args.max_steps - args.warmup_steps) // 2,
        args.warmup_steps + 3 * (args.max_steps - args.warmup_steps) // 4,
        args.max_steps,
    ]

    for s in sample_checkpoints:
        phase = "Warmup" if s < args.warmup_steps else ("Peak" if s == args.warmup_steps else "Cosine Decay")
        lr_val = lrs[s]
        pct = (lr_val / args.peak_lr) * 100
        print(f"{s:<10} | {phase:<15} | {lr_val:<15.6e} | {pct:<9.1f}%")

    print("=" * 60)

    if args.export_plot:
        try:
            import matplotlib.pyplot as plt

            out_path = Path(args.export_plot)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(9, 4.5), dpi=150)
            plt.style.use("dark_background")
            plt.plot(steps, lrs, color="#38bdf8", linewidth=2.2, label="Learning Rate")
            plt.axvline(x=args.warmup_steps, color="#f59e0b", linestyle="--", alpha=0.8, label=f"Warmup End ({args.warmup_steps})")
            plt.title("ForgeLM Cosine Learning Rate Schedule with Warmup", fontsize=13, fontweight="bold", pad=12)
            plt.xlabel("Training Step", fontsize=11)
            plt.ylabel("Learning Rate", fontsize=11)
            plt.grid(True, linestyle=":", alpha=0.3)
            plt.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            print(f"Saved learning rate schedule plot to: {out_path}")
        except ImportError:
            print("Matplotlib not installed. Skipping plot export.")


if __name__ == "__main__":
    main()
