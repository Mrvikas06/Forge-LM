"""Publication-grade metric plotting and visualization for ForgeLM training experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for server and CLI safety
import matplotlib.pyplot as plt


def load_metrics_jsonl(metrics_path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse a metrics.jsonl file into step metrics and eval metrics lists."""
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    step_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []

    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("type") == "step":
                    step_records.append(record)
                elif record.get("type") == "eval":
                    eval_records.append(record)
            except json.JSONDecodeError:
                continue

    return step_records, eval_records


def plot_loss_curves(
    metrics_path: str | Path,
    output_path: str | Path | None = None,
    window_size: int = 10,
) -> Path:
    """Plot training loss with moving average and validation loss / perplexity.

    Args:
        metrics_path: Path to metrics.jsonl or run directory.
        output_path: Destination path for the saved image.
        window_size: Smoothing window size for training loss.

    Returns:
        Path to the saved figure.
    """
    path = Path(metrics_path)
    if path.is_dir():
        metrics_file = path / "metrics.jsonl"
        out_file = Path(output_path) if output_path else path / "loss_curves.png"
    else:
        metrics_file = path
        out_file = Path(output_path) if output_path else path.parent / "loss_curves.png"

    step_records, eval_records = load_metrics_jsonl(metrics_file)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Train Loss & Val Loss
    if step_records:
        steps = [r["step"] for r in step_records]
        train_losses = [r["train_loss"] for r in step_records]

        # Raw loss in light color
        ax1.plot(steps, train_losses, color="#818cf8", alpha=0.35, label="Train Loss (Raw)")

        # Smoothed moving average
        if len(train_losses) >= window_size:
            smoothed = []
            for i in range(len(train_losses)):
                start = max(0, i - window_size + 1)
                smoothed.append(sum(train_losses[start : i + 1]) / (i - start + 1))
            ax1.plot(steps, smoothed, color="#4f46e5", linewidth=2.0, label=f"Train Loss (SMA-{window_size})")

    if eval_records:
        eval_steps = [r["step"] for r in eval_records]
        val_losses = [r["val_loss"] for r in eval_records]
        ax1.plot(eval_steps, val_losses, color="#ef4444", marker="o", linewidth=2.2, label="Validation Loss")

    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax1.set_title("ForgeLM Training & Validation Loss Dynamics", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Plot 2: Perplexity & Learning Rate
    ax2_twin = ax2.twinx()

    if eval_records:
        eval_steps = [r["step"] for r in eval_records]
        ppl_vals = [r["perplexity"] for r in eval_records]
        ax2.plot(eval_steps, ppl_vals, color="#10b981", marker="s", linewidth=2.0, label="Validation Perplexity")

    if step_records:
        steps = [r["step"] for r in step_records]
        lrs = [r["lr"] for r in step_records]
        ax2_twin.plot(steps, lrs, color="#f59e0b", linestyle=":", linewidth=1.8, label="Learning Rate")

    ax2.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Perplexity", color="#10b981", fontsize=11, fontweight="bold")
    ax2_twin.set_ylabel("Learning Rate", color="#f59e0b", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#10b981")
    ax2_twin.tick_params(axis="y", labelcolor="#f59e0b")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Combine legends
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", frameon=True)

    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200)
    plt.close(fig)

    return out_file


def plot_multi_run_comparison(
    run_paths: list[str | Path],
    output_path: str | Path,
    labels: list[str] | None = None,
    window_size: int = 5,
) -> Path:
    """Plot overlaid smoothed loss curves for multiple experiment runs.

    Args:
        run_paths: List of run directory paths or metrics.jsonl file paths.
        output_path: Destination path for saved image.
        labels: Optional custom legend labels.
        window_size: Moving average smoothing window.

    Returns:
        Path to the saved comparison figure.
    """
    out_file = Path(output_path)
    palette = ["#4f46e5", "#06b6d4", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    for i, r_path in enumerate(run_paths):
        p = Path(r_path)
        m_file = p / "metrics.jsonl" if p.is_dir() else p
        if not m_file.exists():
            continue

        step_records, _ = load_metrics_jsonl(m_file)
        if not step_records:
            continue

        steps = [r["step"] for r in step_records]
        losses = [r["train_loss"] for r in step_records]

        # Moving average
        smoothed = []
        for j in range(len(losses)):
            start = max(0, j - window_size + 1)
            smoothed.append(sum(losses[start : j + 1]) / (j - start + 1))

        label = labels[i] if labels and i < len(labels) else p.name
        color = palette[i % len(palette)]
        ax.plot(steps, smoothed, color=color, linewidth=2.2, label=label)

    ax.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Smoothed Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax.set_title("ForgeLM Multi-Run Convergence Comparison", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200)
    plt.close(fig)

    return out_file



def plot_benchmark_bars(
    benchmark_results: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot throughput and speedup comparison bars across configurations.

    Args:
        benchmark_results: List of dicts with 'label', 'tokens_per_sec', and 'latency_ms'.
        output_path: Destination path for the saved image.

    Returns:
        Path to the saved figure.
    """
    out_file = Path(output_path)
    labels = [r["label"] for r in benchmark_results]
    throughputs = [r["tokens_per_sec"] for r in benchmark_results]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, throughputs, color=["#6366f1", "#10b981", "#3b82f6", "#f59e0b"][: len(labels)])

    ax.set_ylabel("Throughput (Tokens / Second)", fontsize=11, fontweight="bold")
    ax.set_title("ForgeLM Decoding Speed Comparison", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Label heights
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f} tok/s",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200)
    plt.close(fig)

    return out_file


def plot_multi_run_comparison(
    run_paths: list[str | Path],
    output_path: str | Path,
    labels: list[str] | None = None,
    window_size: int = 5,
) -> Path:
    """Plot overlaid smoothed loss curves for multiple experiment runs.

    Args:
        run_paths: List of run directory paths or metrics.jsonl file paths.
        output_path: Destination path for saved image.
        labels: Optional custom legend labels.
        window_size: Moving average smoothing window.

    Returns:
        Path to the saved comparison figure.
    """
    out_file = Path(output_path)
    palette = ["#4f46e5", "#06b6d4", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    for i, r_path in enumerate(run_paths):
        p = Path(r_path)
        m_file = p / "metrics.jsonl" if p.is_dir() else p
        if not m_file.exists():
            continue

        step_records, _ = load_metrics_jsonl(m_file)
        if not step_records:
            continue

        steps = [r["step"] for r in step_records]
        losses = [r["train_loss"] for r in step_records]

        # Moving average
        smoothed = []
        for j in range(len(losses)):
            start = max(0, j - window_size + 1)
            smoothed.append(sum(losses[start : j + 1]) / (j - start + 1))

        label = labels[i] if labels and i < len(labels) else p.name
        color = palette[i % len(palette)]
        ax.plot(steps, smoothed, color=color, linewidth=2.2, label=label)

    ax.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Smoothed Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax.set_title("ForgeLM Multi-Run Convergence Comparison", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200)
    plt.close(fig)

    return out_file

