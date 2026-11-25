"""Unit tests for ForgeLM experiment tracking and visualization."""

import json
from pathlib import Path

import pytest

from forgelm.experiments.tracker import ExperimentTracker
from forgelm.experiments.visualizer import plot_benchmark_bars, plot_loss_curves


def test_tracker_initialization_and_logging(tmp_path: Path):
    """Test experiment tracker creates metadata, logs metrics, and outputs summary."""
    tracker = ExperimentTracker(
        experiment_name="test_run",
        base_dir=tmp_path,
        config={"vocab_size": 4096, "batch_size": 8},
        use_tensorboard=False,
    )

    assert tracker.run_dir.exists()
    assert tracker.config_file.exists()

    # Log step metrics
    tracker.log_step(step=1, train_loss=3.85, lr=1e-4, tok_per_sec=1500.0, vram_mb=500.0)
    tracker.log_step(step=2, train_loss=3.42, lr=2e-4, tok_per_sec=1600.0, vram_mb=500.0)

    # Log eval metrics
    tracker.log_eval(step=2, val_loss=3.30, perplexity=27.11)

    # Verify metrics.jsonl
    assert tracker.metrics_file.exists()
    with open(tracker.metrics_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 3
    assert lines[0]["step"] == 1
    assert lines[0]["train_loss"] == 3.85
    assert lines[2]["type"] == "eval"
    assert lines[2]["val_loss"] == 3.30

    # Finish and verify summary
    summary = tracker.finish()
    assert tracker.summary_file.exists()
    assert summary["total_steps"] == 2
    assert summary["total_evals"] == 1
    assert summary["best_val_loss"] == 3.30
    assert summary["best_perplexity"] == 27.11


def test_visualizer_plot_loss_curves(tmp_path: Path):
    """Test plot_loss_curves correctly parses jsonl and saves figure."""
    metrics_file = tmp_path / "metrics.jsonl"
    with open(metrics_file, "w", encoding="utf-8") as f:
        for step in range(1, 21):
            loss = 4.0 - (step * 0.1)
            lr = 1e-4 * step
            f.write(json.dumps({"type": "step", "step": step, "train_loss": loss, "lr": lr}) + "\n")
            if step % 5 == 0:
                f.write(json.dumps({"type": "eval", "step": step, "val_loss": loss + 0.05, "perplexity": 20.0}) + "\n")

    out_plot = tmp_path / "loss_curves.png"
    result = plot_loss_curves(metrics_file, output_path=out_plot)

    assert result.exists()
    assert result.stat().st_size > 1000


def test_visualizer_plot_benchmark_bars(tmp_path: Path):
    """Test plot_benchmark_bars creates comparison bar chart."""
    benchmark_data = [
        {"label": "Without Cache", "tokens_per_sec": 14.5, "latency_ms": 68.9},
        {"label": "With KV Cache", "tokens_per_sec": 42.1, "latency_ms": 23.7},
    ]

    out_plot = tmp_path / "benchmark.png"
    result = plot_benchmark_bars(benchmark_data, output_path=out_plot)

    assert result.exists()
    assert result.stat().st_size > 1000
