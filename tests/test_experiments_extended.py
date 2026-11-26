"""Extended tests for ForgeLM experiment tracking, summary parsing, and visualizer."""

import json
from pathlib import Path
import pytest

from forgelm.experiments.tracker import ExperimentTracker
from forgelm.experiments.visualizer import load_metrics_jsonl, plot_multi_run_comparison
from scripts.compare_runs import parse_run


def test_load_metrics_jsonl_corrupt_line_handling(tmp_path: Path):
    """Verify load_metrics_jsonl skips malformed JSON lines gracefully."""
    metrics_file = tmp_path / "metrics.jsonl"
    lines = [
        json.dumps({"type": "step", "step": 1, "train_loss": 5.2, "lr": 1e-4}),
        "INVALID_JSON_CORRUPT_LINE",
        json.dumps({"type": "eval", "step": 1, "val_loss": 4.8, "perplexity": 121.5}),
        "",
        json.dumps({"type": "step", "step": 2, "train_loss": 4.9, "lr": 2e-4}),
    ]
    metrics_file.write_text("\n".join(lines), encoding="utf-8")

    step_records, eval_records = load_metrics_jsonl(metrics_file)
    assert len(step_records) == 2
    assert len(eval_records) == 1
    assert step_records[0]["step"] == 1
    assert step_records[1]["step"] == 2


def test_plot_multi_run_comparison(tmp_path: Path):
    """Verify plot_multi_run_comparison creates valid image file."""
    run1 = tmp_path / "run_1"
    run2 = tmp_path / "run_2"
    run1.mkdir()
    run2.mkdir()

    r1_lines = [
        json.dumps({"type": "step", "step": i, "train_loss": 5.0 - (i * 0.1), "lr": 1e-4})
        for i in range(1, 15)
    ]
    r2_lines = [
        json.dumps({"type": "step", "step": i, "train_loss": 6.0 - (i * 0.2), "lr": 2e-4})
        for i in range(1, 15)
    ]

    (run1 / "metrics.jsonl").write_text("\n".join(r1_lines), encoding="utf-8")
    (run2 / "metrics.jsonl").write_text("\n".join(r2_lines), encoding="utf-8")

    out_img = tmp_path / "comparison.png"
    result = plot_multi_run_comparison([run1, run2], output_path=out_img, labels=["Run A", "Run B"])

    assert result.exists()
    assert result.stat().st_size > 1000  # Valid non-empty PNG


def test_parse_run_metadata(tmp_path: Path):
    """Verify parse_run aggregates step losses and summary properly."""
    run_dir = tmp_path / "finetune_test_run"
    run_dir.mkdir()

    summary_data = {"duration_seconds": 45.2, "best_val_loss": 1.25}
    (run_dir / "summary.json").write_text(json.dumps(summary_data), encoding="utf-8")

    metrics_lines = [
        json.dumps({"type": "step", "step": 1, "train_loss": 4.5, "tok_per_sec": 1200.0}),
        json.dumps({"type": "step", "step": 2, "train_loss": 3.8, "tok_per_sec": 1400.0}),
        json.dumps({"type": "step", "step": 3, "train_loss": 3.1, "tok_per_sec": 1300.0}),
    ]
    (run_dir / "metrics.jsonl").write_text("\n".join(metrics_lines), encoding="utf-8")

    info = parse_run(run_dir)
    assert info is not None
    assert info["name"] == "finetune_test_run"
    assert info["steps"] == 3
    assert info["start_loss"] == 4.5
    assert info["min_loss"] == 3.1
    assert info["final_loss"] == 3.1
    assert pytest.approx(info["avg_tok_s"]) == 1300.0
    assert info["duration_s"] == 45.2
