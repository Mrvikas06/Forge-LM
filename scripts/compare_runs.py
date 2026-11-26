"""CLI utility to aggregate, compare, and visualize ForgeLM experiment runs.

Usage:
    python scripts/compare_runs.py
    python scripts/compare_runs.py --export-markdown docs/EXPERIMENTS.md
    python scripts/compare_runs.py --plot-run finetune_20261125_212306 --output docs/loss.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgelm.experiments.visualizer import plot_loss_curves

DEFAULT_RUNS_DIR = PROJECT_ROOT / "experiments" / "runs"


def parse_run(run_dir: Path) -> dict[str, any] | None:
    """Extract metrics, config, and performance data from a single run directory."""
    if not run_dir.is_dir():
        return None

    config_file = run_dir / "config.json"
    metrics_file = run_dir / "metrics.jsonl"
    summary_file = run_dir / "summary.json"

    run_info = {
        "name": run_dir.name,
        "type": "pretrain" if "pretrain" in run_dir.name else "finetune",
        "steps": 0,
        "start_loss": None,
        "min_loss": float("inf"),
        "final_loss": None,
        "best_val_loss": None,
        "avg_tok_s": None,
        "duration_s": None,
        "step_losses": [],
    }

    # Load summary if present
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                s_data = json.load(f)
                run_info["duration_s"] = s_data.get("duration_seconds")
                run_info["best_val_loss"] = s_data.get("best_val_loss")
        except Exception:
            pass

    # Read streaming metrics lines
    if metrics_file.exists():
        tok_speeds = []
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    m = json.loads(line)
                    if m.get("type") == "step":
                        loss = m.get("train_loss")
                        step = m.get("step", 0)
                        run_info["steps"] = max(run_info["steps"], step)

                        if loss is not None:
                            run_info["step_losses"].append(loss)
                            if run_info["start_loss"] is None:
                                run_info["start_loss"] = loss
                            run_info["min_loss"] = min(run_info["min_loss"], loss)
                            run_info["final_loss"] = loss

                        if m.get("tok_per_sec"):
                            tok_speeds.append(m["tok_per_sec"])
                    elif m.get("type") == "eval":
                        v_loss = m.get("val_loss")
                        if v_loss is not None:
                            if run_info["best_val_loss"] is None or v_loss < run_info["best_val_loss"]:
                                run_info["best_val_loss"] = v_loss

            if tok_speeds:
                run_info["avg_tok_s"] = sum(tok_speeds) / len(tok_speeds)
        except Exception:
            pass

    if run_info["min_loss"] == float("inf"):
        run_info["min_loss"] = None

    return run_info


def display_runs(runs: list[dict[str, any]]) -> None:
    """Format and print comparison table to console."""
    print("=" * 85)
    print("FORGELM EXPERIMENT RUN REGISTRY & COMPARISON")
    print("=" * 85)
    print(f"{'Run Identifier':<28} | {'Type':<8} | {'Steps':<6} | {'Start':<7} | {'Min Loss':<8} | {'Final':<7} | {'Tok/s':<8}")
    print("-" * 85)

    for r in runs:
        s_loss = f"{r['start_loss']:.3f}" if r["start_loss"] is not None else "N/A"
        m_loss = f"{r['min_loss']:.3f}" if r["min_loss"] is not None else "N/A"
        f_loss = f"{r['final_loss']:.3f}" if r["final_loss"] is not None else "N/A"
        t_speed = f"{r['avg_tok_s']:,.0f}" if r["avg_tok_s"] is not None else "N/A"

        print(f"{r['name']:<28} | {r['type']:<8} | {r['steps']:<6} | {s_loss:<7} | {m_loss:<8} | {f_loss:<7} | {t_speed:<8}")

    print("=" * 85)


def export_markdown(runs: list[dict[str, any]], out_path: Path) -> None:
    """Save markdown report of all experiment runs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ForgeLM Experiment Registry & Benchmarks",
        "",
        "| Run Identifier | Type | Steps | Initial Loss | Minimum Loss | Final Loss | Avg Tok/s |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in runs:
        s_loss = f"{r['start_loss']:.4f}" if r["start_loss"] is not None else "—"
        m_loss = f"{r['min_loss']:.4f}" if r["min_loss"] is not None else "—"
        f_loss = f"{r['final_loss']:.4f}" if r["final_loss"] is not None else "—"
        t_speed = f"{r['avg_tok_s']:,.0f}" if r["avg_tok_s"] is not None else "—"
        lines.append(f"| `{r['name']}` | **{r['type']}** | {r['steps']:,} | {s_loss} | {m_loss} | {f_loss} | {t_speed} |")

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported experiment markdown report to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare and rank ForgeLM experiment runs.")
    parser.add_argument("--runs-dir", type=str, default=str(DEFAULT_RUNS_DIR), help="Path to experiments/runs")
    parser.add_argument("--export-markdown", type=str, default=None, help="Save summary table to markdown file")
    parser.add_argument("--plot-run", type=str, default=None, help="Run directory name to plot loss curve for")
    parser.add_argument("--output", type=str, default="loss_curve.png", help="Output path for plot")
    args = parser.parse_args()

    runs_path = Path(args.runs_dir)
    if not runs_path.exists():
        print(f"No experiments runs found at {runs_path}")
        return

    runs = []
    for item in sorted(runs_path.iterdir()):
        if item.is_dir():
            info = parse_run(item)
            if info:
                runs.append(info)

    display_runs(runs)

    if args.export_markdown:
        export_markdown(runs, Path(args.export_markdown))

    if args.plot_run:
        target_dir = runs_path / args.plot_run
        metrics_file = target_dir / "metrics.jsonl" if target_dir.is_dir() else Path(args.plot_run)
        if not metrics_file.exists():
            print(f"Metrics file not found for run {args.plot_run}")
            return

        out_img = plot_loss_curves(metrics_file, output_path=args.output)
        print(f"Generated loss plot at: {out_img}")


if __name__ == "__main__":
    main()
