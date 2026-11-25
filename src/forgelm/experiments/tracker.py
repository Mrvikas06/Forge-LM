"""Experiment tracker for recording training metrics, hyperparameters, and telemetry."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from forgelm.utils.logging import get_logger

logger = get_logger(__name__)

# Try optional TensorBoard SummaryWriter
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TENSORBOARD = True
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]
    _HAS_TENSORBOARD = False


class ExperimentTracker:
    """Records training loss, validation metrics, learning rates, and hardware telemetry.

    Persists metrics to structured JSONL for programmatic analysis, generates summaries,
    and forwards to TensorBoard if the library is available.
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        base_dir: str | Path = "experiments/runs",
        config: dict[str, Any] | None = None,
        use_tensorboard: bool = True,
    ) -> None:
        """Initialize the experiment tracker.

        Args:
            experiment_name: Optional name for this experiment run.
            base_dir: Root directory for storing experiment logs.
            config: Optional configuration dictionary (model/training hyperparams).
            use_tensorboard: Whether to enable TensorBoard logging if available.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            self.run_name = f"{experiment_name}_{timestamp}"
        else:
            self.run_name = f"run_{timestamp}"

        self.run_dir = Path(base_dir) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.summary_file = self.run_dir / "summary.json"
        self.config_file = self.run_dir / "config.json"

        self.start_time = time.time()
        self.step_metrics: list[dict[str, Any]] = []
        self.eval_metrics: list[dict[str, Any]] = []

        # TensorBoard writer setup
        self.tb_writer = None
        if use_tensorboard and _HAS_TENSORBOARD and SummaryWriter is not None:
            try:
                self.tb_writer = SummaryWriter(log_dir=str(self.run_dir / "tb_logs"))
                logger.info(f"TensorBoard logging enabled at: {self.run_dir / 'tb_logs'}")
            except Exception as e:
                logger.warning(f"Could not initialize TensorBoard writer: {e}")

        # Save run metadata and configuration
        self.metadata = {
            "run_name": self.run_name,
            "created_at": datetime.now().isoformat(),
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "config": config or {},
        }
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"ExperimentTracker initialized: {self.run_dir}")

    def log_step(
        self,
        step: int,
        train_loss: float,
        lr: float,
        tok_per_sec: float | None = None,
        vram_mb: float | None = None,
    ) -> None:
        """Log training step metrics.

        Args:
            step: Training iteration step number.
            train_loss: Micro/accumulated cross-entropy loss.
            lr: Current optimizer learning rate.
            tok_per_sec: Processing throughput in tokens per second.
            vram_mb: Current CUDA VRAM allocation in MB.
        """
        record: dict[str, Any] = {
            "type": "step",
            "step": step,
            "train_loss": round(float(train_loss), 5),
            "lr": float(lr),
            "elapsed_seconds": round(time.time() - self.start_time, 2),
        }
        if tok_per_sec is not None:
            record["tok_per_sec"] = round(float(tok_per_sec), 1)
        if vram_mb is not None:
            record["vram_mb"] = round(float(vram_mb), 2)

        self._append_record(record)
        self.step_metrics.append(record)

        # TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_scalar("Train/Loss", train_loss, step)
            self.tb_writer.add_scalar("Train/LearningRate", lr, step)
            if tok_per_sec is not None:
                self.tb_writer.add_scalar("System/TokensPerSec", tok_per_sec, step)
            if vram_mb is not None:
                self.tb_writer.add_scalar("System/VRAM_MB", vram_mb, step)

    def log_eval(
        self,
        step: int,
        val_loss: float,
        perplexity: float,
    ) -> None:
        """Log validation evaluation metrics.

        Args:
            step: Training iteration step.
            val_loss: Validation cross-entropy loss.
            perplexity: Computed validation perplexity.
        """
        record: dict[str, Any] = {
            "type": "eval",
            "step": step,
            "val_loss": round(float(val_loss), 5),
            "perplexity": round(float(perplexity), 3),
            "elapsed_seconds": round(time.time() - self.start_time, 2),
        }

        self._append_record(record)
        self.eval_metrics.append(record)

        # TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_scalar("Val/Loss", val_loss, step)
            self.tb_writer.add_scalar("Val/Perplexity", perplexity, step)

    def _append_record(self, record: dict[str, Any]) -> None:
        """Append record as a single JSON line to metrics file."""
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def finish(self, extra_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        """Finalize the experiment run, close writers, and write summary report.

        Args:
            extra_summary: Optional additional metrics to include in summary.

        Returns:
            Dictionary containing the run summary.
        """
        duration = time.time() - self.start_time

        best_val_loss = (
            min(m["val_loss"] for m in self.eval_metrics)
            if self.eval_metrics
            else None
        )
        best_ppl = (
            min(m["perplexity"] for m in self.eval_metrics)
            if self.eval_metrics
            else None
        )

        final_train_loss = (
            self.step_metrics[-1]["train_loss"]
            if self.step_metrics
            else None
        )

        summary: dict[str, Any] = {
            "run_name": self.run_name,
            "total_steps": len(self.step_metrics),
            "total_evals": len(self.eval_metrics),
            "duration_seconds": round(duration, 2),
            "final_train_loss": final_train_loss,
            "best_val_loss": best_val_loss,
            "best_perplexity": best_ppl,
            "completed_at": datetime.now().isoformat(),
        }

        if extra_summary:
            summary.update(extra_summary)

        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if self.tb_writer:
            self.tb_writer.close()

        logger.info(
            f"Experiment run '{self.run_name}' finalized in {duration:.2f}s | "
            f"Best Val Loss: {best_val_loss} | Best PPL: {best_ppl}"
        )
        return summary
