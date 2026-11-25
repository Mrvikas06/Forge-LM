"""Pretraining engine and training loop orchestration for ForgeLM."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from forgelm.model.transformer import ForgeTransformer
from forgelm.training.config import TrainingConfig
from forgelm.training.optim import configure_optimizers, get_cosine_schedule_with_warmup, get_lr
from forgelm.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """Orchestrates pretraining with AMP, gradient accumulation, and checkpointing.

    Designed for memory efficiency on consumer hardware (e.g. RTX 3050 6GB VRAM).
    """

    def __init__(
        self,
        model: ForgeTransformer,
        config: TrainingConfig,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
        tracker: Any | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tracker = tracker

        # Resolve execution device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.model.to(self.device)
        logger.info(f"Model placed on device: {self.device}")

        # Setup optimizer and scheduler
        if optimizer is None:
            self.optimizer = configure_optimizers(
                model=self.model,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optimizer

        if scheduler is None:
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                warmup_steps=config.warmup_steps,
                max_steps=config.max_steps,
                min_lr_ratio=config.min_lr_ratio,
            )
        else:
            self.scheduler = scheduler

        # Setup AMP (Automatic Mixed Precision)
        self.use_amp = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        logger.info(f"Automatic Mixed Precision (AMP): {self.use_amp}")

        # Setup DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.val_loader = None
        if self.val_dataset is not None and len(self.val_dataset) > 0:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=False,
            )

        # Training state trackers
        self.step = 0
        self.best_val_loss = float("inf")
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self) -> dict[str, float]:
        """Run evaluation on validation split.

        Returns:
            Dictionary containing 'val_loss' and 'perplexity'.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_loader):
                if i >= self.config.eval_steps:
                    break

                x = x.to(self.device)
                y = y.to(self.device)

                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    _, loss = self.model(x, targets=y)

                if loss is not None and not torch.isnan(loss):
                    total_loss += loss.item()
                    num_batches += 1

        self.model.train()

        if num_batches == 0:
            return {}

        avg_loss = total_loss / num_batches
        perplexity = math.exp(min(avg_loss, 20.0))  # Cap exponent to avoid overflow

        return {"val_loss": avg_loss, "perplexity": perplexity}

    def save_checkpoint(self, path: str | Path, is_best: bool = False) -> None:
        """Save training checkpoint to disk.

        Args:
            path: Target file path.
            is_best: Whether this is currently the best checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "model_config": self.model.config.to_dict(),
            "training_config": self.config.to_dict(),
        }

        torch.save(state, path)
        logger.info(f"Checkpoint saved to {path} (step {self.step})")

        if is_best:
            best_path = path.parent / "best_model.pt"
            torch.save(state, best_path)
            logger.info(f"Best model updated at {best_path}")

    def load_checkpoint(self, path: str | Path) -> int:
        """Load checkpoint state from disk.

        Args:
            path: Path to checkpoint file.

        Returns:
            Step number restored from checkpoint.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint and self.use_amp:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.step = checkpoint.get("step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        logger.info(f"Restored checkpoint from {path} at step {self.step}")
        return self.step

    def train_step(self, x: torch.Tensor, y: torch.Tensor, is_accumulating: bool) -> float:
        """Execute a single forward-backward pass micro-step.

        Args:
            x: Input token IDs tensor.
            y: Target token IDs tensor.
            is_accumulating: Whether this micro-step is accumulating gradients.

        Returns:
            Scalar loss value for this micro-step.
        """
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            _, loss = self.model(x, targets=y)
            # Scale loss by accumulation steps so total gradient magnitude matches effective batch size
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if not is_accumulating:
            # Step optimizer once gradient accumulation is complete
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        return loss.item() * self.config.gradient_accumulation_steps

    def train(self) -> None:
        """Run the complete pretraining loop."""
        logger.info("=" * 60)
        logger.info("Starting ForgeLM Pretraining")
        logger.info(f"Target steps       : {self.config.max_steps:,}")
        logger.info(f"Micro batch size   : {self.config.batch_size}")
        logger.info(f"Grad accum steps   : {self.config.gradient_accumulation_steps}")
        logger.info(f"Effective batch    : {self.config.effective_batch_size}")
        logger.info(f"Peak learning rate : {self.config.learning_rate}")
        logger.info(f"Device             : {self.device}")
        logger.info("=" * 60)

        self.model.train()
        train_iter = iter(self.train_loader)
        running_loss = 0.0
        start_time = time.time()
        tokens_processed = 0

        while self.step < self.config.max_steps:
            accum_loss = 0.0

            # Gradient accumulation loop
            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)

                is_accumulating = micro_step < (self.config.gradient_accumulation_steps - 1)
                loss_val = self.train_step(x, y, is_accumulating=is_accumulating)
                accum_loss += loss_val
                tokens_processed += x.numel()

            step_loss = accum_loss / self.config.gradient_accumulation_steps
            running_loss += step_loss
            self.step += 1

            # Log periodic progress
            elapsed = time.time() - start_time
            tok_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
            current_lr = get_lr(self.optimizer)

            if self.tracker is not None:
                vram_val = (
                    torch.cuda.memory_allocated(0) / (1024 * 1024)
                    if self.device.type == "cuda"
                    else None
                )
                self.tracker.log_step(
                    step=self.step,
                    train_loss=step_loss,
                    lr=current_lr,
                    tok_per_sec=tok_per_sec,
                    vram_mb=vram_val,
                )

            if self.step % 10 == 0 or self.step == 1:
                logger.info(
                    f"Step {self.step:>6d}/{self.config.max_steps} | "
                    f"Loss: {step_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"{tok_per_sec:,.0f} tok/s"
                )

            # Evaluation cycle
            if self.step % self.config.eval_interval == 0 and self.val_loader is not None:
                eval_metrics = self.evaluate()
                val_loss = eval_metrics.get("val_loss", float("inf"))
                val_ppl = eval_metrics.get("perplexity", float("inf"))
                logger.info(
                    f"--- Eval at step {self.step}: Val Loss = {val_loss:.4f}, Perplexity = {val_ppl:.2f} ---"
                )

                if self.tracker is not None:
                    self.tracker.log_eval(
                        step=self.step,
                        val_loss=val_loss,
                        perplexity=val_ppl,
                    )

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        self.checkpoint_dir / f"checkpoint_step_{self.step}.pt",
                        is_best=True,
                    )

            # Checkpoint save cycle
            if self.step % self.config.checkpoint_interval == 0:
                self.save_checkpoint(
                    self.checkpoint_dir / f"checkpoint_step_{self.step}.pt",
                    is_best=False,
                )

        logger.info(f"Training completed in {time.time() - start_time:.2f}s")
        if self.tracker is not None:
            self.tracker.finish({"best_val_loss": self.best_val_loss})

        self.save_checkpoint(self.checkpoint_dir / "final_model.pt", is_best=False)
