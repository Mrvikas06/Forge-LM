"""Tests for training engine, optimizer, and learning rate scheduling."""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from forgelm.model.config import ModelConfig
from forgelm.model.transformer import ForgeTransformer
from forgelm.training.config import TrainingConfig
from forgelm.training.optim import configure_optimizers, get_cosine_schedule_with_warmup, get_lr
from forgelm.training.trainer import Trainer

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "training" / "training_config.yaml"


@pytest.fixture
def tiny_model() -> ForgeTransformer:
    config = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
    )
    return ForgeTransformer(config)


def test_training_config_from_yaml():
    """Verify TrainingConfig loads properly from project YAML config."""
    config = TrainingConfig.from_yaml(CONFIG_PATH)
    assert config.batch_size > 0
    assert config.learning_rate > 0
    assert config.max_steps > 0
    assert config.effective_batch_size == config.batch_size * config.gradient_accumulation_steps


def test_optimizer_parameter_groups(tiny_model: ForgeTransformer):
    """Verify selective weight decay on 2D vs 1D parameters."""
    optimizer = configure_optimizers(tiny_model, weight_decay=0.1)

    decay_group = optimizer.param_groups[0]
    no_decay_group = optimizer.param_groups[1]

    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    # All decay parameters should have dim >= 2
    for p in decay_group["params"]:
        assert p.dim() >= 2

    # All no-decay parameters should have dim < 2
    for p in no_decay_group["params"]:
        assert p.dim() < 2


def test_cosine_schedule_warmup():
    """Verify warmup and cosine decay learning rate progression."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=10,
        max_steps=100,
        min_lr_ratio=0.1,
    )

    # Step 0: warmup initial step
    lr_0 = get_lr(optimizer)
    assert 0.0 < lr_0 < 1.0

    # Step to end of warmup
    for _ in range(9):
        optimizer.step()
        scheduler.step()
    lr_warmup = get_lr(optimizer)
    assert abs(lr_warmup - 1.0) < 1e-4

    # Step to middle of decay
    for _ in range(45):
        optimizer.step()
        scheduler.step()
    lr_mid = get_lr(optimizer)
    assert 0.1 < lr_mid < 1.0

    # Step past max_steps
    for _ in range(50):
        optimizer.step()
        scheduler.step()
    lr_end = get_lr(optimizer)
    assert abs(lr_end - 0.1) < 1e-4


def test_trainer_micro_step_and_eval(tiny_model: ForgeTransformer):
    """Verify single training and evaluation steps with Trainer."""
    # Synthetic dataset
    b, t = 4, 16
    x = torch.randint(0, 64, (8, t))
    y = torch.randint(0, 64, (8, t))
    train_dataset = TensorDataset(x, y)
    val_dataset = TensorDataset(x[:2], y[:2])

    train_config = TrainingConfig(
        batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=2,
        eval_interval=1,
        eval_steps=1,
        device="cpu",
        mixed_precision=False,
    )

    trainer = Trainer(
        model=tiny_model,
        config=train_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # Single train step
    loss = trainer.train_step(x[:2], y[:2], is_accumulating=False)
    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))

    # Evaluate
    metrics = trainer.evaluate()
    assert "val_loss" in metrics
    assert "perplexity" in metrics
    assert metrics["perplexity"] >= 1.0


def test_trainer_checkpoint_cycle(tiny_model: ForgeTransformer):
    """Verify checkpoint save and restore fidelity."""
    x = torch.randint(0, 64, (4, 16))
    y = torch.randint(0, 64, (4, 16))
    dataset = TensorDataset(x, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        train_config = TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=5,
            checkpoint_dir=tmpdir,
            device="cpu",
            mixed_precision=False,
        )

        trainer = Trainer(
            model=tiny_model,
            config=train_config,
            train_dataset=dataset,
        )

        trainer.step = 3
        trainer.best_val_loss = 2.45
        save_path = Path(tmpdir) / "test_ckpt.pt"
        trainer.save_checkpoint(save_path, is_best=True)

        assert save_path.exists()
        assert (Path(tmpdir) / "best_model.pt").exists()

        # Create fresh model & trainer, restore from checkpoint
        fresh_model = ForgeTransformer(tiny_model.config)
        new_trainer = Trainer(
            model=fresh_model,
            config=train_config,
            train_dataset=dataset,
        )
        restored_step = new_trainer.load_checkpoint(save_path)

        assert restored_step == 3
        assert new_trainer.step == 3
        assert abs(new_trainer.best_val_loss - 2.45) < 1e-4
