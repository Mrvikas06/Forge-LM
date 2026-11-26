"""Extended unit tests for ForgeLM training subsystem and optimizers."""

import torch
import torch.nn as nn
import pytest

from forgelm.model.config import ModelConfig
from forgelm.model.transformer import ForgeTransformer
from forgelm.training.config import TrainingConfig
from forgelm.training.optim import calculate_cosine_lr, configure_optimizers
from forgelm.training.trainer import Trainer


def test_calculate_cosine_lr_boundaries():
    """Verify learning rate calculation at key schedule intervals."""
    total_steps = 1000
    warmup_steps = 100
    peak_lr = 3e-4
    min_lr_ratio = 0.1

    # Step 0 should be warm up step 1 fraction
    lr_0 = calculate_cosine_lr(0, total_steps, warmup_steps, peak_lr, min_lr_ratio)
    assert 0 < lr_0 <= (peak_lr / warmup_steps)

    # At warmup_steps, should hit exact peak
    lr_peak = calculate_cosine_lr(warmup_steps, total_steps, warmup_steps, peak_lr, min_lr_ratio)
    assert pytest.approx(lr_peak, rel=1e-4) == peak_lr

    # At total_steps, should decay to exact min_lr
    lr_end = calculate_cosine_lr(total_steps, total_steps, warmup_steps, peak_lr, min_lr_ratio)
    expected_min = peak_lr * min_lr_ratio
    assert pytest.approx(lr_end, rel=1e-4) == expected_min

    # Beyond total_steps, should stay at min_lr
    lr_post = calculate_cosine_lr(total_steps + 50, total_steps, warmup_steps, peak_lr, min_lr_ratio)
    assert pytest.approx(lr_post, rel=1e-4) == expected_min


def test_optimizer_parameter_groups_separation():
    """Verify 2D tensors get weight decay while 1D tensors get 0.0 weight decay."""
    config = ModelConfig(d_model=64, n_layers=2, n_heads=2, vocab_size=128, max_seq_len=32)
    model = ForgeTransformer(config)

    optimizer = configure_optimizers(model, learning_rate=1e-3, weight_decay=0.05)
    assert len(optimizer.param_groups) == 2

    decay_group = optimizer.param_groups[0]
    nodecay_group = optimizer.param_groups[1]

    assert decay_group["weight_decay"] == 0.05
    assert nodecay_group["weight_decay"] == 0.0

    # Ensure every parameter in decay_group has dim >= 2
    for p in decay_group["params"]:
        assert p.dim() >= 2

    # Ensure every parameter in nodecay_group has dim < 2
    for p in nodecay_group["params"]:
        assert p.dim() < 2


def test_trainer_early_stopping_trigger(tmp_path):
    """Verify Trainer breaks early when validation loss fails to improve."""
    from torch.utils.data import TensorDataset

    config = ModelConfig(d_model=32, n_layers=1, n_heads=1, vocab_size=64, max_seq_len=16)
    model = ForgeTransformer(config)

    t_config = TrainingConfig(
        max_steps=50,
        batch_size=2,
        eval_interval=5,
        checkpoint_interval=50,
        checkpoint_dir=str(tmp_path),
        device="cpu",
    )

    # Dummy datasets
    x = torch.randint(0, 64, (20, 16))
    y = torch.randint(0, 64, (20, 16))
    dataset = TensorDataset(x, y)

    trainer = Trainer(
        model=model,
        config=t_config,
        train_dataset=dataset,
        val_dataset=dataset,
        early_stopping_patience=2,  # Stop after 2 stagnant evals
        min_delta=100.0,  # Impossible threshold to force stagnation
    )

    trainer.train()
    # Trainer should have stopped well before step 50
    assert trainer.step < 50
    assert trainer.patience_counter >= 2
