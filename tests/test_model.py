"""Comprehensive tests for ForgeLM Transformer architecture."""

from pathlib import Path

import pytest
import torch

from forgelm.model.attention import CausalSelfAttention
from forgelm.model.block import TransformerBlock
from forgelm.model.config import ModelConfig
from forgelm.model.mlp import FeedForward
from forgelm.model.transformer import ForgeTransformer

CONFIG_FILE = Path(__file__).resolve().parent.parent / "configs" / "model" / "model_config.yaml"


@pytest.fixture
def small_config() -> ModelConfig:
    """Small model configuration for fast unit testing."""
    return ModelConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=32,
        dropout=0.0,
        bias=False,
    )


def test_config_from_yaml():
    """Verify loading ModelConfig directly from project YAML file."""
    config = ModelConfig.from_yaml(CONFIG_FILE)
    assert config.vocab_size == 4096
    assert config.d_model == 384
    assert config.n_heads == 6
    assert config.n_layers == 6
    assert config.d_ff == 1536
    assert config.d_head == 64


def test_config_validation():
    """Invalid d_model/n_heads combinations must raise ValueError."""
    with pytest.raises(ValueError):
        ModelConfig(d_model=65, n_heads=4)


def test_causal_self_attention_shape(small_config: ModelConfig):
    """Attention output tensor must match input tensor shape."""
    attn = CausalSelfAttention(small_config)
    x = torch.randn(2, 10, small_config.d_model)
    out = attn(x)
    assert out.shape == x.shape


def test_causal_masking_strictness(small_config: ModelConfig):
    """Tokens must not attend to future tokens (autoregressive property).

    Altering token at index 5 must NOT change the output representation at index 0.
    """
    attn = CausalSelfAttention(small_config)
    attn.eval()

    x1 = torch.randn(1, 8, small_config.d_model)
    x2 = x1.clone()
    # Change future token at index 7
    x2[0, 7, :] = x2[0, 7, :] + 10.0

    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)

    # Output at index 0 must be identical because it cannot see position 7
    torch.testing.assert_close(out1[0, 0, :], out2[0, 0, :], rtol=1e-5, atol=1e-5)


def test_feed_forward_shape(small_config: ModelConfig):
    """FeedForward output tensor must match input tensor shape."""
    mlp = FeedForward(small_config)
    x = torch.randn(2, 8, small_config.d_model)
    out = mlp(x)
    assert out.shape == x.shape


def test_transformer_block_shape(small_config: ModelConfig):
    """TransformerBlock output tensor must match input tensor shape."""
    block = TransformerBlock(small_config)
    x = torch.randn(2, 8, small_config.d_model)
    out = block(x)
    assert out.shape == x.shape


def test_transformer_lm_forward(small_config: ModelConfig):
    """Test full forward pass with and without target loss calculation."""
    model = ForgeTransformer(small_config)
    b, t = 2, 8
    idx = torch.randint(0, small_config.vocab_size, (b, t))

    # Forward without targets
    logits, loss = model(idx)
    assert logits.shape == (b, t, small_config.vocab_size)
    assert loss is None

    # Forward with targets
    targets = torch.randint(0, small_config.vocab_size, (b, t))
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (b, t, small_config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0  # Scalar loss
    assert not torch.isnan(loss)


def test_weight_tying(small_config: ModelConfig):
    """Token embedding and language modeling head must share weights."""
    model = ForgeTransformer(small_config)
    assert model.transformer["wte"].weight is model.lm_head.weight


def test_parameter_count():
    """Verify parameter count calculation for full model configuration."""
    config = ModelConfig.from_yaml(CONFIG_FILE)
    model = ForgeTransformer(config)
    total_params = model.get_num_params()

    # ~12.5M parameters (fits comfortably in 6GB VRAM)
    assert 10_000_000 < total_params < 15_000_000


def test_max_seq_len_limit(small_config: ModelConfig):
    """Sequences exceeding max_seq_len must raise ValueError."""
    model = ForgeTransformer(small_config)
    excessive_seq = torch.randint(0, small_config.vocab_size, (1, small_config.max_seq_len + 5))
    with pytest.raises(ValueError):
        model(excessive_seq)
