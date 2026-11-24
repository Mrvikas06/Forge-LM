"""Numerical parity tests verifying KV Cache outputs match standard forward pass."""

import pytest
import torch

from forgelm.inference.kv_cache import KVCache
from forgelm.model.config import ModelConfig
from forgelm.model.transformer import ForgeTransformer


@pytest.fixture
def test_model() -> ForgeTransformer:
    config = ModelConfig(
        vocab_size=128,
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=128,
        max_seq_len=64,
        dropout=0.0,
        bias=False,
    )
    model = ForgeTransformer(config)
    model.eval()
    return model


def test_cache_numerical_parity(test_model: ForgeTransformer):
    """Verify incremental decoding with KVCache produces logits identical to full un-cached pass.

    This ensures that KV Cache is a lossless mathematical equivalence optimization.
    """
    b = 1
    prompt_len = 4
    gen_len = 5
    total_len = prompt_len + gen_len

    # Fixed deterministic sequence of tokens
    full_sequence = torch.randint(0, test_model.config.vocab_size, (b, total_len))

    # --- Method 1: Standard un-cached full sequence pass ---
    with torch.no_grad():
        uncached_logits, _ = test_model(full_sequence)  # (b, total_len, vocab_size)

    # --- Method 2: KV Cache incremental pass ---
    cache = KVCache()

    # Step 2a: Prefill prompt tokens
    prompt_tokens = full_sequence[:, :prompt_len]
    with torch.no_grad():
        cached_prefill_logits, _ = test_model(prompt_tokens, kv_cache=cache, start_pos=0)

    # Verify prefill logits match standard pass exactly
    torch.testing.assert_close(
        cached_prefill_logits,
        uncached_logits[:, :prompt_len, :],
        rtol=1e-5,
        atol=1e-5,
    )

    # Step 2b: Incrementally generate one token at a time with KV Cache
    for step in range(gen_len):
        current_pos = prompt_len + step
        next_input = full_sequence[:, current_pos : current_pos + 1]

        with torch.no_grad():
            cached_step_logits, _ = test_model(next_input, kv_cache=cache, start_pos=current_pos)

        # Compare single-step cached logits with un-cached logits at exact position
        target_logits = uncached_logits[:, current_pos : current_pos + 1, :]
        torch.testing.assert_close(
            cached_step_logits,
            target_logits,
            rtol=1e-5,
            atol=1e-5,
            msg=f"Discrepancy at generation step {step} (pos {current_pos})",
        )
