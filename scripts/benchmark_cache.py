"""Benchmark comparing inference throughput with and without Key-Value (KV) Cache.

Usage:
    python scripts/benchmark_cache.py
    python scripts/benchmark_cache.py --tokens 100
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgelm.inference.generator import TextGenerator
from forgelm.model.config import ModelConfig
from forgelm.model.transformer import ForgeTransformer
from forgelm.tokenizer.bpe import BPETokenizer

DEFAULT_TOKENIZER = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model" / "model_config.yaml"


def benchmark_generation(
    generator: TextGenerator,
    prompt: str,
    max_new_tokens: int,
    use_cache: bool,
) -> tuple[float, float]:
    """Measure duration and tokens-per-second for a generation run.

    Returns:
        Tuple of (duration_sec, tokens_per_sec).
    """
    # Warmup pass
    _ = generator.generate(prompt, max_new_tokens=5, use_cache=use_cache)

    t0 = time.time()
    _ = generator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,  # greedy for deterministic timing
        use_cache=use_cache,
    )
    duration = time.time() - t0
    tok_per_sec = max_new_tokens / duration if duration > 0 else 0.0
    return duration, tok_per_sec


def main() -> None:
    """Run comparative benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark KV Cache vs Uncached Generation.")
    parser.add_argument(
        "--tokens",
        type=int,
        default=50,
        help="Number of tokens to generate per benchmark run (default: 50)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time in a small village, a curious girl named Lily",
        help="Input text prompt",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print(" ForgeLM KV Cache Optimization Benchmark")
    print(f" Device: {device} | Generating {args.tokens} new tokens")
    print("=" * 60)

    # Load model and tokenizer
    config = ModelConfig.from_yaml(DEFAULT_CONFIG)
    model = ForgeTransformer(config).to(device).eval()

    tokenizer = BPETokenizer.load(DEFAULT_TOKENIZER)
    generator = TextGenerator(model=model, tokenizer=tokenizer, device=device)

    print("\n1. Running WITHOUT KV Cache (O(N^2) Full Sequence)...")
    time_uncached, rate_uncached = benchmark_generation(
        generator, args.prompt, args.tokens, use_cache=False
    )
    print(f"   Duration   : {time_uncached:.3f} s")
    print(f"   Throughput : {rate_uncached:.1f} tokens/s")

    print("\n2. Running WITH KV Cache (O(N) Incremental Decoding)...")
    time_cached, rate_cached = benchmark_generation(
        generator, args.prompt, args.tokens, use_cache=True
    )
    print(f"   Duration   : {time_cached:.3f} s")
    print(f"   Throughput : {rate_cached:.1f} tokens/s")

    speedup = time_uncached / time_cached if time_cached > 0 else 1.0

    print("\n" + "-" * 60)
    print(f" Results Summary:")
    print(f" - Uncached Throughput : {rate_uncached:.1f} tok/s")
    print(f" - Cached Throughput   : {rate_cached:.1f} tok/s")
    print(f" - Speedup Factor      : {speedup:.2f}x faster with KV Cache")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
