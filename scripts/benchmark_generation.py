"""Ablation benchmarking CLI for ForgeLM text generation and KV cache performance.

Measures throughput, latency, speedup factors, and output diversity across
different sampling configurations (temperature, top-k, top-p, KV cache).

Usage:
    python scripts/benchmark_generation.py
    python scripts/benchmark_generation.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgelm.experiments.visualizer import plot_benchmark_bars
from forgelm.inference.generator import TextGenerator
from forgelm.model.config import ModelConfig
from forgelm.model.transformer import ForgeTransformer
from forgelm.tokenizer.bpe import BPETokenizer
from forgelm.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"
DEFAULT_TOKENIZER = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model" / "model_config.yaml"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "experiments" / "reports"


def main() -> None:
    """Run generation ablation benchmarks."""
    parser = argparse.ArgumentParser(description="Run ForgeLM generation ablation benchmarks.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to trained checkpoint (.pt)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=str(DEFAULT_TOKENIZER),
        help="Path to trained tokenizer JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_REPORT_DIR),
        help="Directory to save benchmark reports and plots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run benchmarks on ('auto', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Number of tokens to generate per benchmark pass (default: 64)",
    )
    args = parser.parse_args()

    # 1. Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 2. Load tokenizer
    tok_path = Path(args.tokenizer)
    if tok_path.exists():
        tokenizer = BPETokenizer.load(tok_path)
    else:
        logger.warning("Tokenizer not found, creating baseline tokenizer for benchmark.")
        tokenizer = BPETokenizer(vocab_size=4096)

    # 3. Load or initialize model
    ckpt_path = Path(args.checkpoint)
    cfg_path = DEFAULT_CONFIG
    if ckpt_path.exists():
        logger.info(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model_config = ModelConfig(**checkpoint.get("model_config", {})) if "model_config" in checkpoint else ModelConfig.from_yaml(cfg_path)
        model = ForgeTransformer(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.warning("No checkpoint found. Initializing architecture for benchmark.")
        model_config = ModelConfig.from_yaml(cfg_path) if cfg_path.exists() else ModelConfig()
        model = ForgeTransformer(model_config)

    model.to(device)
    model.eval()

    generator = TextGenerator(model=model, tokenizer=tokenizer, device=device)
    prompt = "Once upon a time in a faraway forest,"

    logger.info(f"Starting benchmarks on device: {device}")

    # --- BENCHMARK 1: KV Cache ON vs OFF ---
    logger.info("Running KV Cache comparison benchmark...")
    kv_runs = []

    # Warmup
    _ = generator.generate(prompt=prompt, max_new_tokens=10, use_cache=True)

    for use_cache in [False, True]:
        label = "With KV Cache (O(N))" if use_cache else "Without KV Cache (O(N^2))"
        t0 = time.perf_counter()
        completion = generator.generate(
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            temperature=0.0,
            use_cache=use_cache,
        )
        elapsed = time.perf_counter() - t0
        tok_sec = args.max_tokens / elapsed if elapsed > 0 else 0
        kv_runs.append({
            "label": label,
            "use_cache": use_cache,
            "duration_sec": round(elapsed, 4),
            "tokens_per_sec": round(tok_sec, 2),
            "latency_ms_per_tok": round((elapsed / args.max_tokens) * 1000, 2),
        })

    speedup = kv_runs[1]["tokens_per_sec"] / max(kv_runs[0]["tokens_per_sec"], 0.01)

    # --- BENCHMARK 2: Temperature Ablation ---
    logger.info("Running Temperature ablation...")
    temp_results = []
    for temp in [0.0, 0.4, 0.8, 1.2]:
        t0 = time.perf_counter()
        out = generator.generate(
            prompt=prompt,
            max_new_tokens=32,
            temperature=temp,
            use_cache=True,
        )
        elapsed = time.perf_counter() - t0
        temp_results.append({
            "temperature": temp,
            "duration_sec": round(elapsed, 3),
            "preview": out[:120].replace("\n", " "),
        })

    # --- SAVE REPORTS ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON export
    benchmark_data = {
        "device": str(device),
        "max_tokens": args.max_tokens,
        "kv_cache_comparison": kv_runs,
        "speedup_factor": round(speedup, 2),
        "temperature_ablation": temp_results,
    }

    json_file = output_dir / "generation_benchmark.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    # Plot KV Cache speedup
    plot_file = output_dir / "kv_cache_speedup.png"
    plot_benchmark_bars(kv_runs, plot_file)

    # Markdown report
    md_file = output_dir / "ablation_report.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# ForgeLM Generation & KV Cache Ablation Report\n\n")
        f.write(f"- **Device**: `{device}`\n")
        f.write(f"- **Benchmark Tokens**: `{args.max_tokens}` tokens per trial\n")
        f.write(f"- **KV Cache Speedup**: **{speedup:.2f}x faster**\n\n")

        f.write("## 1. KV Cache Acceleration Performance\n\n")
        f.write("| Configuration | Duration (s) | Throughput (tok/s) | Latency (ms/tok) |\n")
        f.write("|---|---|---|---|\n")
        for r in kv_runs:
            f.write(f"| {r['label']} | {r['duration_sec']}s | **{r['tokens_per_sec']}** | {r['latency_ms_per_tok']} ms |\n")

        f.write("\n## 2. Temperature Sampling Ablation\n\n")
        f.write("| Temperature | Latency (s) | Sample Output Preview |\n")
        f.write("|---|---|---|\n")
        for t in temp_results:
            f.write(f"| `{t['temperature']}` | {t['duration_sec']}s | *\"{t['preview']}...\"* |\n")

        f.write(f"\n*Benchmark visualization saved to `{plot_file.name}`.*\n")

    logger.info(f"Benchmark reports generated successfully in {output_dir}:")
    logger.info(f" - {json_file}")
    logger.info(f" - {md_file}")
    logger.info(f" - {plot_file}")

    print("\n" + "=" * 60)
    print(" ForgeLM Generation Ablation Benchmark")
    print(f" KV Cache Speedup: {speedup:.2f}x")
    for r in kv_runs:
        print(f"   {r['label']:<28}: {r['tokens_per_sec']:>6.1f} tok/s ({r['latency_ms_per_tok']} ms/tok)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
