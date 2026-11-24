"""CLI text generation script for ForgeLM.

Supports both single-shot generation and interactive streaming chat loops.

Usage:
    python scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "Once upon a time"
    python scripts/generate.py --checkpoint checkpoints/best_model.pt --temperature 0.7
"""

import argparse
import sys
from pathlib import Path

import torch

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgelm.inference.generator import TextGenerator
from forgelm.model.config import ModelConfig
from forgelm.model.transformer import ForgeTransformer
from forgelm.tokenizer.bpe import BPETokenizer
from forgelm.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"
DEFAULT_TOKENIZER = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model" / "model_config.yaml"


def main() -> None:
    """Run interactive or one-shot text generation."""
    parser = argparse.ArgumentParser(description="Generate text completions with ForgeLM.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=str(DEFAULT_TOKENIZER),
        help="Path to trained tokenizer JSON file",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Optional path to model_config.yaml",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation. If omitted, starts interactive mode.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum new tokens to generate (default: 150)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8, 0.0 = greedy)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k filtering parameter (default: 50)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling parameter (default: 0.9)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (default: 1.1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device ('auto', 'cuda', 'cpu')",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at: {checkpoint_path}")
        sys.exit(1)

    if not tokenizer_path.exists():
        logger.error(f"Tokenizer not found at: {tokenizer_path}")
        sys.exit(1)

    # 1. Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 2. Load tokenizer
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = BPETokenizer.load(tokenizer_path)

    # 3. Load model checkpoint
    logger.info(f"Loading model checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if args.model_config:
        model_config = ModelConfig.from_yaml(args.model_config)
    elif "model_config" in checkpoint:
        model_config = ModelConfig(**checkpoint["model_config"])
    else:
        model_config = ModelConfig.from_yaml(DEFAULT_CONFIG)

    model = ForgeTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Model ready on {device} ({model.get_num_params():,} parameters)")

    generator = TextGenerator(model=model, tokenizer=tokenizer, device=device)

    def run_stream(prompt: str) -> None:
        print(f"\n{prompt}", end="", flush=True)
        for chunk in generator.stream_generate(
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ):
            print(chunk, end="", flush=True)
        print("\n")

    # Single-shot mode
    if args.prompt:
        run_stream(args.prompt)
        return

    # Interactive loop mode
    print("\n" + "=" * 55)
    print(" ForgeLM Interactive Generator")
    print(" Enter your prompt below (type 'exit' or 'quit' to stop)")
    print(f" Settings: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, rep_pen={args.repetition_penalty}")
    print("=" * 55 + "\n")

    while True:
        try:
            prompt = input("Prompt > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit"):
                break
            run_stream(prompt)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
