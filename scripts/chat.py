"""Interactive chat CLI for ForgeLM instruction-tuned models.

Provides a terminal chat loop formatting user queries into standardized
chat templates (<|user|> ... <|assistant|>) with streaming generation.

Usage:
    python scripts/chat.py
    python scripts/chat.py --checkpoint checkpoints/finetuned/finetuned_model.pt
    python scripts/chat.py --temperature 0.7 --top-k 40
"""

import argparse
import sys
from pathlib import Path

import torch

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgelm.data.instruction import ASSISTANT_TAG, USER_TAG, format_instruction
from forgelm.inference.generator import TextGenerator
from forgelm.model.config import ModelConfig
from forgelm.model.transformer import ForgeTransformer
from forgelm.tokenizer.bpe import BPETokenizer
from forgelm.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_FINETUNED_CKPT = PROJECT_ROOT / "checkpoints" / "finetuned" / "finetuned_model.pt"
DEFAULT_BASE_CKPT = PROJECT_ROOT / "checkpoints" / "best_model.pt"
DEFAULT_TOKENIZER = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model" / "model_config.yaml"


def main() -> None:
    """Run interactive chat interface."""
    parser = argparse.ArgumentParser(description="Chat interactively with fine-tuned ForgeLM.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt). Defaults to finetuned, then base model.",
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
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum response tokens to generate (default: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k filtering threshold (default: 40)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling threshold (default: 0.9)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.15,
        help="Repetition penalty parameter (default: 1.15)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on ('auto', 'cuda', 'cpu')",
    )
    args = parser.parse_args()

    # 1. Resolve checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif DEFAULT_FINETUNED_CKPT.exists():
        checkpoint_path = DEFAULT_FINETUNED_CKPT
    elif DEFAULT_BASE_CKPT.exists():
        checkpoint_path = DEFAULT_BASE_CKPT
    else:
        checkpoint_path = DEFAULT_FINETUNED_CKPT

    tokenizer_path = Path(args.tokenizer)

    if not checkpoint_path.exists():
        logger.error(
            f"Checkpoint not found at: {checkpoint_path}. Train or fine-tune a model first!"
        )
        sys.exit(1)

    if not tokenizer_path.exists():
        logger.error(f"Tokenizer not found at: {tokenizer_path}")
        sys.exit(1)

    # 2. Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 3. Load tokenizer
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = BPETokenizer.load(tokenizer_path)

    # 4. Load model
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

    logger.info(f"ForgeLM initialized on {device} ({model.get_num_params():,} parameters)")

    generator = TextGenerator(model=model, tokenizer=tokenizer, device=device)

    # Banner
    print("\n" + "=" * 60)
    print(" ForgeLM Interactive Chat Interface")
    print(f" Checkpoint: {checkpoint_path.name}")
    print(f" Sampling: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, rep={args.repetition_penalty}")
    print(" Commands: 'exit' or 'quit' to end session, '/clear' to reset")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                break

            if user_input.lower() == "/clear":
                print("\n[Conversation reset]\n")
                continue

            # Format using chat template
            formatted_prompt = format_instruction(user_input)

            print("ForgeLM > ", end="", flush=True)

            response_tokens = []
            for token_text in generator.stream_generate(
                prompt=formatted_prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            ):
                # Stop if user token is generated
                if USER_TAG in token_text:
                    break
                response_tokens.append(token_text)
                print(token_text, end="", flush=True)

            print("\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession ended.")
            break


if __name__ == "__main__":
    main()
