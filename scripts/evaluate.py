"""CLI evaluation script for ForgeLM checkpoints.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
    python scripts/evaluate.py --checkpoint checkpoints/checkpoint_step_1000.pt --max-batches 100
"""

import argparse
import sys
from pathlib import Path

import torch

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgelm.data.dataset import PretrainingDataset
from forgelm.evaluation.evaluator import Evaluator
from forgelm.model.config import ModelConfig
from forgelm.model.transformer import ForgeTransformer
from forgelm.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DATA_BIN = PROJECT_ROOT / "data" / "tokenized" / "val.bin"


def main() -> None:
    """Run checkpoint evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate ForgeLM checkpoint on validation dataset.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to saved model checkpoint (.pt)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Optional path to model_config.yaml (defaults to embedded checkpoint config)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_BIN),
        help="Path to tokenized binary dataset (.bin)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size (default: 16)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional limit on number of batches to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device ('auto', 'cuda', 'cpu')",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at: {checkpoint_path}")
        sys.exit(1)

    if not data_path.exists():
        logger.error(f"Evaluation dataset not found at: {data_path}")
        sys.exit(1)

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 1. Load checkpoint
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 2. Resolve model configuration
    if args.model_config:
        model_config = ModelConfig.from_yaml(args.model_config)
    elif "model_config" in checkpoint:
        model_config = ModelConfig(**checkpoint["model_config"])
    else:
        logger.warning("No model_config found in checkpoint. Falling back to default configs/model/model_config.yaml")
        model_config = ModelConfig.from_yaml(PROJECT_ROOT / "configs" / "model" / "model_config.yaml")

    # 3. Instantiate model and load weights
    model = ForgeTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    step = checkpoint.get("step", "N/A")
    total_params = model.get_num_params()
    logger.info(f"Loaded model (Step {step}) | Parameters: {total_params:,}")

    # 4. Load dataset
    dataset = PretrainingDataset(data_path, seq_len=model_config.max_seq_len)
    logger.info(f"Loaded dataset from {data_path} ({len(dataset):,} samples)")

    # 5. Run evaluation
    evaluator = Evaluator(model=model, device=device)
    logger.info("Running evaluation...")
    metrics = evaluator.evaluate(
        data=dataset,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        ignore_index=model_config.pad_token_id,
    )

    # 6. Format report
    print("\n" + "=" * 50)
    print(" ForgeLM Evaluation Results")
    print("=" * 50)
    print(f" Checkpoint       : {checkpoint_path.name}")
    print(f" Training Step    : {step}")
    print(f" Model Parameters : {total_params:,}")
    print(f" Evaluated Tokens : {metrics['total_tokens']:,}")
    print(f" Evaluation Time  : {metrics['duration_sec']:.2f}s ({metrics['tok_per_sec']:,.0f} tok/s)")
    print("-" * 50)
    print(f" Validation Loss  : {metrics['loss']:.4f}")
    print(f" Perplexity (PPL) : {metrics['perplexity']:.2f}")
    print(f" Top-1 Accuracy   : {metrics['top1_accuracy'] * 100:.2f}%")
    print(f" Top-5 Accuracy   : {metrics['top5_accuracy'] * 100:.2f}%")
    print(f" Bits-per-token   : {metrics['bpc']:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
