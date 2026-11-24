"""Supervised Fine-Tuning (SFT) CLI script for ForgeLM.

Adapts a pretrained ForgeLM checkpoint on instruction-response pairs
using prompt-masked cross-entropy loss.

Usage:
    python scripts/finetune.py --base-checkpoint checkpoints/best_model.pt
    python scripts/finetune.py --instruction-data data/instructions.json --max-steps 500
"""

import argparse
import sys
from pathlib import Path

import torch

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgelm.data.instruction import InstructionDataset
from forgelm.model.config import ModelConfig
from forgelm.model.transformer import ForgeTransformer
from forgelm.tokenizer.bpe import BPETokenizer
from forgelm.training.config import TrainingConfig
from forgelm.training.trainer import Trainer
from forgelm.utils.logging import get_logger
from forgelm.utils.seed import set_seed

logger = get_logger(__name__)

DEFAULT_BASE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"
DEFAULT_FINETUNE_CONFIG = PROJECT_ROOT / "configs" / "training" / "finetune_config.yaml"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "model" / "model_config.yaml"
DEFAULT_TOKENIZER = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
DEFAULT_INSTRUCTION_DATA = PROJECT_ROOT / "data" / "instructions.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "finetuned"


def main() -> None:
    """Run supervised instruction fine-tuning."""
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for ForgeLM.")
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default=str(DEFAULT_BASE_CHECKPOINT),
        help="Path to pretrained base model checkpoint (.pt)",
    )
    parser.add_argument(
        "--instruction-data",
        type=str,
        default=str(DEFAULT_INSTRUCTION_DATA),
        help="Path to JSON file containing instruction-response pairs",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Optional path to validation instruction JSON file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=str(DEFAULT_TOKENIZER),
        help="Path to trained tokenizer JSON file",
    )
    parser.add_argument(
        "--finetune-config",
        type=str,
        default=str(DEFAULT_FINETUNE_CONFIG),
        help="Path to fine-tuning YAML configuration",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=str(DEFAULT_MODEL_CONFIG),
        help="Path to model architecture YAML configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save fine-tuned checkpoints",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override total fine-tuning steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override micro-batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override fine-tuning learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override execution device ('cuda', 'cpu', or 'auto')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. Validate files
    base_ckpt_path = Path(args.base_checkpoint)
    tokenizer_path = Path(args.tokenizer)
    instruction_path = Path(args.instruction_data)

    if not tokenizer_path.exists():
        logger.error(f"Tokenizer not found at {tokenizer_path}. Please train tokenizer first.")
        sys.exit(1)

    if not instruction_path.exists():
        logger.error(f"Instruction dataset not found at {instruction_path}.")
        sys.exit(1)

    # 2. Load configurations
    finetune_config = TrainingConfig.from_yaml(args.finetune_config)
    if args.max_steps is not None:
        finetune_config.max_steps = args.max_steps
    if args.batch_size is not None:
        finetune_config.batch_size = args.batch_size
    if args.learning_rate is not None:
        finetune_config.learning_rate = args.learning_rate
    if args.device is not None:
        finetune_config.device = args.device

    finetune_config.checkpoint_dir = str(args.output_dir)

    # 3. Load tokenizer & dataset
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = BPETokenizer.load(tokenizer_path)

    logger.info(f"Loading instruction dataset from: {instruction_path}")
    train_dataset = InstructionDataset(
        data=instruction_path,
        tokenizer=tokenizer,
        max_seq_len=512,
        pad_token_id=tokenizer.pad_id,
    )
    logger.info(f"Training dataset size: {len(train_dataset)} examples")

    val_dataset = None
    if args.val_data and Path(args.val_data).exists():
        logger.info(f"Loading validation dataset from: {args.val_data}")
        val_dataset = InstructionDataset(
            data=args.val_data,
            tokenizer=tokenizer,
            max_seq_len=512,
            pad_token_id=tokenizer.pad_id,
        )

    # 4. Initialize model & load base weights
    if base_ckpt_path.exists():
        logger.info(f"Loading base checkpoint: {base_ckpt_path}")
        device_str = (
            "cuda"
            if torch.cuda.is_available() and finetune_config.device in ("auto", "cuda")
            else "cpu"
        )
        checkpoint = torch.load(base_ckpt_path, map_location=device_str)

        if "model_config" in checkpoint:
            model_config = ModelConfig(**checkpoint["model_config"])
        else:
            model_config = ModelConfig.from_yaml(args.model_config)

        model = ForgeTransformer(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Successfully loaded pretrained weights into model.")
    else:
        logger.warning(
            f"Base checkpoint not found at {base_ckpt_path}. Initializing from scratch for demonstration."
        )
        model_config = ModelConfig.from_yaml(args.model_config)
        model = ForgeTransformer(model_config)

    # 5. Initialize trainer and run SFT
    logger.info("Starting Supervised Fine-Tuning...")
    trainer = Trainer(
        model=model,
        config=finetune_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.train()

    # Save final fine-tuned model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "finetuned_model.pt"
    torch.save(
        {
            "step": trainer.step,
            "model_state_dict": model.state_dict(),
            "model_config": model_config.__dict__,
            "finetune_config": finetune_config.__dict__,
        },
        final_path,
    )
    logger.info(f"Fine-tuning complete! Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
