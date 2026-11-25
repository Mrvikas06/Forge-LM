"""Main pretraining CLI script for ForgeLM.

Usage:
    python scripts/train.py
    python scripts/train.py --max-steps 5000 --batch-size 16
    python scripts/train.py --resume checkpoints/checkpoint_step_1000.pt
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgelm.data.dataset import PretrainingDataset
from forgelm.experiments.tracker import ExperimentTracker
from forgelm.model.config import ModelConfig
from forgelm.model.transformer import ForgeTransformer
from forgelm.training.config import TrainingConfig
from forgelm.training.trainer import Trainer
from forgelm.utils.logging import get_logger
from forgelm.utils.seed import set_seed

logger = get_logger(__name__)

DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "model" / "model_config.yaml"
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "training" / "training_config.yaml"
DEFAULT_TRAIN_BIN = PROJECT_ROOT / "data" / "tokenized" / "train.bin"
DEFAULT_VAL_BIN = PROJECT_ROOT / "data" / "tokenized" / "val.bin"


def main() -> None:
    """Run model pretraining."""
    parser = argparse.ArgumentParser(description="Pretrain ForgeLM on tokenized binary dataset.")
    parser.add_argument(
        "--model-config",
        type=str,
        default=str(DEFAULT_MODEL_CONFIG),
        help="Path to model architecture YAML config",
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default=str(DEFAULT_TRAIN_CONFIG),
        help="Path to training hyperparameters YAML config",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=str(DEFAULT_TRAIN_BIN),
        help="Path to tokenized train.bin dataset",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=str(DEFAULT_VAL_BIN),
        help="Path to tokenized val.bin dataset",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override total training steps",
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
        help="Override peak learning rate",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Override maximum sequence length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device ('cuda', 'cpu', or 'auto')",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file (.pt) to resume training from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="pretrain",
        help="Name prefix for this experiment run",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable ExperimentTracker metric logging",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. Load configurations
    model_config = ModelConfig.from_yaml(args.model_config)
    train_config = TrainingConfig.from_yaml(args.training_config)

    # Apply command-line overrides
    if args.max_steps is not None:
        train_config.max_steps = args.max_steps
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.grad_accum is not None:
        train_config.gradient_accumulation_steps = args.grad_accum
    if args.learning_rate is not None:
        train_config.learning_rate = args.learning_rate
    if args.seq_len is not None:
        model_config.max_seq_len = args.seq_len
    if args.device is not None:
        train_config.device = args.device

    # 2. Check and load datasets
    train_bin = Path(args.train_data)
    val_bin = Path(args.val_data)

    if not train_bin.exists():
        logger.error(
            f"Train data file not found at {train_bin}. "
            "Please run 'python scripts/tokenize_dataset.py' to generate binary tokens."
        )
        sys.exit(1)

    logger.info(f"Loading training dataset from {train_bin}")
    train_dataset = PretrainingDataset(train_bin, seq_len=model_config.max_seq_len)
    logger.info(f"Training dataset size: {len(train_dataset):,} samples")

    val_dataset = None
    if val_bin.exists():
        logger.info(f"Loading validation dataset from {val_bin}")
        val_dataset = PretrainingDataset(val_bin, seq_len=model_config.max_seq_len)
        logger.info(f"Validation dataset size: {len(val_dataset):,} samples")

    # 3. Instantiate model
    logger.info("Initializing ForgeTransformer model architecture")
    model = ForgeTransformer(model_config)
    logger.info(f"Total parameters: {model.get_num_params():,}")

    # 4. Setup experiment tracking
    tracker = None
    if not args.no_track:
        tracker = ExperimentTracker(
            experiment_name=args.experiment_name,
            config={
                "model": model_config.to_dict(),
                "training": train_config.to_dict(),
            },
        )

    # 5. Instantiate trainer
    trainer = Trainer(
        model=model,
        config=train_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tracker=tracker,
    )

    # 5. Optionally resume from checkpoint
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 6. Run pretraining loop
    trainer.train()


if __name__ == "__main__":
    main()
