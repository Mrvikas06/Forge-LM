"""CLI tool to run formal validation evaluation on ForgeLM checkpoints.

Usage:
    python scripts/evaluate_checkpoint.py --checkpoint checkpoints/best_model.pt --max-batches 25
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgelm.data.dataset import PretrainingDataset
from forgelm.evaluation.evaluator import Evaluator
from forgelm.evaluation.metrics import calculate_accuracy, calculate_bpc, calculate_perplexity, calculate_topk_accuracy
from forgelm.model.config import ModelConfig
from forgelm.model.transformer import ForgeTransformer

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"
DEFAULT_VAL_DATA = PROJECT_ROOT / "data" / "tokenized" / "val.bin"


def evaluate(checkpoint_path: Path, val_data_path: Path, max_batches: int, batch_size: int, device_str: str) -> None:
    """Run full validation evaluation on a model checkpoint."""
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() and device_str in ("auto", "cuda") else "cpu")
    print("=" * 65)
    print("FORGELM MODEL EVALUATION HARNESS")
    print("=" * 65)
    print(f"Checkpoint File   : {checkpoint_path}")
    print(f"Validation Data   : {val_data_path}")
    print(f"Execution Device  : {device}")
    print(f"Max Eval Batches  : {max_batches}")
    print("-" * 65)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_config" in ckpt:
        model_config = ModelConfig(**ckpt["model_config"])
    else:
        model_config = ModelConfig()

    model = ForgeTransformer(model_config)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if not val_data_path.exists():
        print(f"Warning: Validation data not found at {val_data_path}. Generating dummy evaluation batches.")
        val_dataset = torch.utils.data.TensorDataset(
            torch.randint(0, model_config.vocab_size, (100, model_config.max_seq_len)),
            torch.randint(0, model_config.vocab_size, (100, model_config.max_seq_len)),
        )
    else:
        val_dataset = PretrainingDataset(val_data_path, seq_len=model_config.max_seq_len)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_top1_acc = 0.0
    total_top5_acc = 0.0
    num_batches = 0

    t0 = time.perf_counter()
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break

            x = x.to(device)
            y = y.to(device)

            logits, loss = model(x, targets=y)

            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                total_top1_acc += calculate_accuracy(logits, y)
                total_top5_acc += calculate_topk_accuracy(logits, y, k=5)
                num_batches += 1

    duration = time.perf_counter() - t0

    if num_batches == 0:
        print("No valid evaluation batches processed.")
        return

    avg_loss = total_loss / num_batches
    ppl = calculate_perplexity(avg_loss)
    bpc = calculate_bpc(avg_loss)
    top1 = (total_top1_acc / num_batches) * 100
    top5 = (total_top5_acc / num_batches) * 100

    print("EVALUATION METRICS RESULTS:")
    print("-" * 65)
    print(f"Validation Loss (nats) : {avg_loss:.4f}")
    print(f"Perplexity (PPL)       : {ppl:.2f}")
    print(f"Bits-Per-Character     : {bpc:.4f} bits/token")
    print(f"Top-1 Token Accuracy   : {top1:.2f}%")
    print(f"Top-5 Token Accuracy   : {top5:.2f}%")
    print(f"Batches Evaluated      : {num_batches} batches")
    print(f"Evaluation Duration    : {duration:.2f} seconds ({num_batches/duration:.1f} batch/s)")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ForgeLM model checkpoint on validation data.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to checkpoint .pt")
    parser.add_argument("--val-data", type=str, default=str(DEFAULT_VAL_DATA), help="Path to val.bin tokenized file")
    parser.add_argument("--max-batches", type=int, default=20, help="Maximum validation batches to evaluate")
    parser.add_argument("--batch-size", type=int, default=4, help="Validation micro-batch size")
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('auto', 'cpu', 'cuda')")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=Path(args.checkpoint),
        val_data_path=Path(args.val_data),
        max_batches=args.max_batches,
        batch_size=args.batch_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
