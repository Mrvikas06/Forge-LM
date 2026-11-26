"""CLI utility to inspect, validate, and display metadata from ForgeLM checkpoints.

Usage:
    python scripts/inspect_checkpoint.py --checkpoint checkpoints/best_model.pt
    python scripts/inspect_checkpoint.py --checkpoint checkpoints/finetuned/finetuned_model.pt --tensors
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"


def inspect_checkpoint(path: Path, show_tensors: bool = False) -> None:
    """Inspect checkpoint structure, configurations, and parameter counts."""
    if not path.exists():
        print(f"Error: Checkpoint file not found: {path}")
        sys.exit(1)

    file_size_mb = path.stat().st_size / (1024 * 1024)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    print("=" * 65)
    print("FORGELM CHECKPOINT METADATA INSPECTION")
    print("=" * 65)
    print(f"File Path        : {path}")
    print(f"File Size        : {file_size_mb:.2f} MB")
    print(f"Training Step    : {ckpt.get('step', 'N/A')}")
    print(f"Val Loss         : {ckpt.get('val_loss', ckpt.get('loss', 'N/A'))}")
    print(f"Top-Level Keys   : {list(ckpt.keys())}")
    print("-" * 65)

    # Architecture Config
    cfg = ckpt.get("model_config", {})
    if isinstance(cfg, dict) and cfg:
        print("Model Architecture Parameters:")
        print(f"  Vocab Size     : {cfg.get('vocab_size', 'N/A'):,}")
        print(f"  Hidden Dim (d) : {cfg.get('d_model', 'N/A')}")
        print(f"  Layers         : {cfg.get('n_layers', 'N/A')}")
        print(f"  Attention Heads: {cfg.get('n_heads', 'N/A')}")
        print(f"  Max Context Len: {cfg.get('max_seq_len', 'N/A')}")
    else:
        print("Model Config     : Not explicitly stored in checkpoint dictionary")

    # State Dict Analysis
    state_dict = ckpt.get("model_state_dict", ckpt if isinstance(ckpt, dict) and "step" not in ckpt else {})
    if state_dict:
        total_params = sum(p.numel() for p in state_dict.values())
        total_tensors = len(state_dict)
        print("-" * 65)
        print(f"Total Tensors    : {total_tensors} weight tensors")
        print(f"Total Parameters : {total_params:,} parameters")

        if show_tensors:
            print("-" * 65)
            print(f"{'Tensor Name':<42} | {'Shape':<16} | {'Params':<10}")
            print("-" * 65)
            for name, tensor in state_dict.items():
                shape_str = str(list(tensor.shape))
                print(f"{name:<42} | {shape_str:<16} | {tensor.numel():<10,}")

    # Optimizer State
    has_optim = "optimizer_state_dict" in ckpt
    print("-" * 65)
    print(f"Optimizer State  : {'Present (Resumable)' if has_optim else 'Omitted (Inference Checkpoint)'}")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ForgeLM model checkpoint (.pt).")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to checkpoint .pt file")
    parser.add_argument("--tensors", action="store_true", help="Print breakdown of individual weight tensors")
    args = parser.parse_args()

    inspect_checkpoint(Path(args.checkpoint), show_tensors=args.tensors)


if __name__ == "__main__":
    main()
