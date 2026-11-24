"""CLI script to launch the ForgeLM FastAPI inference server.

Usage:
    python scripts/serve.py
    python scripts/serve.py --port 8080 --host 127.0.0.1
    python scripts/serve.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import sys
from pathlib import Path

import uvicorn

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgelm.backend.service import model_service
from forgelm.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Launch ForgeLM inference server CLI."""
    parser = argparse.ArgumentParser(description="Serve ForgeLM inference API via FastAPI.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Network interface to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional path to tokenizer JSON file",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Optional path to model config YAML",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device ('auto', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    args = parser.parse_args()

    # Preload model with specific arguments before server start
    logger.info("Pre-loading model into ModelService...")
    model_service.load(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        model_config_path=args.model_config,
        device_str=args.device,
    )

    logger.info(f"Starting server on http://{args.host}:{args.port}")
    uvicorn.run(
        "forgelm.backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
