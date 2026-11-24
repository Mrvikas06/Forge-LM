"""CLI launcher for the ForgeLM Full-Stack Web Application.

Boots the FastAPI backend and optionally opens the interactive web UI
in the user's default browser.

Usage:
    python scripts/web.py
    python scripts/web.py --port 8080 --no-browser
    python scripts/web.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgelm.backend.service import model_service
from forgelm.utils.logging import get_logger

logger = get_logger(__name__)


def open_browser(url: str, delay: float = 1.0) -> None:
    """Open browser in a background timer thread once server is ready."""
    time.sleep(delay)
    webbrowser.open(url)


def main() -> None:
    """Launch full-stack web UI application."""
    parser = argparse.ArgumentParser(description="Launch ForgeLM Web Application.")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)",
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
        help="Optional path to tokenizer JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on ('auto', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically launch web browser",
    )
    args = parser.parse_args()

    # Pre-load model weights
    model_service.load(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device_str=args.device,
    )

    url = f"http://{args.host}:{args.port}/app"
    logger.info(f"ForgeLM Web UI available at: {url}")

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(url, 1.2), daemon=True).start()

    uvicorn.run("forgelm.backend.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
