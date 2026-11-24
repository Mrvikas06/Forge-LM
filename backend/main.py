"""Uvicorn server entrypoint for ForgeLM."""

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import uvicorn


def main() -> None:
    """Launch Uvicorn server."""
    uvicorn.run("forgelm.backend.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
