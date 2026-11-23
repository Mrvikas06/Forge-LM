"""Load YAML configuration files."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return its contents as a dictionary.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Dictionary of configuration values. Returns an empty dict for
        files that contain only comments or are empty.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data if data is not None else {}
