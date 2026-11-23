"""Tests for YAML configuration loading."""

from pathlib import Path

import pytest

from forgelm.config.loader import load_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_load_model_config():
    """Model config file should load without errors."""
    config = load_config(CONFIGS_DIR / "model" / "model_config.yaml")
    assert isinstance(config, dict)


def test_load_training_config():
    """Training config file should load without errors."""
    config = load_config(CONFIGS_DIR / "training" / "training_config.yaml")
    assert isinstance(config, dict)


def test_load_data_config():
    """Data config file should load without errors."""
    config = load_config(CONFIGS_DIR / "data" / "data_config.yaml")
    assert isinstance(config, dict)


def test_load_missing_config():
    """Loading a nonexistent config should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_file.yaml")
