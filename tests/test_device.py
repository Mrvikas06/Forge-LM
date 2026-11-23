"""Tests for device detection and seed management."""

import torch

from forgelm.utils.device import get_device, get_device_info
from forgelm.utils.seed import set_seed


def test_get_device_returns_device():
    """get_device should return a torch.device object."""
    dev = get_device()
    assert isinstance(dev, torch.device)
    assert dev.type in ("cpu", "cuda")


def test_get_device_info_returns_dict():
    """get_device_info should return a dict with expected keys."""
    info = get_device_info()
    assert isinstance(info, dict)
    assert "device" in info
    assert "cuda_available" in info


def test_set_seed_runs_without_error():
    """set_seed should execute without raising exceptions."""
    set_seed(123)

    # Verify determinism for CPU tensors
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b), "set_seed did not produce deterministic results"
