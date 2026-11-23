"""Device detection and management utilities."""

import torch


def get_device() -> torch.device:
    """Return the best available device (CUDA GPU or CPU).

    Returns:
        torch.device configured for the best available hardware.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> dict:
    """Gather basic device and hardware information.

    Returns:
        Dictionary with device type, name, and memory details.
    """
    info: dict = {"device": "cpu", "cuda_available": torch.cuda.is_available()}

    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_mem / 1024**2
        )

    return info
