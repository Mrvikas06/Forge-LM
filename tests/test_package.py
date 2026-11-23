"""Basic tests to verify the ForgeLM package is importable and functional."""

import importlib


def test_package_import():
    """Verify that the forgelm package can be imported."""
    import forgelm

    assert hasattr(forgelm, "__version__")
    assert isinstance(forgelm.__version__, str)


def test_utils_import():
    """Verify that utility subpackages are importable."""
    from forgelm.utils import device, seed, logging

    assert callable(device.get_device)
    assert callable(device.get_device_info)
    assert callable(seed.set_seed)
    assert callable(logging.get_logger)


def test_config_import():
    """Verify that configuration subpackage is importable."""
    from forgelm.config import loader

    assert callable(loader.load_config)
