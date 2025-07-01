import torch
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        nargs="+",
        choices=["cpu", "cuda", "mps"],
        help="Device(s) to run tests on.",
    )


def pytest_configure(config):
    devices = config.getoption("--device")
    if devices is None:
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            devices.append("mps")

    pytest.devices = devices
