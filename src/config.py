"""Runtime configuration shared by training and evaluation code."""

from __future__ import annotations

import torch


def _mps_is_available() -> bool:
    """Return whether this PyTorch build can use Apple's MPS backend."""
    return bool(torch.backends.mps.is_available())


def select_device() -> torch.device:
    """Select the fastest available PyTorch device.

    The capability-based order also works under WSL, where the operating
    system is reported as Linux even though the host machine runs Windows.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = select_device()
