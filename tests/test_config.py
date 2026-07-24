import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None
else:
    from src.config import select_device


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DeviceSelectionTests(unittest.TestCase):
    def test_cuda_has_highest_priority(self):
        with (
            patch("src.config.torch.cuda.is_available", return_value=True),
            patch("src.config._mps_is_available", return_value=True),
        ):
            self.assertEqual(select_device().type, "cuda")

    def test_mps_is_used_when_cuda_is_unavailable(self):
        with (
            patch("src.config.torch.cuda.is_available", return_value=False),
            patch("src.config._mps_is_available", return_value=True),
        ):
            self.assertEqual(select_device().type, "mps")

    def test_cpu_is_the_fallback(self):
        with (
            patch("src.config.torch.cuda.is_available", return_value=False),
            patch("src.config._mps_is_available", return_value=False),
        ):
            self.assertEqual(select_device().type, "cpu")


if __name__ == "__main__":
    unittest.main()
