"""Utilities to safely configure GPU usage."""

from __future__ import annotations

import os
import logging


class GPUManager:
    """Simple GPU configuration helper for TensorFlow and PyTorch."""

    def __init__(self, device_id: int | str = 0) -> None:
        self.device_id = str(device_id)
        self.logger = logging.getLogger(__name__)

    def configure_tensorflow(self) -> None:
        """Set TensorFlow GPU memory growth if available."""
        try:
            import tensorflow as tf
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", self.device_id)
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception as exc:  # pragma: no cover - safety
                        self.logger.warning("TensorFlow memory growth failed: %s", exc)
                self.logger.info("TensorFlow configured for %s GPU(s)", len(gpus))
            else:
                self.logger.info("TensorFlow: no GPU detected")
        except Exception as exc:
            self.logger.debug("TensorFlow not available: %s", exc)

    def configure_pytorch(self) -> None:
        """Ensure PyTorch uses the selected GPU if available."""
        try:
            import torch
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", self.device_id)
            if torch.cuda.is_available():
                torch.cuda.set_device(int(self.device_id))
                self.logger.info(
                    "PyTorch using GPU: %s",
                    torch.cuda.get_device_name(torch.cuda.current_device()),
                )
            else:
                self.logger.info("PyTorch: no GPU detected")
        except Exception as exc:
            self.logger.debug("PyTorch not available: %s", exc)

    def configure_all(self) -> None:
        """Configure both TensorFlow and PyTorch."""
        self.configure_tensorflow()
        self.configure_pytorch()
