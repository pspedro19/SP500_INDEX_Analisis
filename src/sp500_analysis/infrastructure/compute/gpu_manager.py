import os
import logging

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - optional dependency
    tf = None


def configure_gpu(use_gpu: bool = True, memory_limit: int = 0) -> bool:
    """Configure TensorFlow GPU usage."""
    if not use_gpu or tf is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logging.info("GPU disabled or TensorFlow not available")
        return False

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if not gpus:
        logging.info("No GPU detected")
        return False

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if memory_limit > 0:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)],
            )
        logging.info("GPU configured")
        return True
    except RuntimeError as exc:  # pragma: no cover - configuration error
        logging.error("Failed to configure GPU: %s", exc)
        return False
