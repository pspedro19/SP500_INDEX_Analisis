from pathlib import Path
from datetime import datetime
import logging


def configurar_logging(log_file: str = 'app.log', name: str = 'app') -> logging.Logger:
    """Configure and return a logger writing to the given file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(name)


class ContextAdapter(logging.LoggerAdapter):
    """Logger adapter that adds contextual information to log records."""

    def process(self, msg, kwargs):
        if self.extra:
            context = " | ".join(f"{k}={v}" for k, v in self.extra.items())
            return f"{msg} [{context}]", kwargs
        return msg, kwargs


def get_logger(name: str, **context) -> logging.LoggerAdapter:
    """Return a logger adapter with the provided contextual information."""
    logger = logging.getLogger(name)
    return ContextAdapter(logger, context)


def setup_logging(log_dir: str | Path, prefix: str, name: str = __name__) -> logging.Logger:
    """Set up logging to a file inside *log_dir* using *prefix* and return the logger."""
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_file = path / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    configurar_logging(str(log_file), name)
    logger = logging.getLogger(name)
    logger.info("Logging initialized", extra={"file": str(log_file)})
    return logger


def catch_exceptions(func):
    """Decorator that logs exceptions and returns None on failure."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - runtime behavior
            logging.getLogger(func.__module__).error(str(exc), exc_info=True)
            return None

    return wrapper
