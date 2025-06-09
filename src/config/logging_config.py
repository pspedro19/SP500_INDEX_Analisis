import logging
from pathlib import Path


def setup_logging(log_file: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Configure logging to file and stdout.

    Parameters
    ----------
    log_file: str | Path
        Path to the log file.
    level: int, optional
        Logging level, defaults to ``logging.INFO``.

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()
