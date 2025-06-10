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
