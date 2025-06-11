import logging
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - pandas may be unavailable
    pd = None


def load_csv(path: str, delimiter: str = ";") -> Any:
    """Load a CSV file with logging support."""
    if pd is None:
        raise ImportError("pandas is required for load_csv")
    logging.info("Loading CSV %s", path)
    df = pd.read_csv(path, delimiter=delimiter)
    logging.info("Loaded %s rows and %s columns", len(df), len(df.columns))
    return df


def save_csv(df: Any, path: str, delimiter: str = ";") -> None:
    """Save a DataFrame to CSV."""
    if pd is None:
        raise ImportError("pandas is required for save_csv")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=delimiter, index=False)
    logging.info("Saved CSV to %s", path)
