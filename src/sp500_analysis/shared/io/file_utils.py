import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_dataframe(path: str | Path) -> pd.DataFrame:
    """Read a DataFrame from CSV or Excel depending on file extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to CSV or Excel based on the path extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def read_json(path: str | Path) -> Any:
    """Load JSON data from a file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(data: Any, path: str | Path) -> None:
    """Write JSON data to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
