from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV file with an optional ``date`` column.

    If a ``date`` column is present it will be parsed and set as the index.
    """
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def split_data(df: pd.DataFrame, val_size: int = 5, test_size: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into train/validation/test subsets."""
    if len(df) < val_size + test_size + 1:
        raise ValueError("not enough data to split")

    train_end = len(df) - (val_size + test_size)
    val_end = len(df) - test_size
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test
