from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def average_series(series_list: Iterable[pd.Series]) -> pd.Series:
    """Return the element-wise mean of a list of Series."""
    series_list = list(series_list)
    if not series_list:
        raise ValueError("no series provided")
    df = pd.concat(series_list, axis=1)
    avg = df.mean(axis=1)
    avg.name = "ensemble"
    return avg


def load_forecast(path: str | Path) -> pd.Series:
    """Load a forecast CSV produced by ``result_export.export_forecast``."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date")["forecast"]


def export_ensemble(series: pd.Series, output_file: str | Path) -> Path:
    from .result_export import export_forecast

    return export_forecast(series, output_file)
