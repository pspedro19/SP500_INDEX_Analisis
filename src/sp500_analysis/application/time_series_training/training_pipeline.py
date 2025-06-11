from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .data_loading import load_data, split_data
from .hyperparameter_search import search_arima_order
from .model_fitting import fit_model
from .result_export import export_forecast
from .ensemble import average_series, load_forecast, export_ensemble


def run_training_pipeline(
    data_path: str | Path,
    output_dir: str | Path,
    ensemble_inputs: Iterable[str | Path] | None = None,
) -> Path:
    """Run a simple training pipeline and optionally build an ensemble."""
    df = load_data(data_path)
    train, _, test = split_data(df)
    target = df.columns[0]
    order = search_arima_order(train[target])
    _, preds = fit_model(train[target], test[target], order)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_path = export_forecast(preds, output_dir / "forecast.csv")

    if ensemble_inputs:
        series_list = [preds]
        for path in ensemble_inputs:
            series_list.append(load_forecast(path))
        ensemble_series = average_series(series_list)
        export_ensemble(ensemble_series, output_dir / "forecast_ensemble.csv")

    return forecast_path
