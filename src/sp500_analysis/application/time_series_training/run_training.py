from __future__ import annotations

import argparse
from pathlib import Path

from .data_loading import load_data, split_data
from .hyperparameter_search import search_arima_order
from .model_fitting import fit_model
from .result_export import export_forecast


def run(data_path: str, output_dir: str) -> None:
    df = load_data(data_path)
    train, _, test = split_data(df)
    target = df.columns[0]
    order = search_arima_order(train[target])
    model, preds = fit_model(train[target], test[target], order)
    export_forecast(preds, Path(output_dir) / "forecast.csv")
    print(f"Trained ARIMA{order} and saved forecast to {output_dir}")


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Simple time series training")
    parser.add_argument("data", help="CSV file containing a 'date' column and target")
    parser.add_argument("--output", default="outputs", help="directory for results")
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
