from __future__ import annotations

import argparse
from pathlib import Path

from sp500_analysis.application.time_series_training.training_pipeline import (
    run_training_pipeline,
)


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Run simple time series pipeline")
    parser.add_argument("data", help="CSV file with date column and target")
    parser.add_argument("--output", default="outputs", help="directory for results")
    parser.add_argument(
        "--ensemble", nargs="*", default=None, help="extra forecast csv files to ensemble"
    )
    args = parser.parse_args()
    run_training_pipeline(args.data, args.output, args.ensemble)


if __name__ == "__main__":  # pragma: no cover
    main()
