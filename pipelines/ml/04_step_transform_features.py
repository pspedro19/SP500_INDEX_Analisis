#!/usr/bin/env python
"""Pipeline step that performs feature engineering."""
from __future__ import annotations

from pathlib import Path
import argparse

from sp500_analysis.application.feature_engineering import FeatureEngineeringService


def main(
    input_file: str = "data/2_processed/MERGEDEXCELS_CATEGORIZADO_LIMPIO.xlsx",
    output_file: str = "data/2_processed/features.xlsx",
    target_column: str = "S&P500",
    use_lag: bool = True,
    lag_days: int = 20,
    lag_type: str = "future",
) -> None:
    service = FeatureEngineeringService(Path(__file__).resolve().parents[2])
    service.run(
        input_file=input_file,
        output_file=output_file,
        target_column=target_column,
        use_lag=use_lag,
        lag_days=lag_days,
        lag_type=lag_type,
    )


if __name__ == "__main__":  # pragma: no cover - manual execution
    parser = argparse.ArgumentParser(description="Feature engineering step")
    parser.add_argument("--input_file", default="data/2_processed/MERGEDEXCELS_CATEGORIZADO_LIMPIO.xlsx")
    parser.add_argument("--output_file", default="data/2_processed/features.xlsx")
    parser.add_argument("--target_column", default="S&P500")
    parser.add_argument("--use_lag", action="store_true")
    parser.add_argument("--lag_days", type=int, default=20)
    parser.add_argument("--lag_type", choices=["future", "past", "current"], default="future")
    args = parser.parse_args()
    main(
        input_file=args.input_file,
        output_file=args.output_file,
        target_column=args.target_column,
        use_lag=args.use_lag,
        lag_days=args.lag_days,
        lag_type=args.lag_type,
    )

