#!/usr/bin/env python
"""Pipeline step to compute S&P500 predicted values."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
import os

from sp500_analysis.config.settings import settings
from sp500_analysis.shared.logging.logger import configurar_logging
from sp500_analysis.application.inference.data_loader import load_csv, save_csv
from sp500_analysis.application.inference.calculations import (
    compute_predicted_sp500,
    format_for_powerbi,
)


log_file = os.path.join(
    settings.log_dir,
    f"calculo_valor_sp500_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)
configurar_logging(log_file)


def main(
    input_file: str = "data/4_results/hechos_predicciones_fields_con_sp500.csv",
    output_file: str = "data/4_results/hechos_predicciones_sp500_powerbi.csv",
) -> bool:
    """Compute and store predicted S&P500 values."""
    df = load_csv(input_file)
    df = compute_predicted_sp500(df)
    df = format_for_powerbi(df)
    save_csv(df, output_file)
    logging.info("File saved to %s", output_file)
    return True


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Calculate S&P500 predicted values")
    parser.add_argument("--input_file", default="data/4_results/hechos_predicciones_fields_con_sp500.csv")
    parser.add_argument("--output_file", default="data/4_results/hechos_predicciones_sp500_powerbi.csv")
    args = parser.parse_args()
    main(input_file=args.input_file, output_file=args.output_file)
