"""Pipeline step to clean column names in the merged dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sp500_analysis.config.settings import settings
from sp500_analysis.shared.logging.logger import setup_logging
from sp500_analysis.application.preprocessing import (
    STANDARD_NAMES,
    clean_dataframe_columns,
)


def main() -> None:
    logger = setup_logging(settings.log_dir, "clean_columns")

    input_file = Path(settings.preprocess_dir) / "MERGEDEXCELS_CATEGORIZADO.xlsx"
    output_dir = Path(settings.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "MERGEDEXCELS_CATEGORIZADO_LIMPIO.xlsx"

    if not input_file.exists():
        logger.error(f"El archivo de entrada no existe: {input_file}")
        return

    df = pd.read_excel(input_file)
    logger.info("Archivo cargado correctamente. Columnas: %d", len(df.columns))

    df, renamed = clean_dataframe_columns(df, STANDARD_NAMES)

    if renamed:
        logger.info("Se modificaron %d nombres de columnas.", len(renamed))
        for original, nuevo in renamed.items():
            logger.info("Renombrando: '%s' -> '%s'", original, nuevo)
    else:
        logger.info("No se encontraron columnas que necesiten ser renombradas.")

    df.to_excel(output_file, index=False)
    logger.info("Archivo con columnas limpias guardado en: %s", output_file)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
