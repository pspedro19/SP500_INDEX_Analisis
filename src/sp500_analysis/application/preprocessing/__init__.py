"""Preprocessing utilities and processors."""

from .cleaning import STANDARD_NAMES, clean_dataframe_columns, limpiar_nombre_columna

# Note: legacy_step_0.py contains the complete original preprocessing pipeline
# It can be imported via importlib.util when needed by the pipeline orchestrator

__all__ = [
    "STANDARD_NAMES",
    "limpiar_nombre_columna",
    "clean_dataframe_columns",
]
