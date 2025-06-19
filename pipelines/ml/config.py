"""Utility helpers for ML pipelines."""

from pathlib import Path

from sp500_analysis.config import settings

# Agregamos PROJECT_ROOT para compatibilidad con código estable
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Variables que necesita el código original
MODELS_DIR = str(settings.models_dir)
TRAINING_DIR = str(settings.training_dir)
RESULTS_DIR = str(settings.results_dir)
IMG_CHARTS_DIR = str(settings.img_charts_dir)
DATE_COL = "date"
LOCAL_REFINEMENT_DAYS = 225
TRAIN_TEST_SPLIT_RATIO = 0.8
FORECAST_HORIZON_1MONTH = 20
FORECAST_HORIZON_3MONTHS = 60
RANDOM_SEED = 42

__all__ = ["settings", "ensure_directories", "PROJECT_ROOT", 
           "MODELS_DIR", "TRAINING_DIR", "RESULTS_DIR", "IMG_CHARTS_DIR",
           "DATE_COL", "LOCAL_REFINEMENT_DAYS", "TRAIN_TEST_SPLIT_RATIO",
           "FORECAST_HORIZON_1MONTH", "FORECAST_HORIZON_3MONTHS", "RANDOM_SEED"]


def ensure_directories() -> None:
    """Create all required directories for pipeline artifacts."""
    dirs = [
        settings.raw_dir,
        settings.preprocess_dir,
        settings.ts_prep_dir,
        settings.processed_dir,
        settings.model_input_dir,
        settings.ts_train_dir,
        settings.training_dir,
        settings.results_dir,
        settings.metrics_dir,
        settings.log_dir,
        settings.reports_dir,
        settings.models_dir,
        settings.img_charts_dir,
        settings.metrics_charts_dir,
        settings.csv_reports_dir,
        settings.subperiods_charts_dir,
    ]
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
