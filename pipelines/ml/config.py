"""Utility helpers for ML pipelines."""

from pathlib import Path

from sp500_analysis.config import settings

__all__ = ["settings", "ensure_directories"]


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
