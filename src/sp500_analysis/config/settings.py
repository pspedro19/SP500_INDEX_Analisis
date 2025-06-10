from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Central project configuration loaded from environment variables."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    data_path: Path = field(default_factory=lambda: Path(os.getenv("DATA_PATH", "./data")))

    db_host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", 5432)))
    db_user: str = field(default_factory=lambda: os.getenv("DB_USER", "user"))
    db_password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", "password"))
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", "sp500"))

    alpha_vantage_key: str = field(default_factory=lambda: os.getenv("ALPHA_VANTAGE_KEY", ""))
    another_api_key: str = field(default_factory=lambda: os.getenv("ANOTHER_API_KEY", ""))

    def __post_init__(self) -> None:
        self.root = self.project_root
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "0_raw"
        self.preprocess_dir = self.data_dir / "1_preprocess"
        self.ts_prep_dir = self.data_dir / "1_preprocess_ts"
        self.processed_dir = self.data_dir / "2_processed"
        self.model_input_dir = self.data_dir / "2_model_input"
        self.ts_train_dir = self.data_dir / "2_trainingdata_ts"
        self.training_dir = self.data_dir / "3_trainingdata"
        self.results_dir = self.data_dir / "4_results"
        self.metrics_dir = self.data_dir / "5_metrics"
        self.log_dir = self.project_root / "logs"
        self.reports_dir = self.project_root / "reports"
        self.models_dir = self.project_root / "models"

        self.img_charts_dir = self.results_dir / "charts"
        self.metrics_charts_dir = self.metrics_dir / "charts"
        self.subperiods_charts_dir = self.metrics_charts_dir / "subperiods"
        self.csv_reports_dir = self.reports_dir / "csv"

        self.date_col = "date"
        self.id_col = "id"
        self.target_suffix = "_Target"

        self.forecast_horizon_1month = 20
        self.forecast_horizon_3months = 60
        self.local_refinement_days = 225
        self.train_test_split_ratio = 0.8

        self.cv_splits = 5
        self.cv_gap_1month = self.forecast_horizon_1month
        self.cv_gap_3months = self.forecast_horizon_3months

        self.constant_threshold = 0.95
        self.corr_threshold = 0.85
        self.vif_threshold = 10
        self.fpi_threshold = 0.000001

        self.random_seed = 42
        self.scorer = "neg_mean_squared_error"

        self.catboost_params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "early_stopping_rounds": 50,
            "verbose": 0,
            "random_seed": self.random_seed,
            "n_estimators": 500,
            "learning_rate": 0.01,
            "max_depth": 8,
        }

        self.algorithm_colors = {
            "CatBoost": "#e377c2",
            "LightGBM": "#7f7f7f",
            "XGBoost": "#bcbd22",
            "MLP": "#17becf",
            "SVM": "#1f77b4",
            "Ensemble": "#ff7f0e",
        }

        self.periodo_labels = {
            "Training": "Entrenamiento",
            "Validation": "Validación",
            "Test": "Test",
            "Forecast": "Pronóstico",
        }

        self.data_raw = self.raw_dir
        self.data_prep = self.preprocess_dir
        self.csv_reports = self.csv_reports_dir


settings = Settings()
