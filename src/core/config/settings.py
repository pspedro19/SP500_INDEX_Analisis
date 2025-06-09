from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Project configuration loaded from environment variables."""

    # Root paths
    project_root: Path = Path(__file__).resolve().parents[3]
    data_dir: Path | None = None
    raw_dir: Path | None = None
    preprocess_dir: Path | None = None
    ts_prep_dir: Path | None = None
    processed_dir: Path | None = None
    model_input_dir: Path | None = None
    ts_train_dir: Path | None = None
    training_dir: Path | None = None
    results_dir: Path | None = None
    metrics_dir: Path | None = None
    log_dir: Path | None = None
    reports_dir: Path | None = None
    models_dir: Path | None = None
    img_charts_dir: Path | None = None
    metrics_charts_dir: Path | None = None
    subperiods_charts_dir: Path | None = None
    csv_reports_dir: Path | None = None

    # Common columns
    date_col: str = "date"
    id_col: str = "id"
    target_suffix: str = "_Target"

    # Forecasting parameters
    forecast_horizon_1month: int = 20
    forecast_horizon_3months: int = 60
    local_refinement_days: int = 225
    train_test_split_ratio: float = 0.8

    # Validation
    cv_splits: int = 5
    cv_gap_1month: int = 20
    cv_gap_3months: int = 60

    # Feature selection thresholds
    constant_threshold: float = 0.95
    corr_threshold: float = 0.85
    vif_threshold: int = 10
    fpi_threshold: float = 0.000001

    # General settings
    random_seed: int = 42
    scorer: str = "neg_mean_squared_error"

    catboost_params: dict = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "early_stopping_rounds": 50,
        "verbose": 0,
        "random_seed": 42,
        "n_estimators": 500,
        "learning_rate": 0.01,
        "max_depth": 8,
    }

    class Config:
        env_file = os.getenv("ENV_FILE", ".env")
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        prj = self.project_root
        self.data_dir = Path(os.getenv("DATA_DIR", self.data_dir or prj / "data"))
        self.raw_dir = Path(os.getenv("RAW_DIR", self.raw_dir or self.data_dir / "0_raw"))
        self.preprocess_dir = Path(os.getenv("PREPROCESS_DIR", self.preprocess_dir or self.data_dir / "1_preprocess"))
        self.ts_prep_dir = Path(os.getenv("TS_PREP_DIR", self.ts_prep_dir or self.data_dir / "1_preprocess_ts"))
        self.processed_dir = Path(os.getenv("PROCESSED_DIR", self.processed_dir or self.data_dir / "2_processed"))
        self.model_input_dir = Path(os.getenv("MODEL_INPUT_DIR", self.model_input_dir or self.data_dir / "2_model_input"))
        self.ts_train_dir = Path(os.getenv("TS_TRAIN_DIR", self.ts_train_dir or self.data_dir / "2_trainingdata_ts"))
        self.training_dir = Path(os.getenv("TRAINING_DIR", self.training_dir or self.data_dir / "3_trainingdata"))
        self.results_dir = Path(os.getenv("RESULTS_DIR", self.results_dir or self.data_dir / "4_results"))
        self.metrics_dir = Path(os.getenv("METRICS_DIR", self.metrics_dir or self.data_dir / "5_metrics"))
        self.log_dir = Path(os.getenv("LOG_DIR", self.log_dir or prj / "logs"))
        self.reports_dir = Path(os.getenv("REPORTS_DIR", self.reports_dir or prj / "reports"))
        self.models_dir = Path(os.getenv("MODELS_DIR", self.models_dir or prj / "models"))
        self.img_charts_dir = Path(os.getenv("IMG_CHARTS_DIR", self.img_charts_dir or self.results_dir / "charts"))
        self.metrics_charts_dir = Path(os.getenv("METRICS_CHARTS_DIR", self.metrics_charts_dir or self.metrics_dir / "charts"))
        self.subperiods_charts_dir = Path(os.getenv("SUBPERIODS_CHARTS_DIR", self.subperiods_charts_dir or self.metrics_charts_dir / "subperiods"))
        self.csv_reports_dir = Path(os.getenv("CSV_REPORTS_DIR", self.csv_reports_dir or self.reports_dir / "csv"))

        # Convenience aliases
        self.data_raw = self.raw_dir
        self.data_prep = self.preprocess_dir
        self.csv_reports = self.csv_reports_dir

        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        dirs = [
            self.raw_dir,
            self.preprocess_dir,
            self.ts_prep_dir,
            self.processed_dir,
            self.model_input_dir,
            self.ts_train_dir,
            self.training_dir,
            self.results_dir,
            self.metrics_dir,
            self.log_dir,
            self.reports_dir,
            self.models_dir,
            self.img_charts_dir,
            self.metrics_charts_dir,
            self.csv_reports_dir,
            self.subperiods_charts_dir,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()

