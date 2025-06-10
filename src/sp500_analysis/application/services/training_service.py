from __future__ import annotations

import json
import logging
import os
from typing import Dict


from pipelines.ml.config import ensure_directories
from sp500_analysis.config.settings import settings
from sp500_analysis.application.model_training.trainer import ModelTrainer
from sp500_analysis.domain.data_repository import DataRepository
from sp500_analysis.infrastructure.compute.gpu_manager import configure_gpu
from sp500_analysis.infrastructure.models.wrappers import (
    CatBoostWrapper,
    LightGBMWrapper,
    MLPWrapper,
    SVMWrapper,
    XGBoostWrapper,
)

RESULTS_DIR = settings.results_dir


class TrainingService:
    """Service orchestrating full model training."""

    def __init__(self, repository: DataRepository, trainer: ModelTrainer) -> None:
        self.repository = repository
        self.trainer = trainer
        self.models = {
            "CatBoost": CatBoostWrapper,
            "LightGBM": LightGBMWrapper,
            "XGBoost": XGBoostWrapper,
            "MLP": MLPWrapper,
            "SVM": SVMWrapper,
        }

    def run_training(self) -> Dict[str, Dict[str, float]]:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        ensure_directories()
        configure_gpu(use_gpu=True)

        df = self.repository.load_latest()
        if df is None:
            logging.error("No training data available")
            return {}

        target = df.columns[-1]
        X = df.drop(columns=[target])
        y = df[target]

        results: Dict[str, Dict[str, float]] = {}
        for name, cls in self.models.items():
            logging.info("Training %s", name)
            results[name] = self.trainer.train_model(cls, X, y)

        os.makedirs(RESULTS_DIR, exist_ok=True)
        metrics_file = os.path.join(RESULTS_DIR, "metrics_simple.json")
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=4)
        logging.info("Metrics saved to %s", metrics_file)
        return results
