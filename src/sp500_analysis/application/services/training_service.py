from __future__ import annotations

import json
import logging
from datetime import datetime
from sp500_analysis.shared.logging.logger import configurar_logging
import os
from typing import Dict


from pipelines.ml.config import ensure_directories
from sp500_analysis.config.settings import settings
from sp500_analysis.application.model_training.trainer import ModelTrainer
from sp500_analysis.domain.data_repository import DataRepository
from sp500_analysis.infrastructure.compute.gpu_manager import configure_gpu
from sp500_analysis.infrastructure.models.registry import ModelRegistry

RESULTS_DIR = settings.results_dir


class TrainingService:
    """Service orchestrating full model training."""

    def __init__(self, repository: DataRepository, trainer: ModelTrainer, registry: ModelRegistry) -> None:
        self.repository = repository
        self.trainer = trainer
        self.registry = registry

    def run_training(self) -> Dict[str, Dict[str, float]]:
        log_file = settings.log_dir / f"training_service_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        configurar_logging(str(log_file))
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
        for name, cls in self.registry.items():
            logging.info("Training %s", name)
            results[name] = self.trainer.train_model(cls, X, y)

        os.makedirs(RESULTS_DIR, exist_ok=True)
        metrics_file = os.path.join(RESULTS_DIR, "metrics_simple.json")
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=4)
        logging.info("Metrics saved to %s", metrics_file)
        return results


def run_training() -> Dict[str, Dict[str, float]]:
    """Entry point used by the CLI to run training via the DI container."""
    from sp500_analysis.shared.container import container, setup_container

    setup_container()
    service: TrainingService = container.resolve("training_service")
    return service.run_training()
