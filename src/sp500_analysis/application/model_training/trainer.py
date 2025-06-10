from __future__ import annotations

from typing import Dict, Type

import pandas as pd

from sp500_analysis.application.evaluation.evaluator import Evaluator
from sp500_analysis.shared.container import container, setup_container


class ModelTrainer:
    """Train a single model and return evaluation metrics."""

    def train_model(self, model_cls: Type, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        model = model_cls()
        model.fit(X, y)
        preds = model.predict(X)
        return Evaluator.evaluate(y, preds)


def run_training() -> Dict[str, Dict[str, float]]:
    """Entry point used by the pipeline to run training via the DI container."""
    setup_container()
    service = container.resolve("training_service")
    return service.run_training()


def main() -> None:
    run_training()


if __name__ == "__main__":  # pragma: no cover
    main()
