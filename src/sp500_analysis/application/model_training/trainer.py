from __future__ import annotations

from typing import Callable, Dict, Type, Optional

import pandas as pd

from sp500_analysis.application.evaluation.evaluator import Evaluator
from sp500_analysis.application.model_training.hyperparameter_optimizer import (
    HyperparameterOptimizer,
)
from sp500_analysis.shared.container import container, setup_container
from sp500_analysis.infrastructure.models.wrappers import (
    CatBoostWrapper,
    LightGBMWrapper,
    XGBoostWrapper,
    MLPWrapper,
    SVMWrapper,
    LSTMWrapper,
)


DEFAULT_PARAM_SPACES: Dict[Type, Dict[str, Callable]] = {
    CatBoostWrapper: {
        "depth": lambda t: t.suggest_int("depth", 4, 10),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "iterations": lambda t: t.suggest_int("iterations", 100, 500),
    },
    LightGBMWrapper: {
        "num_leaves": lambda t: t.suggest_int("num_leaves", 31, 128),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 500),
    },
    XGBoostWrapper: {
        "max_depth": lambda t: t.suggest_int("max_depth", 3, 10),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 500),
    },
    SVMWrapper: {
        "C": lambda t: t.suggest_float("C", 0.1, 10.0, log=True),
        "epsilon": lambda t: t.suggest_float("epsilon", 0.001, 1.0, log=True),
        "kernel": lambda t: t.suggest_categorical("kernel", ["rbf", "linear"]),
    },
    MLPWrapper: {
        "hidden_layer_sizes": lambda t: (t.suggest_int("hidden_units", 50, 200),),
        "learning_rate_init": lambda t: t.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
    },
    LSTMWrapper: {
        "units": lambda t: t.suggest_int("units", 32, 128),
        "dropout_rate": lambda t: t.suggest_float("dropout_rate", 0.0, 0.5),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "sequence_length": lambda t: t.suggest_int("sequence_length", 5, 20),
    },
}


class ModelTrainer:
    """Train a single model with optional hyperparameter optimization."""

    def __init__(self, n_trials: int = 10, param_spaces: Optional[Dict[Type, Dict[str, Callable]]] = None) -> None:
        self.n_trials = n_trials
        self.param_spaces = param_spaces or DEFAULT_PARAM_SPACES

    def train_model(self, model_cls: Type, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the given model class on the provided data."""
        param_space = self.param_spaces.get(model_cls)
        best_params: Dict[str, float] = {}
        if param_space:
            optimizer = HyperparameterOptimizer(model_cls, param_space, n_trials=self.n_trials)
            best_params = optimizer.optimize(X, y)

        model = model_cls(**best_params)
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
