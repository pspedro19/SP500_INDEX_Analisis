from __future__ import annotations

import optuna
from sklearn.model_selection import TimeSeriesSplit

from ..evaluation.evaluator import Evaluator


class HyperparameterOptimizer:
    """Basic Optuna optimizer for regression models."""

    def __init__(self, model_class, param_space: dict[str, callable], n_trials: int = 10):
        self.model_class = model_class
        self.param_space = param_space
        self.n_trials = n_trials

    def _objective(self, trial, X, y):
        params = {name: fn(trial) for name, fn in self.param_space.items()}
        model = self.model_class(**params)
        splitter = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in splitter.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            scores.append(Evaluator.rmse(y.iloc[val_idx], preds))
        return sum(scores) / len(scores)

    def optimize(self, X, y) -> dict[str, float]:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: self._objective(t, X, y), n_trials=self.n_trials)
        return study.best_params
