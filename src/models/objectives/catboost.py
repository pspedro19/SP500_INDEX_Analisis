from __future__ import annotations

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor


def objective(trial, X, y, cv_splits: int, gap: int, random_seed: int, base_params=None) -> float:
    """Optuna objective for CatBoost."""
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.05, log=True)
    depth = trial.suggest_int("depth", 4, 10)
    iterations = trial.suggest_int("iterations", 200, 2000)

    if base_params:
        learning_rate = trial.suggest_float(
            "learning_rate", max(base_params.get("learning_rate", learning_rate) * 0.5, 0.0005),
            min(base_params.get("learning_rate", learning_rate) * 2, 0.1), log=True
        )
        depth = trial.suggest_int(
            "depth", max(base_params.get("depth", depth) - 2, 3), min(base_params.get("depth", depth) + 2, 12)
        )
        iterations = trial.suggest_int(
            "iterations", max(base_params.get("iterations", iterations) - 300, 100),
            min(base_params.get("iterations", iterations) + 300, 3000)
        )

    tscv = TimeSeriesSplit(n_splits=cv_splits, gap=gap)
    rmses = []
    for train_idx, val_idx in tscv.split(X):
        model = CatBoostRegressor(
            learning_rate=learning_rate,
            depth=depth,
            iterations=iterations,
            random_seed=random_seed,
            verbose=0,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        rmses.append(mean_squared_error(y.iloc[val_idx], preds, squared=False))
    return float(np.mean(rmses))
