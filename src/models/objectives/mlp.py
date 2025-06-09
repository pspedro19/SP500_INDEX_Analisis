from __future__ import annotations

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor


def objective(trial, X, y, cv_splits: int, gap: int, random_seed: int, base_params=None) -> float:
    hidden_neurons = trial.suggest_int("hidden_neurons", 50, 200)
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
    max_iter = trial.suggest_int("max_iter", 200, 1000)

    if base_params:
        hidden_neurons = trial.suggest_int(
            "hidden_neurons", max(base_params.get("hidden_neurons", hidden_neurons) - 30, 50),
            min(base_params.get("hidden_neurons", hidden_neurons) + 50, 250)
        )
        learning_rate_init = trial.suggest_float(
            "learning_rate_init", max(base_params.get("learning_rate_init", learning_rate_init) * 0.5, 1e-4),
            min(base_params.get("learning_rate_init", learning_rate_init) * 2, 1e-2), log=True
        )
        max_iter = trial.suggest_int(
            "max_iter", max(base_params.get("max_iter", max_iter) - 200, 200),
            min(base_params.get("max_iter", max_iter) + 200, 1200)
        )

    tscv = TimeSeriesSplit(n_splits=cv_splits, gap=gap)
    rmses = []
    for train_idx, val_idx in tscv.split(X):
        model = MLPRegressor(
            hidden_layer_sizes=(hidden_neurons, hidden_neurons),
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_seed,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        rmses.append(mean_squared_error(y.iloc[val_idx], preds, squared=False))
    return float(np.mean(rmses))
