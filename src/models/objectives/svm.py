from __future__ import annotations

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


def objective(trial, X, y, cv_splits: int, gap: int, random_seed: int, base_params=None) -> float:
    C = trial.suggest_float("C", 0.1, 10, log=True)
    epsilon = trial.suggest_float("epsilon", 0.01, 0.5)

    if base_params:
        C = trial.suggest_float(
            "C", max(base_params.get("C", C) * 0.5, 0.1), min(base_params.get("C", C) * 2, 20), log=True
        )
        epsilon = trial.suggest_float(
            "epsilon", max(base_params.get("epsilon", epsilon) * 0.5, 0.01), min(base_params.get("epsilon", epsilon) * 2, 1.0)
        )

    tscv = TimeSeriesSplit(n_splits=cv_splits, gap=gap)
    rmses = []
    for train_idx, val_idx in tscv.split(X):
        model = SVR(C=C, epsilon=epsilon)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        rmses.append(mean_squared_error(y.iloc[val_idx], preds, squared=False))
    return float(np.mean(rmses))
