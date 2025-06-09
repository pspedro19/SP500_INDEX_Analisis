from __future__ import annotations

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


def objective(trial, X, y, cv_splits: int, gap: int, random_seed: int, base_params=None) -> float:
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.05, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 12)
    n_estimators = trial.suggest_int("n_estimators", 200, 2000)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)

    if base_params:
        learning_rate = trial.suggest_float(
            "learning_rate", max(base_params.get("learning_rate", learning_rate) * 0.5, 0.0005),
            min(base_params.get("learning_rate", learning_rate) * 2, 0.1), log=True
        )
        max_depth = trial.suggest_int(
            "max_depth", max(base_params.get("max_depth", max_depth) - 2, 3), min(base_params.get("max_depth", max_depth) + 2, 12)
        )
        n_estimators = trial.suggest_int(
            "n_estimators", max(base_params.get("n_estimators", n_estimators) - 300, 100),
            min(base_params.get("n_estimators", n_estimators) + 300, 3000)
        )
        subsample = trial.suggest_float(
            "subsample", max(base_params.get("subsample", subsample) - 0.2, 0.5), min(base_params.get("subsample", subsample) + 0.2, 1.0)
        )

    tscv = TimeSeriesSplit(n_splits=cv_splits, gap=gap)
    rmses = []
    for train_idx, val_idx in tscv.split(X):
        model = lgb.LGBMRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=subsample,
            random_state=random_seed,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        rmses.append(mean_squared_error(y.iloc[val_idx], preds, squared=False))
    return float(np.mean(rmses))
