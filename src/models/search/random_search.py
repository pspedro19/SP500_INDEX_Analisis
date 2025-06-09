from __future__ import annotations

import os
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import uniform, randint


PARAM_DISTS = {
    "CatBoost": {
        "learning_rate": uniform(0.01, 0.04),
        "depth": randint(4, 7),
        "iterations": randint(50, 151),
    },
    "LightGBM": {
        "learning_rate": uniform(0.01, 0.04),
        "max_depth": randint(4, 7),
        "n_estimators": randint(50, 151),
        "subsample": uniform(0.7, 0.3),
    },
    "XGBoost": {
        "learning_rate": uniform(0.01, 0.04),
        "max_depth": randint(4, 7),
        "n_estimators": randint(50, 151),
        "subsample": uniform(0.7, 0.3),
    },
    "MLP": {
        "hidden_neurons": randint(50, 101),
        "learning_rate_init": uniform(0.001, 0.009),
        "max_iter": randint(50, 201),
    },
    "SVM": {
        "C": uniform(0.5, 4.5),
        "epsilon": uniform(0.05, 0.15),
    },
}


MODEL_FACTORY = {
    "CatBoost": lambda rs: __import__("catboost").CatBoostRegressor(random_seed=rs, verbose=0),
    "LightGBM": lambda rs: __import__("lightgbm").LGBMRegressor(random_state=rs),
    "XGBoost": lambda rs: __import__("xgboost").XGBRegressor(random_state=rs, verbosity=0),
    "MLP": lambda rs: __import__("sklearn.neural_network").neural_network.MLPRegressor(random_state=rs),
    "SVM": lambda rs: __import__("sklearn.svm").svm.SVR(),
}


def run_randomized_search(algo_name: str, X, y, cv_splits: int, gap: int, n_iter: int = 10, random_seed: int = 42):
    if algo_name not in PARAM_DISTS:
        return None
    model = MODEL_FACTORY[algo_name](random_seed)
    tscv = TimeSeriesSplit(n_splits=cv_splits, gap=gap)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=PARAM_DISTS[algo_name],
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=random_seed,
        n_jobs=min(4, os.cpu_count() or 1),
    )
    search.fit(X, y)
    return search.best_params_
