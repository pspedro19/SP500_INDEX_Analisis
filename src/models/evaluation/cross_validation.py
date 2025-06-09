from __future__ import annotations

import joblib
from pathlib import Path
from typing import Tuple

from .metrics import calcular_metricas_basicas

from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


def _get_model(algo: str, params: dict, random_seed: int = 42):
    if algo == "CatBoost":
        return CatBoostRegressor(random_seed=random_seed, verbose=0, **params)
    if algo == "LightGBM":
        return lgb.LGBMRegressor(random_state=random_seed, **params)
    if algo == "XGBoost":
        return xgb.XGBRegressor(random_state=random_seed, verbosity=0, **params)
    if algo == "MLP":
        return MLPRegressor(random_state=random_seed, **params)
    if algo == "SVM":
        return SVR(**params)
    raise ValueError(f"Unknown algorithm {algo}")


def evaluate_backtest(algo: str, X_train, y_train, X_val, y_val, params: dict, random_seed: int = 42) -> Tuple[object, dict]:
    model = _get_model(algo, params, random_seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    metrics = calcular_metricas_basicas(y_val, preds)
    return model, metrics


def evaluate_holdout(algo: str, X_train, y_train, X_test, y_test, params: dict, random_seed: int = 42) -> Tuple[object, dict]:
    model = _get_model(algo, params, random_seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = calcular_metricas_basicas(y_test, preds)
    return model, metrics


def save_model(model, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
