"""Coordinación de entrenamiento de modelos."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import optuna
import pandas as pd

from src.config.base import ProjectConfig
from src.models.search.random_search import run_randomized_search
from src.models.objectives import (
    catboost,
    lightgbm,
    xgboost,
    mlp,
    svm,
    lstm,
)
from src.models.evaluation.cross_validation import (
    evaluate_backtest,
    evaluate_holdout,
    save_model,
)


OBJECTIVES = {
    "CatBoost": catboost.objective,
    "LightGBM": lightgbm.objective,
    "XGBoost": xgboost.objective,
    "MLP": mlp.objective,
    "SVM": svm.objective,
    # "LSTM": lstm.objective,  # Activar si se dispone de TensorFlow
}


def load_latest_file(directory: Path) -> Path:
    files = [directory / f for f in os.listdir(directory) if f.endswith(".xlsx")]
    if not files:
        raise FileNotFoundError("No training files found")
    return max(files, key=lambda p: p.stat().st_mtime)


def split_zones(X, y):
    cut_70 = int(len(X) * 0.7)
    cut_90 = int(len(X) * 0.9)
    return (
        (X.iloc[:cut_70], y.iloc[:cut_70]),
        (X.iloc[cut_70:cut_90], y.iloc[cut_70:cut_90]),
        (X.iloc[cut_90:], y.iloc[cut_90:]),
    )


def main() -> None:
    config = ProjectConfig.from_env()
    logging.basicConfig(level=logging.INFO)

    input_file = load_latest_file(config.training_dir)
    df = pd.read_excel(input_file)
    df.sort_values(config.date_col, inplace=True)
    target_col = df.columns[-1]
    y = df[target_col]
    X = df.drop(columns=[target_col, config.date_col])

    (X_a, y_a), (X_b, y_b), (X_c, y_c) = split_zones(X, y)

    for algo, objective in OBJECTIVES.items():
        logging.info("Training %s", algo)
        base_params = run_randomized_search(
            algo, X_a, y_a, config.cv_splits, config.cv_gap_1month, n_iter=3, random_seed=config.random_seed
        )
        study = optuna.create_study(direction="minimize")
        func = lambda trial: objective(
            trial, X_a, y_a, config.cv_splits, config.cv_gap_1month, config.random_seed, base_params
        )
        study.optimize(func, n_trials=2)
        model, back_metrics = evaluate_backtest(
            algo, X_a, y_a, X_b, y_b, study.best_params, config.random_seed
        )
        model, hold_metrics = evaluate_holdout(
            algo, pd.concat([X_a, X_b]), pd.concat([y_a, y_b]), X_c, y_c, study.best_params, config.random_seed
        )
        save_model(model, config.models_dir / f"{algo.lower()}_model.pkl")
        logging.info("%s backtest RMSE %.4f", algo, back_metrics["RMSE"])
        logging.info("%s holdout RMSE %.4f", algo, hold_metrics["RMSE"])


if __name__ == "__main__":
    main()
