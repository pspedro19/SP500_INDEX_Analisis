import json
import logging
import os
from typing import Dict

import pandas as pd

from pipelines.ml.config import RESULTS_DIR, TRAINING_DIR, ensure_directories
from ...infrastructure.compute.gpu_manager import configure_gpu
from ...infrastructure.models.wrappers import (
    CatBoostWrapper,
    LightGBMWrapper,
    MLPWrapper,
    SVMWrapper,
    XGBoostWrapper,
)
from ..evaluation.evaluator import Evaluator


def load_latest_training_file() -> str | None:
    files = [f for f in os.listdir(TRAINING_DIR) if f.endswith(".xlsx")]
    if not files:
        return None
    files = [os.path.join(TRAINING_DIR, f) for f in files]
    return max(files, key=os.path.getmtime)


def run_training() -> Dict[str, Dict[str, float]]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ensure_directories()
    configure_gpu(use_gpu=True)

    input_file = load_latest_training_file()
    if not input_file:
        logging.error("No training data available")
        return {}

    df = pd.read_excel(input_file)
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    models = {
        "CatBoost": CatBoostWrapper,
        "LightGBM": LightGBMWrapper,
        "XGBoost": XGBoostWrapper,
        "MLP": MLPWrapper,
        "SVM": SVMWrapper,
    }

    results = {}
    for name, cls in models.items():
        logging.info("Training %s", name)
        model = cls()
        model.fit(X, y)
        preds = model.predict(X)
        results[name] = Evaluator.evaluate(y, preds)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_file = os.path.join(RESULTS_DIR, "metrics_simple.json")
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=4)
    logging.info("Metrics saved to %s", metrics_file)
    return results


def main():
    run_training()


if __name__ == "__main__":  # pragma: no cover
    main()
