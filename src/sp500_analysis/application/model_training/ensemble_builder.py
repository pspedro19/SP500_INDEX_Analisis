from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    import pandas as pd
    import numpy as np


from sp500_analysis.config.settings import settings

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from sp500_analysis.domain.data_repository import DataRepository


class GreedyEnsembleRegressor:
    """Simple greedy ensemble for regression models."""

    def __init__(self, models: List, model_names: List[str] | None = None) -> None:
        self.models = models
        self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
        self.selected_models: List = []
        self.selected_names: List[str] = []
        self.score_history: List[float] = []
        self.individual_scores: dict[str, float] = {}

    def fit(self, X, y) -> None:
        import numpy as np
        import pandas as pd
        from sklearn.metrics import mean_squared_error
        n = len(X)
        val_size = max(1, int(n * 0.2))
        X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
        y_train, y_val = y.iloc[:-val_size], y.iloc[-val_size:]

        all_preds: List[np.ndarray] = []
        all_scores: List[float] = []

        for name, model in zip(self.model_names, self.models):
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            all_preds.append(preds)
            score = float(np.sqrt(mean_squared_error(y_val, preds)))
            all_scores.append(score)

        self.individual_scores = dict(zip(self.model_names, all_scores))

        remaining = list(range(len(self.models)))
        selected: List[int] = []
        best_score = float("inf")

        while remaining:
            best_idx = -1
            for idx in remaining:
                current = selected + [idx]
                ensemble_pred = np.mean([all_preds[i] for i in current], axis=0)
                score = float(np.sqrt(mean_squared_error(y_val, ensemble_pred)))
                if score < best_score:
                    best_score = score
                    best_idx = idx
            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
            self.score_history.append(best_score)
            self.selected_models.append(self.models[best_idx])
            self.selected_names.append(self.model_names[best_idx])

            if len(selected) > 1 and best_score >= self.score_history[-2]:
                break

    def predict(self, X):
        import numpy as np
        if not self.selected_models:
            raise ValueError("Ensemble has not been fitted")
        preds = np.array([m.predict(X) for m in self.selected_models])
        return np.mean(preds, axis=0)

    def metrics(self, X_val, y_val) -> dict[str, float]:
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        pred = self.predict(X_val)
        return {
            "RMSE": float(np.sqrt(mean_squared_error(y_val, pred))),
            "MAE": float(mean_absolute_error(y_val, pred)),
            "R2": float(r2_score(y_val, pred)),
            "Num_Models": len(self.selected_models),
        }


class EnsembleBuilder:
    """Build and persist a greedy ensemble using trained models."""

    def __init__(self, repository: DataRepository) -> None:
        self.repository = repository
        self.models_dir = Path(settings.models_dir)
        self.results_dir = Path(settings.results_dir)

    def build(self) -> Path | None:
        df = self.repository.load_latest()
        if df is None:
            logging.error("No training data available for ensemble building")
            return None

        target = df.columns[-1]
        X = df.drop(columns=[target])
        y = df[target]

        import joblib

        model_files = list(self.models_dir.glob("*.pkl"))
        models = []
        names = []
        for path in model_files:
            name = path.stem
            if "ensemble" in name.lower() or name.startswith("."):
                continue
            try:
                models.append(joblib.load(path))
                names.append(name)
            except Exception as exc:
                logging.error("Failed loading %s: %s", path, exc)

        if not models:
            logging.error("No base models found for ensemble")
            return None

        ensemble = GreedyEnsembleRegressor(models, names)
        ensemble.fit(X, y)

        ensemble_path = self.models_dir / "ensemble_greedy.pkl"
        joblib.dump(ensemble, ensemble_path)

        info = {
            "selected_models": ensemble.selected_names,
            "metrics": ensemble.metrics(X.iloc[-int(len(X)*0.2):], y.iloc[-int(len(y)*0.2):]),
            "score_history": ensemble.score_history,
            "individual_scores": ensemble.individual_scores,
        }
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_dir / "ensemble_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4)

        logging.info("Ensemble built with %s models", len(ensemble.selected_models))
        return ensemble_path


def run_ensemble() -> Path | None:
    """Entry point used by CLI and pipeline scripts."""
    from sp500_analysis.shared.container import container, setup_container

    setup_container()
    builder: EnsembleBuilder = container.resolve("ensemble_builder")
    return builder.build()


if __name__ == "__main__":  # pragma: no cover
    run_ensemble()
