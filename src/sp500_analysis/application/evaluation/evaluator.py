from __future__ import annotations

from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Evaluator:
    """Simple regression metrics helper."""

    @staticmethod
    def rmse(y_true, y_pred) -> float:
        return sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true, y_pred) -> float:
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def r2(y_true, y_pred) -> float:
        return r2_score(y_true, y_pred)

    @staticmethod
    def evaluate(y_true, y_pred) -> dict[str, float]:
        return {
            "RMSE": Evaluator.rmse(y_true, y_pred),
            "MAE": Evaluator.mae(y_true, y_pred),
            "R2": Evaluator.r2(y_true, y_pred),
        }
