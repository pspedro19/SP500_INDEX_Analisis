from __future__ import annotations

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def calcular_metricas_basicas(y_true, y_pred):
    """Return basic regression metrics in a dict."""
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    smape = 100 * np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "SMAPE": smape}
