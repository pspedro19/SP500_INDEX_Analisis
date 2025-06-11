"""Utilities to perform model inference and forecasting."""

from __future__ import annotations

import glob
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - numpy may be unavailable
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - pandas may be unavailable
    pd = None

from sp500_analysis.config.settings import settings

DATE_COL = settings.date_col
LOCAL_REFINEMENT_DAYS = settings.local_refinement_days
TRAIN_TEST_SPLIT_RATIO = settings.train_test_split_ratio
FORECAST_HORIZON_1MONTH = settings.forecast_horizon_1month
FORECAST_HORIZON_3MONTHS = settings.forecast_horizon_3months
IMG_CHARTS = settings.img_charts_dir
RESULTS_DIR = settings.results_dir


def get_most_recent_file(directory: str | os.PathLike, pattern: str = "*.xlsx") -> str | None:
    """Return the most recent file matching the given pattern."""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_all_models(models_dir: str | os.PathLike) -> Dict[str, Any]:
    """Load all pickle models from the given directory."""
    t0 = time.perf_counter()
    models: Dict[str, Any] = {}
    from importlib import import_module

    try:
        joblib = import_module("joblib")  # type: ignore
    except Exception:  # pragma: no cover - joblib optional
        logging.error("joblib not available")
        return {}

    ensemble_path = Path(models_dir) / "ensemble_greedy.pkl"
    if ensemble_path.exists():
        try:
            models["Ensemble"] = joblib.load(ensemble_path)
            logging.info("Modelo ensemble cargado desde %s", ensemble_path)
        except Exception as exc:  # pragma: no cover - runtime failure
            logging.error("Error al cargar ensemble: %s", exc)
    for model_path in Path(models_dir).glob("*.pkl"):
        name = model_path.stem
        if "ensemble" in name.lower() or name.startswith("."):
            continue
        try:
            models[name] = joblib.load(model_path)
            logging.info("Modelo %s cargado desde %s", name, model_path)
        except Exception as exc:  # pragma: no cover - runtime failure
            logging.error("Error al cargar %s: %s", name, exc)
    if not models:
        logging.error("No se pudo cargar ningún modelo")
        return {}
    logging.info("Total de modelos cargados: %s", len(models))
    logging.info("Tiempo de carga de modelos: %.2fs", time.perf_counter() - t0)
    return models


def refine_model_locally(model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
    """Refit the model on the most recent LOCAL_REFINEMENT_DAYS samples."""
    if pd is None or np is None:
        raise ImportError("pandas and numpy are required for refine_model_locally")
    t0 = time.perf_counter()
    if not hasattr(model, "fit"):
        logging.warning("El modelo no tiene método 'fit'")
        return model
    if len(X) > LOCAL_REFINEMENT_DAYS:
        X_local = X.tail(LOCAL_REFINEMENT_DAYS)
        y_local = y.tail(LOCAL_REFINEMENT_DAYS)
    else:
        X_local = X.copy()
        y_local = y.copy()
    train_size = int(len(X_local) * TRAIN_TEST_SPLIT_RATIO)
    X_train = X_local.iloc[:train_size]
    y_train = y_local.iloc[:train_size]
    X_test = X_local.iloc[train_size:]
    y_test = y_local.iloc[train_size:]
    try:
        from sklearn.metrics import mean_squared_error

        model.fit(X_train, y_train)
        if len(X_test) > 0:
            pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            logging.info("RMSE local tras refinamiento: %.4f", rmse)
        logging.info("Tiempo de refinamiento local: %.2fs", time.perf_counter() - t0)
        return model
    except Exception as exc:  # pragma: no cover - runtime failure
        logging.error("Error al refinar modelo localmente: %s", exc)
        return model


def get_inference_for_all_models(
    models: Dict[str, Any],
    dataset: pd.DataFrame,
    *,
    date_inference: str | datetime | None = None,
    forecast_period: str = "1MONTH",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, pd.DataFrame]]:
    """Run inference for each model returning results and forecast DataFrames."""
    if pd is None or np is None:
        raise ImportError("pandas and numpy are required for get_inference_for_all_models")

    t0 = time.perf_counter()
    results: Dict[str, Dict[str, Any]] = {}
    forecasts_df: Dict[str, pd.DataFrame] = {}
    if not models:
        logging.error("No se proporcionaron modelos válidos para la inferencia")
        return results, forecasts_df
    horizon = FORECAST_HORIZON_1MONTH if forecast_period == "1MONTH" else FORECAST_HORIZON_3MONTHS
    dataset = dataset.copy()
    dataset[DATE_COL] = pd.to_datetime(dataset[DATE_COL])
    if date_inference is None:
        date_inference = dataset[DATE_COL].max()
    else:
        date_inference = pd.to_datetime(date_inference)
    if date_inference not in dataset[DATE_COL].values:
        logging.error("No hay datos para la fecha de inferencia: %s", date_inference)
        return results, forecasts_df
    target_col = dataset.columns[-1]
    if target_col.endswith(settings.target_suffix):
        real_col = target_col[: -len(settings.target_suffix)]
    else:
        real_col = target_col
    hist_df = dataset[dataset[DATE_COL] < date_inference].sort_values(DATE_COL)
    X_hist = hist_df.drop(columns=[target_col, DATE_COL])
    y_hist = hist_df[target_col]
    inference_row = dataset[dataset[DATE_COL] == date_inference]
    X_inf = inference_row.drop(columns=[target_col, DATE_COL])
    dates_future = pd.date_range(start=date_inference + pd.Timedelta(days=1), periods=horizon, freq="B")
    for name, model in models.items():
        try:
            refined = refine_model_locally(model, X_hist, y_hist)
            pred = refined.predict(X_inf)[0]
        except Exception as exc:  # pragma: no cover - runtime failure
            logging.error("Error al predecir con %s: %s", name, exc)
            continue
        target_date = date_inference + pd.Timedelta(days=int(horizon * 1.4))
        results[name] = {
            "date_inference": date_inference.strftime("%Y-%m-%d"),
            "target_date": target_date.strftime("%Y-%m-%d"),
            "prediction": float(pred),
            "forecast_period": forecast_period,
            "forecast_horizon": horizon,
            "model_type": type(model).__name__,
            "model_name": name,
            "refinement_days": LOCAL_REFINEMENT_DAYS,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        future_preds = [pred]
        current_features = X_inf.values.reshape(1, -1)
        for _ in range(horizon - 1):
            try:
                next_pred = refined.predict(current_features)[0]
            except Exception as exc:  # pragma: no cover - runtime failure
                logging.error("Error en forecast para %s: %s", name, exc)
                next_pred = np.nan
            future_preds.append(next_pred)
        hist_dates = hist_df[DATE_COL].tolist() + [date_inference]
        hist_real = hist_df[real_col].tolist() + [inference_row[real_col].values[0]]
        hist_pred = [np.nan] * len(hist_df) + [pred]
        future_dates = dates_future.tolist()
        future_real = [np.nan] * len(future_dates)
        forecast_df = pd.DataFrame(
            {
                DATE_COL: hist_dates + future_dates,
                "Valor_Real": hist_real + future_real,
                "Valor_Predicho": hist_pred + future_preds,
                "Modelo": name,
                "Periodo": ["Historico"] * len(hist_dates) + ["Forecast"] * len(future_dates),
            }
        )
        forecasts_df[name] = forecast_df
    logging.info("Tiempo total de inferencia: %.2fs", time.perf_counter() - t0)
    return results, forecasts_df


def save_all_inference_results(
    results: Dict[str, Dict[str, Any]],
    forecasts_df: Dict[str, pd.DataFrame],
    *,
    output_dir: str | os.PathLike | None = None,
) -> Dict[str, str]:
    """Persist inference results and forecast visualisations."""
    if pd is None or np is None:
        raise ImportError("pandas and numpy are required for save_all_inference_results")

    t0 = time.perf_counter()
    if not results:
        logging.error("No hay resultados para guardar")
        return {}
    if output_dir is None:
        date_str = list(results.values())[0]["date_inference"].replace("-", "")
        output_dir = Path(RESULTS_DIR) / f"inference_{date_str}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: Dict[str, str] = {}
    consolidated_file = output_dir / "all_models_inference.json"
    with open(consolidated_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    fecha_inf = list(results.values())[0]["date_inference"]
    consolidated_df = pd.DataFrame(
        [
            {
                "Fecha_Inferencia": fecha_inf,
                "Modelo": name,
                "Prediccion": res["prediction"],
                "Fecha_Objetivo": res["target_date"],
                "Horizonte_Dias": res["forecast_horizon"],
                "Tiempo_Ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            for name, res in results.items()
        ]
    )
    consolidated_csv = output_dir / "predictions_api.csv"
    consolidated_df.to_csv(consolidated_csv, index=False)
    api_json = Path(RESULTS_DIR) / "predictions_api.json"
    with open(api_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    for name, res in results.items():
        model_file = output_dir / f"{name}_inference.json"
        with open(model_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=4)
        saved_files[name] = str(model_file)
        if name in forecasts_df:
            chart_path = Path(IMG_CHARTS) / f"{name}_forecast.png"
            df_model = forecasts_df[name]
            df_eval = df_model.dropna(subset=["Valor_Real", "Valor_Predicho"])
            metrics: Dict[str, float] = {}
            if len(df_eval) > 0:
                from sklearn.metrics import mean_squared_error

                rmse = np.sqrt(mean_squared_error(df_eval["Valor_Real"], df_eval["Valor_Predicho"]))
                metrics["RMSE"] = rmse
            from sp500_analysis.shared.visualization.time_series_plots import plot_forecast

            plot_forecast(
                forecasts_df[name],
                inference_date=res["date_inference"],
                title=f"Forecast {name} - Horizonte: {res['forecast_horizon']} días",
                metrics=metrics,
                model_name=name,
                output_path=str(chart_path),
            )
    all_forecasts = pd.concat(forecasts_df.values())
    all_forecasts_csv = output_dir / "all_forecasts.csv"
    all_forecasts.to_csv(all_forecasts_csv, index=False)
    logging.info("Tiempo para guardar resultados: %.2fs", time.perf_counter() - t0)
    return saved_files
