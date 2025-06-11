from __future__ import annotations

"""Feature selection utilities based on permutation importance."""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:  # pragma: no cover - optional dependencies
    import pandas as pd
    import numpy as np
    from catboost import CatBoostRegressor
    from feature_engine.selection import SelectByShuffling
    from sklearn.model_selection import TimeSeriesSplit
except Exception:  # pragma: no cover - optional
    pd = None  # type: ignore
    np = None  # type: ignore

__all__ = [
    "get_most_recent_file",
    "plot_cv_splits",
    "plot_performance_drift",
    "select_features_fpi",
]


def get_most_recent_file(directory: str, extension: str = ".xlsx") -> str | None:
    """Return the most recent file with ``extension`` in ``directory``."""
    files = list(Path(directory).glob(f"*{extension}"))
    if not files:
        return None
    return str(max(files, key=lambda p: p.stat().st_mtime))


def plot_cv_splits(
    X: "pd.DataFrame",
    tscv: TimeSeriesSplit,
    output_path: str | Path,
    *,
    cv_splits: int,
    gap: int,
) -> None:
    """Save a visualisation of the ``tscv`` splits used."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15, 5))
    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        ax.scatter(train_idx, [i + 0.5] * len(train_idx), c="blue", marker="_", s=40, label="Train" if i == 0 else "")
        ax.scatter(val_idx, [i + 0.5] * len(val_idx), c="red", marker="_", s=40, label="Validation" if i == 0 else "")
        ax.text(
            X.shape[0] + 5,
            i + 0.5,
            f"Split {i+1}: {len(train_idx)} train, {len(val_idx)} val",
            va="center",
            ha="left",
        )

    ax.legend(loc="upper right")
    ax.set_xlabel("Índice de muestra")
    ax.set_yticks(range(1, cv_splits + 1))
    ax.set_yticklabels([f"Split {i+1}" for i in range(cv_splits)])
    ax.set_title(f"Validación Cruzada Temporal (CV_SPLITS={cv_splits}, GAP={gap})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info("Gráfico de CV splits guardado en: %s", output_path)


def plot_performance_drift(
    features: Iterable[str],
    drifts: Iterable[float] | dict[str, float],
    selected: Iterable[str],
    threshold: float,
    output_path: str | Path,
) -> None:
    """Save a bar plot of the drift scores for ``features``."""
    import matplotlib.pyplot as plt

    if isinstance(drifts, dict):
        drifts_list = [drifts.get(f, 0) for f in features]
    else:
        drifts_list = list(drifts)

    df = pd.DataFrame({"feature": list(features), "drift": drifts_list})
    df["selected"] = df["feature"].isin(list(selected))
    df = df.sort_values("drift", ascending=False)

    fig, ax = plt.subplots(figsize=(12, max(8, len(features) / 5)))
    colors = ["green" if sel else "red" for sel in df["selected"]]
    bars = ax.barh(df["feature"], df["drift"], color=colors)
    ax.axvline(x=threshold, color="black", linestyle="--", label=f"Threshold ({threshold:.4f})")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.4f}", va="center", ha="left")

    ax.legend()
    ax.set_xlabel("Performance Drift")
    ax.set_ylabel("Feature")
    ax.set_title("Performance Drift por Feature (FPI)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info("Gráfico de performance drift guardado en: %s", output_path)


def select_features_fpi(
    X: "pd.DataFrame",
    y: "pd.Series",
    *,
    cv_splits: int,
    gap: int,
    threshold: float,
    catboost_params: dict | None = None,
    scorer=None,
    plots_dir: str | Path | None = None,
    timestamp: str | None = None,
) -> tuple[list[str], list[float]]:
    """Run permutation importance based feature selection using CatBoost."""
    if pd is None or np is None:  # pragma: no cover - optional deps
        raise ImportError("pandas and numpy are required for select_features_fpi")

    start_time = time.time()
    logging.info("=" * 50)
    logging.info("[FPI] INICIANDO ANÁLISIS DE FEATURE PERMUTATION IMPORTANCE")
    logging.info("=" * 50)
    logging.info("[FPI] Dimensiones de datos - X: %s, y: %s", X.shape, y.shape)
    logging.info(
        "[FPI] Parámetros - CV splits: %s, gap: %s, threshold: %s",
        cv_splits,
        gap,
        threshold,
    )

    tscv = TimeSeriesSplit(n_splits=cv_splits, gap=gap)

    if plots_dir is not None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        cv_plot = plots_dir / f"cv_splits_{ts}.png"
        plot_cv_splits(X, tscv, cv_plot, cv_splits=cv_splits, gap=gap)
    else:
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    params = catboost_params or {"iterations": 100, "verbose": False}
    regressor = CatBoostRegressor(**params)
    selector = SelectByShuffling(estimator=regressor, scoring=scorer, cv=tscv, threshold=threshold)

    selector.fit(X, y)

    selected_features = list(selector.get_feature_names_out())
    performance_drifts = selector.performance_drifts_

    if plots_dir is not None:
        drift_csv = plots_dir / f"fpi_drifts_{ts}.csv"
        pd.DataFrame({"feature": X.columns, "performance_drift": performance_drifts}).to_csv(drift_csv, index=False)
        drift_plot = plots_dir / f"performance_drift_{ts}.png"
        plot_performance_drift(
            list(X.columns),
            performance_drifts,
            selected_features,
            threshold,
            drift_plot,
        )

    total_time = time.time() - start_time
    logging.info("[FPI] Proceso FPI completado en %.2f segundos", total_time)
    logging.info("[FPI] Número de features seleccionadas: %d", len(selected_features))

    return selected_features, performance_drifts
