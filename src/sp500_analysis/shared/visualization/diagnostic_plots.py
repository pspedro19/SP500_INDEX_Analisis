"""Diagnostics and metric visualization utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from .forecast_plots import COLORS, DEFAULT_DPI, configure_axis_date

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")

logger = logging.getLogger(__name__)
DEFAULT_DIAGNOSTIC_FIGSIZE = (16, 10)


def plot_model_diagnostics(
    model_fit: Any,
    residuals: pd.Series,
    title: str = "Diagnóstico de Modelo",
    figsize: Tuple[int, int] = DEFAULT_DIAGNOSTIC_FIGSIZE,
    save_path: Optional[Path] = None,
    instrument: str = "",
    model_name: str = "",
) -> plt.Figure:
    """Generate diagnostic charts for a time series model."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    full_title = f"{title} - {instrument} - {model_name}" if model_name and instrument else title
    fig.suptitle(full_title, fontsize=16)

    axes[0, 0].plot(residuals.index, residuals, color="#2980B9")
    axes[0, 0].axhline(y=0, color="#E74C3C", linestyle="--")
    axes[0, 0].set_title("Residuos vs Tiempo")
    configure_axis_date(axes[0, 0])

    sns.histplot(residuals, kde=True, ax=axes[0, 1], color="#2980B9")
    axes[0, 1].set_title("Distribución de Residuos")

    plot_acf(residuals.values, ax=axes[1, 0], lags=min(30, len(residuals) // 4), alpha=0.05)
    axes[1, 0].set_title("ACF de Residuos")

    plot_pacf(residuals.values, ax=axes[1, 1], lags=min(30, len(residuals) // 4), alpha=0.05)
    axes[1, 1].set_title("PACF de Residuos")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_name: str = "RMSE",
    title: str = "Comparación de Modelos",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    instrument: str = "",
    is_lower_better: bool = True,
) -> plt.Figure:
    """Bar comparison of a single metric across models."""
    fig, ax = plt.subplots(figsize=figsize)
    full_title = f"{title} - {metric_name} - {instrument}" if instrument else f"{title} - {metric_name}"

    models = []
    values = []
    for model_name, model_metrics in metrics.items():
        if metric_name in model_metrics:
            models.append(model_name)
            values.append(model_metrics[metric_name])
    if not models:
        plt.close(fig)
        return None

    sorted_indices = np.argsort(values)
    if not is_lower_better:
        sorted_indices = sorted_indices[::-1]

    sorted_models = [models[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    colors = ["#9B59B6" if model == "ENSEMBLE" else "#3498DB" for model in sorted_models]
    bars = ax.bar(sorted_models, sorted_values, color=colors)

    ax.set_title(full_title, fontsize=14)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_xlabel("Modelo", fontsize=12)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01 * max(sorted_values), f"{height:.4f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_metrics_radar(
    metrics: Dict[str, Dict[str, float]],
    metric_names: List[str],
    title: str = "Métricas por Modelo",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
    instrument: str = "",
    is_lower_better: Dict[str, bool] | None = None,
) -> plt.Figure:
    """Radar chart comparing multiple metrics."""
    if is_lower_better is None:
        is_lower_better = {
            "RMSE": True,
            "MAE": True,
            "MAPE": True,
            "MSE": True,
            "R2": False,
            "R2_adjusted": False,
            "Hit_Direction": False,
            "Amplitude_Score": False,
            "Phase_Score": False,
            "Weighted_Hit_Direction": False,
        }

    available_metrics = [m for m in metric_names if all(m in mm for mm in metrics.values())]
    if not available_metrics:
        return None

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)
    full_title = f"{title} - {instrument}" if instrument else title
    N = len(available_metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    normalized_metrics: Dict[str, List[float]] = {}
    for model_name, model_metrics in metrics.items():
        normalized_metrics[model_name] = []
        for i, metric in enumerate(available_metrics):
            value = model_metrics.get(metric, 0)
            if i not in normalized_metrics:
                normalized_metrics[i] = []
            normalized_metrics[model_name].append(value)

    for i, metric in enumerate(available_metrics):
        values = [metrics[model][metric] for model in metrics if metric in metrics[model]]
        if not values:
            continue
        min_val = min(values)
        max_val = max(values)
        if min_val == max_val:
            for model_name in normalized_metrics:
                if isinstance(model_name, str):
                    normalized_metrics[model_name][i] = 1.0
        else:
            for model_name in normalized_metrics:
                if isinstance(model_name, str):
                    value = normalized_metrics[model_name][i]
                    if metric in is_lower_better and is_lower_better[metric]:
                        normalized_val = 1 - ((value - min_val) / (max_val - min_val))
                    else:
                        normalized_val = (value - min_val) / (max_val - min_val)
                    normalized_metrics[model_name][i] = normalized_val

    for model_name in metrics:
        if model_name not in normalized_metrics:
            continue
        values = normalized_metrics[model_name]
        values += values[:1]
        color = "#9B59B6" if model_name == "ENSEMBLE" else None
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=model_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_metrics)
    plt.title(full_title, size=14, y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_metric_evolution(
    metric_values: pd.DataFrame,
    metric_name: str,
    title: str = "Evolución de Métricas",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    instrument: str = "",
    model_name: str = "",
) -> plt.Figure:
    """Plot metric value evolution over time."""
    fig, ax = plt.subplots(figsize=figsize)
    full_title = f"{title} - {metric_name} - {instrument} - {model_name}" if model_name and instrument else f"{title} - {metric_name}"
    ax.plot(metric_values.index, metric_values[metric_name], marker="o", linewidth=2, color="#3498DB")

    if len(metric_values) > 3:
        window = min(5, len(metric_values) // 2)
        if window > 0:
            rolling_mean = metric_values[metric_name].rolling(window=window).mean()
            ax.plot(metric_values.index, rolling_mean, color="#E74C3C", linewidth=2, linestyle="--", label=f"Media Móvil ({window} períodos)")

    ax.set_title(full_title, fontsize=14)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    if len(metric_values) > 0:
        last_value = metric_values[metric_name].iloc[-1]
        ax.annotate(f"{last_value:.4f}", xy=(metric_values.index[-1], last_value), xytext=(10, 0), textcoords="offset points", fontsize=10, fontweight="bold")
    ax.legend(loc="best")
    configure_axis_date(ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig
