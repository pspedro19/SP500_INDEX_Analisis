"""Visualization helpers for time series related charts."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from matplotlib.figure import Figure

from .common import COLORS, create_directory


def plot_real_vs_pred(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    real_col: str = "Valor_Real",
    pred_col: str = "Valor_Predicho",
    title: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    periodo: str | None = None,
    model_name: str | None = None,
    output_path: str | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """Plot real vs predicted values."""
    fig, ax = plt.subplots(figsize=figsize)

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)

    mask_real = ~df[real_col].isna()
    mask_pred = ~df[pred_col].isna()

    if mask_real.any():
        ax.plot(
            df.loc[mask_real, date_col],
            df.loc[mask_real, real_col],
            label="Real",
            color=COLORS["real"],
            linewidth=2,
        )
    if mask_pred.any():
        ax.plot(
            df.loc[mask_pred, date_col],
            df.loc[mask_pred, pred_col],
            label="Predicho",
            color=COLORS["predicted"],
            linewidth=2,
            linestyle="--",
            marker="o",
            markersize=4,
        )

    date_formatter = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)
    plt.xlabel("Fecha")
    plt.ylabel("Valor")

    titulo: list[str] = []
    if model_name:
        titulo.append(f"Modelo: {model_name}")
    if periodo:
        titulo.append(f"Período: {periodo}")
    if title:
        titulo.append(title)
    plt.title(" | ".join(titulo))

    if metrics is not None:
        metrics_text = [
            f"{name}: {value:.4f}" if isinstance(value, (int, float)) else f"{name}: {value}"
            for name, value in metrics.items()
        ]
        metrics_str = " | ".join(metrics_text)
        plt.figtext(
            0.5,
            0.01,
            metrics_str,
            ha="center",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
        )

    plt.legend(loc="best")
    plt.tight_layout()

    if output_path:
        create_directory(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig


def plot_forecast(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    real_col: str = "Valor_Real",
    pred_col: str = "Valor_Predicho",
    inference_date: str | None = None,
    cutoff_date: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    title: str | None = None,
    model_name: str | None = None,
    output_path: str | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """Plot forecast results separating historical and forecast values."""
    fig, ax = plt.subplots(figsize=figsize)

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)

    if inference_date:
        inference_dt = pd.to_datetime(inference_date)
        hist_data = df[df[date_col] <= inference_dt]
        forecast_data = df[df[date_col] > inference_dt]
    else:
        hist_data = df.dropna(subset=[real_col])
        forecast_data = df[df[real_col].isna()]

    if not hist_data.empty:
        ax.plot(hist_data[date_col], hist_data[real_col], label="Histórico (real)", color=COLORS["real"], linewidth=2)
    if not hist_data.empty and pred_col in hist_data.columns:
        ax.plot(
            hist_data[date_col],
            hist_data[pred_col],
            label="Predicho (histórico)",
            color=COLORS["validation"],
            linewidth=1.5,
            linestyle="--",
        )
    if not forecast_data.empty:
        ax.plot(
            forecast_data[date_col],
            forecast_data[pred_col],
            label="Forecast",
            color=COLORS["forecast"],
            linewidth=2.5,
            linestyle="-",
            marker="o",
            markersize=5,
        )

    cutoff = cutoff_date or inference_date
    if cutoff:
        cutoff_dt = pd.to_datetime(cutoff)
        ax.axvline(
            x=cutoff_dt,
            color="black",
            linestyle="--",
            alpha=0.7,
            label=f"Fecha de corte: {cutoff_dt.strftime('%Y-%m-%d')}",
        )

    date_formatter = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)
    plt.xlabel("Fecha")
    plt.ylabel("Valor")

    titulo: list[str] = []
    if model_name:
        titulo.append(f"Modelo: {model_name}")
    if title:
        titulo.append(title)
    else:
        titulo.append("Inferencia y Forecast")
    plt.title(" | ".join(titulo))

    if metrics is not None:
        metrics_text = [
            f"{name}: {value:.4f}" if isinstance(value, (int, float)) else f"{name}: {value}"
            for name, value in metrics.items()
        ]
        metrics_str = " | ".join(metrics_text)
        plt.figtext(
            0.5,
            0.01,
            metrics_str,
            ha="center",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
        )

    plt.legend(loc="best")
    plt.tight_layout()

    if output_path:
        create_directory(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig


def plot_training_curves(
    df: pd.DataFrame,
    *,
    model_names: Sequence[str] | None = None,
    metric_col: str = "RMSE",
    trial_col: str = "Trial",
    title: str | None = None,
    output_path: str | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """Plot training metric evolution for multiple models."""
    fig, ax = plt.subplots(figsize=figsize)

    df_plot = df[df["Modelo"].isin(model_names)] if model_names else df.copy()

    if "Modelo" in df_plot.columns and trial_col in df_plot.columns:
        df_pivot = df_plot.pivot(index=trial_col, columns="Modelo", values=metric_col)
        df_pivot.plot(ax=ax, marker="o")
    else:
        for model in df_plot["Modelo"].unique():
            model_data = df_plot[df_plot["Modelo"] == model]
            ax.plot(model_data[trial_col], model_data[metric_col], label=model, marker="o")

    plt.xlabel("Trial")
    plt.ylabel(metric_col)
    plt.title(title or f"Evolución de {metric_col} por Trial")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()

    if output_path:
        create_directory(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig


def plot_pipeline_timeline(
    times_dict: Mapping[str, float],
    output_path: str | None = None,
    figsize: tuple[int, int] = (14, 8),
) -> Figure:
    """Create a simple Gantt-like timeline for pipeline execution."""
    stages = sorted(times_dict.keys())
    times = [times_dict[stage] for stage in stages]

    cumulative_times = np.zeros(len(times))
    for i in range(1, len(times)):
        cumulative_times[i] = cumulative_times[i - 1] + times[i - 1]

    time_labels: list[str] = []
    for t in times:
        if t < 60:
            time_labels.append(f"{t:.1f}s")
        elif t < 3600:
            time_labels.append(f"{t/60:.1f}m")
        else:
            time_labels.append(f"{t/3600:.1f}h")

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(stages, times, left=cumulative_times, height=0.5)
    for bar, label in zip(bars, time_labels):
        width = bar.get_width()
        if width > 0:
            x = bar.get_x() + width / 2
            y = bar.get_y() + bar.get_height() / 2
            ax.text(x, y, label, ha="center", va="center", color="white", fontweight="bold")

    total_time = sum(times)
    total_label = f"{total_time:.1f}s"
    if total_time >= 60:
        minutes = total_time / 60
        total_label = f"{minutes:.1f}m"
        if minutes >= 60:
            total_label = f"{minutes/60:.1f}h {int(minutes%60)}m"
    plt.xlabel(f"Tiempo (Total: {total_label})")
    plt.title("Tiempos de ejecución del pipeline")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if output_path:
        create_directory(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig
