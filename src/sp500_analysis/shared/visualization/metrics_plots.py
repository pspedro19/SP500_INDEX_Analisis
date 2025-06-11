"""Visualization helpers for metric comparison charts."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .common import COLORS, create_directory
from .time_series_plots import plot_real_vs_pred


def plot_radar_metrics(
    metrics_dict: Mapping[str, Mapping[str, float]],
    *,
    model_names: Iterable[str] | None = None,
    title: str | None = None,
    output_path: str | None = None,
    figsize: tuple[int, int] = (10, 10),
) -> Figure:
    """Generate a radar chart comparing metrics for each model."""
    if model_names:
        metrics_dict = {k: v for k, v in metrics_dict.items() if k in model_names}

    all_metrics = sorted({m for metrics in metrics_dict.values() for m in metrics})
    angles = np.linspace(0, 2 * np.pi, len(all_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    normalized_metrics: Dict[str, Dict[str, float]] = {}
    for metric in all_metrics:
        values = [metrics_dict[model].get(metric, np.nan) for model in metrics_dict]
        values = [v for v in values if not np.isnan(v)]
        if not values:
            continue
        better_high = any(t in metric.lower() for t in ["r2", "accuracy", "precision", "recall", "score"])
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            normalized_metrics[metric] = {model: 1.0 for model in metrics_dict}
        elif better_high:
            normalized_metrics[metric] = {
                model: (metrics_dict[model].get(metric, min_val) - min_val) / (max_val - min_val)
                for model in metrics_dict
            }
        else:
            normalized_metrics[metric] = {
                model: 1 - (metrics_dict[model].get(metric, max_val) - min_val) / (max_val - min_val)
                for model in metrics_dict
            }

    for i, (model, metrics) in enumerate(metrics_dict.items()):
        values = [normalized_metrics.get(metric, {}).get(model, 0) for metric in all_metrics]
        values += values[:1]
        color = COLORS.get(model.lower(), f"C{i}")
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_metrics)
    plt.title(title or "Comparación de Métricas")
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()

    if output_path:
        create_directory(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig


def plot_metrics_by_subperiod(
    metrics_by_period: pd.DataFrame,
    *,
    metric_col: str = "RMSE",
    model_col: str = "Modelo",
    period_col: str = "Periodo",
    title: str | None = None,
    output_path: str | None = None,
    figsize: tuple[int, int] = (14, 8),
) -> Figure:
    """Plot metrics grouped by subperiods."""
    fig, ax = plt.subplots(figsize=figsize)

    if model_col in metrics_by_period.columns and period_col in metrics_by_period.columns:
        df_pivot = metrics_by_period.pivot(index=period_col, columns=model_col, values=metric_col)
        df_pivot.plot(kind="bar", ax=ax)
    else:
        import seaborn as sns

        sns.barplot(x=period_col, y=metric_col, hue=model_col, data=metrics_by_period, ax=ax)

    plt.xlabel("Período")
    plt.ylabel(metric_col)
    plt.title(title or f"{metric_col} por Período")
    plt.legend(title="Modelo")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if output_path:
        create_directory(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig


def generate_report_figures(
    results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: str,
    *,
    model_names: Sequence[str] | None = None,
    prefix: str = "",
    include_subplots: bool = True,
) -> list[str]:
    """Generate a set of common report figures."""
    create_directory(output_dir)
    generated_figures: list[str] = []

    if model_names:
        results_filtered = results_df[results_df["Modelo"].isin(model_names)]
        metrics_filtered = metrics_df[metrics_df["Modelo"].isin(model_names)]
    else:
        results_filtered = results_df
        metrics_filtered = metrics_df
        model_names = list(results_df["Modelo"].unique())

    if len(model_names) > 1:
        all_models_fig_path = os.path.join(output_dir, f"{prefix}all_models_comparison.png")
        plot_real_vs_pred(results_filtered, title="Comparación de todos los modelos", output_path=all_models_fig_path)
        generated_figures.append(all_models_fig_path)

    if include_subplots:
        for model in model_names:
            model_results = results_filtered[results_filtered["Modelo"] == model]
            if model_results.empty:
                continue
            model_metrics: Dict[str, Any] = {}
            if not metrics_filtered.empty:
                model_metrics_row = metrics_filtered[metrics_filtered["Modelo"] == model]
                if not model_metrics_row.empty:
                    model_metrics = model_metrics_row.iloc[0].to_dict()
            model_fig_path = os.path.join(output_dir, f"{prefix}{model.replace(' ', '_')}_comparison.png")
            plot_real_vs_pred(
                model_results,
                title=f"Modelo: {model}",
                metrics=model_metrics,
                model_name=model,
                output_path=model_fig_path,
            )
            generated_figures.append(model_fig_path)

    if len(model_names) > 1 and not metrics_filtered.empty:
        metrics_dict: Dict[str, Dict[str, float]] = {}
        for _, row in metrics_filtered.iterrows():
            model = row["Modelo"]
            metrics = {col: row[col] for col in metrics_filtered.columns if col != "Modelo" and not pd.isna(row[col])}
            metrics_dict[model] = metrics
        radar_path = os.path.join(output_dir, f"{prefix}radar_comparison.png")
        plot_radar_metrics(metrics_dict, title="Comparación de métricas", output_path=radar_path)
        generated_figures.append(radar_path)

    metrics_cols = [col for col in metrics_filtered.columns if col not in ["Modelo", "Tipo_de_Mercado"]]
    for metric in ["RMSE", "MAE", "SMAPE", "R2"]:
        if metric in metrics_cols:
            metric_path = os.path.join(output_dir, f"{prefix}{metric.lower()}_comparison.png")
            plt.figure(figsize=(10, 6))
            metrics_plot = metrics_filtered.sort_values(metric)
            plt.bar(metrics_plot["Modelo"], metrics_plot[metric])
            plt.title(f"Comparación de {metric} entre modelos")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(metric_path, dpi=300)
            plt.close()
            generated_figures.append(metric_path)

    return generated_figures
