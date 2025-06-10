"""Common visualization helpers used across the ML pipeline."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter
from matplotlib.figure import Figure

# Global style configuration for consistency
plt.style.use("seaborn-v0_8-whitegrid")

COLORS: Dict[str, str] = {
    "real": "#1f77b4",
    "predicted": "#ff7f0e",
    "training": "#2ca02c",
    "validation": "#d62728",
    "test": "#9467bd",
    "forecast": "#8c564b",
    "catboost": "#e377c2",
    "lightgbm": "#7f7f7f",
    "xgboost": "#bcbd22",
    "mlp": "#17becf",
    "svm": "#1f77b4",
    "ensemble": "#ff7f0e",
}


def create_directory(directory: str) -> None:
    """Create the given directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


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
            df.loc[mask_real, date_col], df.loc[mask_real, real_col], label="Real", color=COLORS["real"], linewidth=2
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

    titulo: List[str] = []
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

    titulo: List[str] = []
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


def plot_feature_importance(
    importances: Sequence[float] | np.ndarray,
    feature_names: Sequence[str],
    *,
    title: str | None = None,
    top_n: int = 20,
    model_name: str | None = None,
    output_path: str | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> Figure:
    """Visualize feature importances."""
    if len(importances) != len(feature_names):
        raise ValueError("Las longitudes de importances y feature_names deben coincidir")

    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(top_indices)), np.asarray(importances)[top_indices], align="center")
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([feature_names[i] for i in top_indices])

    if title:
        plt.title(title)
    else:
        titulo = "Importancia de Features"
        if model_name:
            titulo += f" - {model_name}"
        plt.title(titulo)

    plt.xlabel("Importancia")
    plt.tight_layout()

    if output_path:
        create_directory(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig


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

    time_labels: List[str] = []
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


def plot_correlation_matrix(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
    title: str | None = None,
    target_col: str | None = None,
    threshold: float = 0.7,
    figsize: tuple[int, int] = (14, 12),
    output_path: str | None = None,
) -> Figure:
    """Plot a correlation matrix heatmap."""
    corr_matrix = df.corr(method=method)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax,
    )

    if target_col and target_col in corr_matrix.columns:
        target_corrs = corr_matrix[target_col].abs().sort_values(ascending=False)
        high_corrs = target_corrs[target_corrs > threshold].index.tolist()
        high_corrs_text = "\n".join(
            [f"{col}: {corr_matrix.loc[col, target_col]:.3f}" for col in high_corrs if col != target_col]
        )
        if high_corrs_text:
            plt.figtext(
                0.01,
                0.01,
                f"Correlaciones altas con {target_col}:\n{high_corrs_text}",
                fontsize=10,
                ha="left",
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
            )

    plt.title(title or f"Matriz de Correlación ({method})")
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
) -> List[str]:
    """Generate a set of common report figures."""
    create_directory(output_dir)
    generated_figures: List[str] = []

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
