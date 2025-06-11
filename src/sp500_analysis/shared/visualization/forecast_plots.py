"""Forecast and comparison plotting utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Basic style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")

logger = logging.getLogger(__name__)

DEFAULT_FIGSIZE = (12, 6)
DEFAULT_FORECAST_FIGSIZE = (14, 8)
DEFAULT_DPI = 300

COLORS = {
    "real": "#2C3E50",
    "prediction": "#E74C3C",
    "forecast": "#3498DB",
    "ci_upper": "#AED6F1",
    "ci_lower": "#AED6F1",
    "ensemble": "#9B59B6",
    "ensemble_ci": "#D2B4DE",
    "outlier": "#FF5733",
}


def configure_axis_date(ax, date_format: str = "%Y-%m-%d"):
    """Format x-axis ticks as dates."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    fig = ax.figure
    fig.autofmt_xdate()
    return ax


def plot_series_comparison(
    real_series: pd.Series,
    pred_series: pd.Series,
    title: str = "Real vs Predicción",
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    save_path: Optional[Path] = None,
    plot_diff: bool = False,
    model_name: str = "",
    instrument: str = "",
    inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> plt.Figure:
    """Plot real vs predicted series."""
    if inverse_transform:
        real_plot = pd.Series(inverse_transform(real_series.values), index=real_series.index)
        pred_plot = pd.Series(inverse_transform(pred_series.values), index=pred_series.index)
        ylabel = "Precio"
    else:
        real_plot = real_series
        pred_plot = pred_series
        ylabel = "Valor"

    if plot_diff:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=figsize)

    full_title = f"{title} - {instrument} - {model_name}" if model_name and instrument else title

    ax1.plot(real_plot.index, real_plot, label="Real", color=COLORS["real"], linewidth=2)
    ax1.plot(pred_plot.index, pred_plot, label="Predicción", color=COLORS["prediction"], linewidth=2, linestyle="--")
    ax1.set_title(full_title, fontsize=14)
    ax1.legend(loc="best")
    ax1.set_ylabel(ylabel, fontsize=12)
    configure_axis_date(ax1)

    if plot_diff:
        aligned_real = real_plot.reindex(pred_plot.index)
        diff = aligned_real - pred_plot
        ax2.plot(diff.index, diff, color="#27AE60", label="Diferencia")
        ax2.axhline(y=0, color="#7F8C8D", linestyle="-", alpha=0.3)
        ax2.set_ylabel("Diferencia")
        ax2.legend(loc="best")
        configure_axis_date(ax2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_forecast(
    historical: pd.Series,
    forecast: pd.Series,
    pred_series: Optional[pd.Series] = None,
    ci_lower: Optional[pd.Series] = None,
    ci_upper: Optional[pd.Series] = None,
    title: str = "Pronóstico",
    figsize: Tuple[int, int] = DEFAULT_FORECAST_FIGSIZE,
    save_path: Optional[Path] = None,
    instrument: str = "",
    model_name: str = "",
    forecast_periods: int = 21,
    train_end_date: Optional[pd.Timestamp] = None,
    val_end_date: Optional[pd.Timestamp] = None,
    inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> plt.Figure:
    """Plot forecast with optional confidence intervals."""
    if inverse_transform:
        hist_plot = pd.Series(inverse_transform(historical.values), index=historical.index)
        fc_plot = pd.Series(inverse_transform(forecast.values), index=forecast.index)
        ci_low = pd.Series(inverse_transform(ci_lower.values), index=ci_lower.index) if ci_lower is not None else None
        ci_up = pd.Series(inverse_transform(ci_upper.values), index=ci_upper.index) if ci_upper is not None else None
        ylabel = "Precio"
    else:
        hist_plot, fc_plot, ci_low, ci_up = historical, forecast, ci_lower, ci_upper
        ylabel = "Valor"

    fig, ax = plt.subplots(figsize=figsize)
    full_title = f"{title} - {instrument} - {model_name}" if model_name and instrument else title

    history_points = min(len(hist_plot), forecast_periods * 3)
    hist_subset = hist_plot.iloc[-history_points:]
    ax.plot(hist_subset.index, hist_subset, label="Histórico", color=COLORS["real"], linewidth=2)

    if pred_series is not None:
        ps = pd.Series(inverse_transform(pred_series.values), index=pred_series.index) if inverse_transform else pred_series
        ax.plot(ps.index, ps, label="Predicción (Val/Test)", color=COLORS["prediction"], linestyle="--", linewidth=2)

    if train_end_date is not None:
        ax.axvline(x=train_end_date, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.annotate("FIN TRAIN", xy=(train_end_date, ax.get_ylim()[0]), xytext=(5, 15), textcoords="offset points", color="gray", fontweight="bold", fontsize=10)
    if val_end_date is not None:
        ax.axvline(x=val_end_date, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.annotate("FIN VALIDACIÓN", xy=(val_end_date, ax.get_ylim()[0]), xytext=(5, 15), textcoords="offset points", color="black", fontweight="bold", fontsize=10)

    if len(fc_plot) == 0:
        logger.warning("El pronóstico está vacío, no hay datos para graficar")
        ax.text(0.5, 0.5, "SIN DATOS DE PRONÓSTICO", ha="center", va="center", transform=ax.transAxes, fontsize=14, color="red")
    else:
        ax.plot(fc_plot.index, fc_plot, label=f"Pronóstico ({len(fc_plot)} días)", color=COLORS["forecast"], linewidth=3.0, marker="o", markersize=4)
        if len(fc_plot) >= 2:
            forecast_mean = fc_plot.mean()
            forecast_std = fc_plot.std()
            limits = (forecast_mean - 3 * forecast_std, forecast_mean + 3 * forecast_std)
            outliers = fc_plot[(fc_plot < limits[0]) | (fc_plot > limits[1])]
            if not outliers.empty:
                ax.scatter(outliers.index, outliers, color=COLORS["outlier"], s=80, zorder=5, label="Valores atípicos")

    last_historical_date = hist_plot.index[-1]
    ax.axvline(x=last_historical_date, color="#E74C3C", linestyle="--", linewidth=2, alpha=0.8)
    ax.annotate(
        "INICIO PRONÓSTICO",
        xy=(last_historical_date, ax.get_ylim()[0]),
        xytext=(10, 30),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#E74C3C"),
        color="#E74C3C",
        fontweight="bold",
        fontsize=12,
    )

    if train_end_date is not None:
        ax.axvline(x=train_end_date, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.annotate("FIN TRAIN", xy=(train_end_date, ax.get_ylim()[0]), xytext=(5, 15), textcoords="offset points", color="gray", fontweight="bold", fontsize=10)
    if val_end_date is not None:
        ax.axvline(x=val_end_date, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.annotate("FIN VALIDACIÓN", xy=(val_end_date, ax.get_ylim()[0]), xytext=(5, 15), textcoords="offset points", color="black", fontweight="bold", fontsize=10)

    if ci_low is not None and ci_up is not None:
        ax.fill_between(fc_plot.index, ci_low, ci_up, color=COLORS["ci_lower"], alpha=0.3, label="95% CI")

    ax.set_title(full_title, fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    configure_axis_date(ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_ensemble_comparison(
    historical: pd.Series,
    forecasts: Dict[str, pd.Series],
    ensemble_forecast: pd.Series,
    ci_lower: Optional[pd.Series] = None,
    ci_upper: Optional[pd.Series] = None,
    title: str = "Comparación de Modelos y Ensemble",
    figsize: Tuple[int, int] = DEFAULT_FORECAST_FIGSIZE,
    save_path: Optional[Path] = None,
    instrument: str = "",
    inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> plt.Figure:
    """Compare individual forecasts against an ensemble."""
    if inverse_transform:
        hist_plot = pd.Series(inverse_transform(historical.values), index=historical.index)
        ens_plot = pd.Series(inverse_transform(ensemble_forecast.values), index=ensemble_forecast.index)
        fc_plots = {model: pd.Series(inverse_transform(fc.values), index=fc.index) for model, fc in forecasts.items()}
        ci_low = pd.Series(inverse_transform(ci_lower.values), index=ci_lower.index) if ci_lower is not None else None
        ci_up = pd.Series(inverse_transform(ci_upper.values), index=ci_upper.index) if ci_upper is not None else None
        ylabel = "Precio"
    else:
        hist_plot = historical
        ens_plot = ensemble_forecast
        fc_plots = forecasts
        ci_low = ci_lower
        ci_up = ci_upper
        ylabel = "Valor"

    fig, ax = plt.subplots(figsize=figsize)
    full_title = f"{title} - {instrument}" if instrument else title

    forecast_periods = len(ens_plot)
    history_points = min(len(hist_plot), forecast_periods * 2)
    hist_subset = hist_plot.iloc[-history_points:]
    ax.plot(hist_subset.index, hist_subset, label="Histórico", color=COLORS["real"], linewidth=2.5)

    for model_name, forecast in fc_plots.items():
        ax.plot(forecast.index, forecast, label=model_name, alpha=0.7, linewidth=1, linestyle="--")

    ax.plot(ens_plot.index, ens_plot, label="ENSEMBLE", color=COLORS["ensemble"], linewidth=2.5)

    if ci_low is not None and ci_up is not None:
        ax.fill_between(ens_plot.index, ci_low, ci_up, color=COLORS["ensemble_ci"], alpha=0.2, label="95% CI (ENSEMBLE)")

    last_historical_date = hist_plot.index[-1]
    ax.axvline(x=last_historical_date, color="#95A5A6", linestyle="--", alpha=0.5)

    ax.set_title(full_title, fontsize=14)
    ax.legend(loc="best")
    ax.set_ylabel(ylabel, fontsize=12)
    configure_axis_date(ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig


def create_dashboard(
    real_series: pd.Series,
    pred_series: pd.Series,
    forecast_series: pd.Series,
    metrics: Dict[str, float],
    title: str = "Dashboard de Modelo",
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[Path] = None,
    instrument: str = "",
    model_name: str = "",
    ci_lower: Optional[pd.Series] = None,
    ci_upper: Optional[pd.Series] = None,
    train_end_date: Optional[pd.Timestamp] = None,
    val_end_date: Optional[pd.Timestamp] = None,
    inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> plt.Figure:
    """Create a dashboard with historical series, forecast and metrics."""
    if inverse_transform:
        real_plot = pd.Series(inverse_transform(real_series.values), index=real_series.index)
        pred_plot = pd.Series(inverse_transform(pred_series.values), index=pred_series.index)
        fc_plot = pd.Series(inverse_transform(forecast_series.values), index=forecast_series.index)
        ci_low = pd.Series(inverse_transform(ci_lower.values), index=ci_lower.index) if ci_lower is not None else None
        ci_up = pd.Series(inverse_transform(ci_upper.values), index=ci_upper.index) if ci_upper is not None else None
        ylabel = "Precio"
    else:
        real_plot = real_series
        pred_plot = pred_series
        fc_plot = forecast_series
        ci_low = ci_lower
        ci_up = ci_upper
        ylabel = "Valor"

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3)
    full_title = f"{title} - {instrument} - {model_name}" if model_name and instrument else title
    fig.suptitle(full_title, fontsize=16, y=0.98)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(real_plot.index, real_plot, label="Real", color=COLORS["real"], linewidth=2)
    ax1.plot(pred_plot.index, pred_plot, label="Predicción", color=COLORS["prediction"], linewidth=2, linestyle="--")
    if train_end_date is not None:
        ax1.axvline(x=train_end_date, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax1.annotate("FIN TRAIN", xy=(train_end_date, ax1.get_ylim()[0]), xytext=(5, 15), textcoords="offset points", color="gray", fontweight="bold", fontsize=10)
    if val_end_date is not None:
        ax1.axvline(x=val_end_date, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        ax1.annotate("FIN VALIDACIÓN", xy=(val_end_date, ax1.get_ylim()[0]), xytext=(5, 15), textcoords="offset points", color="black", fontweight="bold", fontsize=10)
    ax1.set_title("Serie Histórica vs Predicción", fontsize=12)
    ax1.legend(loc="best")
    ax1.set_ylabel(ylabel, fontsize=12)
    configure_axis_date(ax1)

    ax2 = fig.add_subplot(gs[1, :])
    history_points = min(len(real_plot), len(fc_plot) * 2)
    hist_subset = real_plot.iloc[-history_points:]
    ax2.plot(hist_subset.index, hist_subset, label="Histórico", color=COLORS["real"], linewidth=2)
    if len(fc_plot) == 0:
        logger.warning("El pronóstico está vacío, no hay datos para graficar en el dashboard")
        ax2.text(0.5, 0.5, "SIN DATOS DE PRONÓSTICO", ha="center", va="center", transform=ax2.transAxes, fontsize=14, color="red")
    else:
        ax2.plot(fc_plot.index, fc_plot, label=f"Pronóstico ({len(fc_plot)} días)", color=COLORS["forecast"], linewidth=3.0, marker="o", markersize=4)
        if ci_low is not None and ci_up is not None:
            ax2.fill_between(fc_plot.index, ci_low, ci_up, color=COLORS["ci_lower"], alpha=0.3, label="95% CI")
        last_historical_date = real_plot.index[-1]
        ax2.axvline(x=last_historical_date, color="#E74C3C", linestyle="--", linewidth=2, alpha=0.8)
        ax2.annotate(
            "INICIO PRONÓSTICO",
            xy=(last_historical_date, ax2.get_ylim()[0]),
            xytext=(10, 30),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#E74C3C"),
            color="#E74C3C",
            fontweight="bold",
            fontsize=12,
        )
    if train_end_date is not None:
        ax2.axvline(x=train_end_date, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax2.annotate("FIN TRAIN", xy=(train_end_date, ax2.get_ylim()[0]), xytext=(5, 15), textcoords="offset points", color="gray", fontweight="bold", fontsize=10)
    if val_end_date is not None:
        ax2.axvline(x=val_end_date, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        ax2.annotate("FIN VALIDACIÓN", xy=(val_end_date, ax2.get_ylim()[0]), xytext=(5, 15), textcoords="offset points", color="black", fontweight="bold", fontsize=10)
    ax2.set_title("Pronóstico", fontsize=12)
    ax2.legend(loc="best")
    ax2.set_ylabel(ylabel, fontsize=12)
    configure_axis_date(ax2)

    ax3 = fig.add_subplot(gs[2, 0:2])
    key_metrics = ["RMSE", "MAE", "Hit_Direction", "R2"]
    available_metrics = [m for m in key_metrics if m in metrics]
    if available_metrics:
        metric_values = [metrics[m] for m in available_metrics]
        bars = ax3.bar(available_metrics, metric_values, color="#3498DB")
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01 * max(metric_values), f"{height:.4f}", ha="center", va="bottom", fontsize=9)
        ax3.set_title("Métricas Clave", fontsize=12)

    ax4 = fig.add_subplot(gs[2, 2])
    ax4.axis("off")
    if metrics:
        metrics_text = "MÉTRICAS:\n\n" + "\n".join(f"{m}: {v:.4f}" for m, v in metrics.items())
        ax4.text(0, 0.9, metrics_text, fontsize=10, verticalalignment="top")

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)

    return fig


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists and return it as Path."""
    if not isinstance(path, Path):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_model_visualizations(
    model: Any,
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    exog_cols: Optional[list[str]],
    test_pred: np.ndarray,
    forecast_df: pd.DataFrame,
    metrics: Dict[str, Any],
    model_type: str,
    instrument: str,
    output_dir: Path,
    df_train: Optional[pd.DataFrame] = None,
    df_val: Optional[pd.DataFrame] = None,
) -> Dict[str, Path]:
    """Generate and save basic visualizations for a fitted model."""
    if "plots" not in globals():
        return {}

    logger = get_logger(__name__, instrument=instrument, model_type=model_type)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True, parents=True)
    paths: Dict[str, Path] = {}

    try:
        y_train = df[target_col]
        y_test = df_test[target_col]
        test_pred_series = pd.Series(test_pred, index=df_test.index)

        train_end_date = df_train.index[-1] if df_train is not None and len(df_train) > 0 else None
        if df_val is not None and len(df_val) > 0:
            val_end_date = df_val.index[-1]
        elif len(df_test) > 0:
            val_end_date = df_test.index[0] - pd.Timedelta(days=1)
        else:
            val_end_date = None

        comparison_path = charts_dir / f"{instrument}_{model_type}_comparison.png"
        plots.plot_series_comparison(
            real_series=y_test,
            pred_series=test_pred_series,
            title="Real vs Predicción",
            save_path=comparison_path,
            plot_diff=True,
            model_name=model_type,
            instrument=instrument,
        )
        paths["comparison"] = comparison_path
        logger.info(f"Gráfico de comparación guardado en {comparison_path}")

        forecast_series = forecast_df["forecast"]
        ci_lower = forecast_df["lower_95"] if "lower_95" in forecast_df else None
        ci_upper = forecast_df["upper_95"] if "upper_95" in forecast_df else None

        forecast_path = charts_dir / f"{instrument}_{model_type}_forecast.png"
        try:
            plots.plot_forecast(
                historical=y_train,
                forecast=forecast_series,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                title="Pronóstico",
                save_path=forecast_path,
                instrument=instrument,
                model_name=model_type,
            )
        except Exception as e:
            logger.warning(f"Error en primera llamada a plot_forecast: {str(e)}. Intentando alternativa.")
            try:
                plots.plot_forecast(
                    historical=y_train,
                    forecast=forecast_series,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    title="Pronóstico",
                    save_path=forecast_path,
                    instrument=instrument,
                    model_name=model_type,
                    train_end_date=train_end_date,
                    val_end_date=val_end_date,
                )
            except Exception as e2:
                logger.warning(f"Error en segunda llamada a plot_forecast: {e2}. Creando gráfico básico.")
                plt.figure(figsize=(12, 6))
                plt.plot(y_train.iloc[-90:].index, y_train.iloc[-90:], label="Histórico", color="blue")
                plt.plot(forecast_series.index, forecast_series, label="Pronóstico", color="red", linewidth=2, linestyle="--")
                if ci_lower is not None and ci_upper is not None:
                    plt.fill_between(forecast_series.index, ci_lower, ci_upper, alpha=0.3, color="lightblue")
                plt.title(f"Pronóstico - {instrument} - {model_type}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(forecast_path, dpi=300)
                plt.close()

        paths["forecast"] = forecast_path
        logger.info(f"Gráfico de pronóstico guardado en {forecast_path}")

        try:
            residuals = pd.Series(y_test.values - test_pred, index=y_test.index)
            diagnostic_path = charts_dir / f"{instrument}_{model_type}_diagnostics.png"
            plots.plot_model_diagnostics(
                model_fit=model,
                residuals=residuals,
                title="Diagnóstico de Modelo",
                save_path=diagnostic_path,
                instrument=instrument,
                model_name=model_type,
            )
            paths["diagnostics"] = diagnostic_path
            logger.info(f"Gráfico de diagnóstico guardado en {diagnostic_path}")
        except Exception as e:
            logger.warning(f"Error generando diagnóstico: {str(e)}")

        metrics_dict = metrics.to_dict()
        dashboard_path = charts_dir / f"{instrument}_{model_type}_dashboard.png"
        try:
            plots.create_dashboard(
                real_series=y_test,
                pred_series=test_pred_series,
                forecast_series=forecast_series,
                metrics=metrics_dict,
                title="Dashboard",
                save_path=dashboard_path,
                instrument=instrument,
                model_name=model_type,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            )
        except Exception as e:
            logger.warning(f"Error en primera llamada a create_dashboard: {str(e)}. Intentando alternativa.")
            try:
                plots.create_dashboard(
                    real_series=y_test,
                    pred_series=test_pred_series,
                    forecast_series=forecast_series,
                    metrics=metrics_dict,
                    title="Dashboard",
                    save_path=dashboard_path,
                    instrument=instrument,
                    model_name=model_type,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    train_end_date=train_end_date,
                    val_end_date=val_end_date,
                )
            except Exception as e2:
                logger.warning(f"Error en segunda llamada a create_dashboard: {e2}. No se generó el dashboard.")

        paths["dashboard"] = dashboard_path
        logger.info(f"Dashboard guardado en {dashboard_path}")

        if df_train is None or df_val is None:
            df_train, df_val, _ = split_data(df)
        timeline_path = generate_comprehensive_timeline(
            train_df=df_train,
            val_df=df_val,
            test_df=df_test,
            forecast_df=forecast_df,
            instrument=instrument,
            model_type=model_type,
            output_dir=charts_dir,
        )
        paths["timeline"] = timeline_path
        logger.info(f"Línea de tiempo guardada en {timeline_path}")

    except Exception as e:  # pragma: no cover - runtime behaviour
        logger.error(f"Error generando visualizaciones: {str(e)}")
        logger.error(f"Traza detallada: {traceback.format_exc()}")

    return paths


def generate_model_plots(
    real_series: pd.Series,
    pred_series: pd.Series,
    forecast_series: pd.Series,
    metrics: Dict[str, float],
    residuals: pd.Series,
    model_fit: Any,
    output_dir: Path,
    instrument: str,
    model_name: str,
    ci_lower: Optional[pd.Series] = None,
    ci_upper: Optional[pd.Series] = None,
    inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, Path]:
    """Convenience wrapper to generate all plots for a model."""
    charts_dir = ensure_directory(output_dir / "charts")
    paths = {
        "comparison": charts_dir / f"{instrument}_{model_name}_comparison.png",
        "forecast": charts_dir / f"{instrument}_{model_name}_forecast.png",
        "diagnostics": charts_dir / f"{instrument}_{model_name}_diagnostics.png",
        "dashboard": charts_dir / f"{instrument}_{model_name}_dashboard.png",
    }

    plot_series_comparison(
        real_series=real_series,
        pred_series=pred_series,
        title="Real vs Predicción",
        save_path=paths["comparison"],
        plot_diff=True,
        instrument=instrument,
        model_name=model_name,
        inverse_transform=inverse_transform,
    )

    try:
        plot_forecast(
            historical=real_series,
            forecast=forecast_series,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            title="Pronóstico",
            save_path=paths["forecast"],
            instrument=instrument,
            model_name=model_name,
            inverse_transform=inverse_transform,
        )
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            logger.warning(f"Usando llamada simplificada a plot_forecast debido a error: {str(e)}")
            plot_forecast(
                historical=real_series,
                forecast=forecast_series,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                title="Pronóstico",
                save_path=paths["forecast"],
                instrument=instrument,
                model_name=model_name,
            )
        else:
            raise

    try:
        plot_model_diagnostics(
            model_fit=model_fit,
            residuals=residuals,
            title="Diagnóstico de Modelo",
            save_path=paths["diagnostics"],
            instrument=instrument,
            model_name=model_name,
        )
    except Exception as e:  # pragma: no cover - runtime behaviour
        logger.warning(f"Error generando diagnósticos: {str(e)}")

    try:
        create_dashboard(
            real_series=real_series,
            pred_series=pred_series,
            forecast_series=forecast_series,
            metrics=metrics,
            title="Dashboard",
            save_path=paths["dashboard"],
            instrument=instrument,
            model_name=model_name,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            inverse_transform=inverse_transform,
        )
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            logger.warning(f"Usando llamada simplificada a create_dashboard debido a error: {str(e)}")
            create_dashboard(
                real_series=real_series,
                pred_series=pred_series,
                forecast_series=forecast_series,
                metrics=metrics,
                title="Dashboard",
                save_path=paths["dashboard"],
                instrument=instrument,
                model_name=model_name,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            )
        else:
            raise

    return paths
