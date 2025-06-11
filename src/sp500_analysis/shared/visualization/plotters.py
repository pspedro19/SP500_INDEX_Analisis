"""Backward compatible exports for visualization helpers."""

from __future__ import annotations

from .common import COLORS, create_directory
from .time_series_plots import (
    plot_real_vs_pred,
    plot_training_curves,
    plot_pipeline_timeline,
)
from .forecast_plots import (
    configure_axis_date,
    plot_series_comparison,
    plot_forecast,
    plot_ensemble_comparison,
    create_dashboard,
    ensure_directory,
    generate_model_visualizations,
    generate_model_plots,
)
from .diagnostic_plots import (
    plot_model_diagnostics,
    plot_metrics_comparison,
    plot_metrics_radar,
    plot_metric_evolution,
)
from .importance_plots import (
    plot_feature_importance,
    plot_correlation_matrix,
)
from .metrics_plots import (
    plot_radar_metrics,
    plot_metrics_by_subperiod,
    generate_report_figures,
)

__all__ = [
    "COLORS",
    "create_directory",
    "plot_real_vs_pred",
    "plot_forecast",
    "plot_training_curves",
    "plot_pipeline_timeline",
    "configure_axis_date",
    "plot_series_comparison",
    "plot_ensemble_comparison",
    "create_dashboard",
    "ensure_directory",
    "generate_model_visualizations",
    "generate_model_plots",
    "plot_model_diagnostics",
    "plot_metrics_comparison",
    "plot_feature_importance",
    "plot_correlation_matrix",
    "plot_radar_metrics",
    "plot_metrics_radar",
    "plot_metric_evolution",
    "plot_metrics_by_subperiod",
    "generate_report_figures",
]
