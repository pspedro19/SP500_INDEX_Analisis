from __future__ import annotations

from sp500_analysis.config.settings import settings


class EvaluationService:
    """Service responsible for evaluating model predictions."""

    def run_evaluation(self) -> bool:
        from sp500_analysis.application.evaluation.backtester import run_backtest

        run_backtest(
            results_dir=settings.results_dir,
            metrics_dir=settings.metrics_dir,
            charts_dir=settings.metrics_charts_dir,
            subperiods_dir=settings.subperiods_charts_dir,
            date_col=settings.date_col,
        )
        return True
