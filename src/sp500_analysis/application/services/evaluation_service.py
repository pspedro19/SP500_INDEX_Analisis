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


def run_evaluation() -> bool:
    """Entry point used by the CLI to run evaluation via the DI container."""
    from sp500_analysis.shared.container import container, setup_container

    setup_container()
    service: EvaluationService = container.resolve("evaluation_service")
    return service.run_evaluation()
