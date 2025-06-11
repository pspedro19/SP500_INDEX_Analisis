from __future__ import annotations

"""Pipeline step to run model backtests."""

from sp500_analysis.config.settings import settings
from sp500_analysis.application.evaluation.backtester import run_backtest


def main() -> None:
    run_backtest(
        results_dir=settings.results_dir,
        metrics_dir=settings.metrics_dir,
        charts_dir=settings.metrics_charts_dir,
        subperiods_dir=settings.subperiods_charts_dir,
        date_col=settings.date_col,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
