"""Thin wrapper to build the greedy ensemble via the application layer."""

from sp500_analysis.application.model_training.ensemble_builder import run_ensemble


def main() -> None:
    run_ensemble()


if __name__ == "__main__":  # pragma: no cover
    main()
