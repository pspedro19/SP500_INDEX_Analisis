"""Thin wrapper to trigger model training via the service layer."""

from sp500_analysis.application.model_training.trainer import run_training


def main() -> None:
    """Execute the training pipeline."""
    run_training()


if __name__ == "__main__":  # pragma: no cover
    main()
