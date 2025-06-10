import click
from sp500_analysis.shared.container import container, setup_container


@click.group()
def cli() -> None:
    """Command line interface for the SP500 analysis toolkit."""
    pass


@cli.command()
def preprocess() -> None:
    """Run the data preprocessing pipeline."""
    setup_container()
    service = container.resolve("preprocessing_service")
    service.run_preprocessing()


@cli.command()
def train() -> None:
    """Train all configured models."""
    setup_container()
    service = container.resolve("training_service")
    service.run_training()


@cli.command(name="infer")
def infer_command() -> None:
    """Run inference using the trained models."""
    setup_container()
    service = container.resolve("inference_service")
    service.run_inference()


@cli.command()
def backtest() -> None:
    """Run model backtests on the latest predictions."""
    setup_container()
    service = container.resolve("evaluation_service")
    service.run_evaluation()


if __name__ == "__main__":  # pragma: no cover
    cli()
