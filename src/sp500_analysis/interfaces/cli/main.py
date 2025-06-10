import click


@click.group()
def cli() -> None:
    """Command line interface for the SP500 analysis toolkit."""
    pass


@cli.command()
def preprocess() -> None:
    """Run the data preprocessing pipeline."""
    from sp500_analysis.application.services.preprocessing_service import run_preprocessing

    run_preprocessing()


@cli.command()
def train() -> None:
    """Train all configured models."""
    from sp500_analysis.application.model_training.trainer import run_training

    run_training()


@cli.command(name="infer")
def infer_command() -> None:
    """Run inference using the trained models."""
    from sp500_analysis.application.services.inference_service import run_inference

    run_inference()


if __name__ == "__main__":  # pragma: no cover
    cli()
