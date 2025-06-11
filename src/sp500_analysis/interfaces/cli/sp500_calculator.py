import click

from sp500_analysis.application.inference.data_loader import load_csv, save_csv
from sp500_analysis.application.inference.calculations import (
    compute_predicted_sp500,
    format_for_powerbi,
)


@click.command()
@click.argument("input_file")
@click.argument("output_file")
def main(input_file: str, output_file: str) -> None:
    """Calculate predicted S&P500 values and save them."""
    df = load_csv(input_file)
    df = compute_predicted_sp500(df)
    df = format_for_powerbi(df)
    save_csv(df, output_file)
    click.echo(f"Saved {output_file}")


if __name__ == "__main__":  # pragma: no cover
    main()
