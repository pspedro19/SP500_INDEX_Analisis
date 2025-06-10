"""Module entry point for the ``sp500`` command line interface."""

from sp500_analysis.interfaces.cli.main import cli


def main() -> None:
    """Invoke the `sp500` CLI."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
