from unittest import mock

import sp500_analysis.__main__ as cli


def test_cli_invokes_run_pipeline_main():
    with mock.patch("sp500_analysis.__main__.run_pipeline_main") as run:
        cli.main()
        run.assert_called_once()
