from unittest import mock
import types
from click.testing import CliRunner

from sp500_analysis.interfaces.cli.main import cli


def test_preprocess_invokes_service():
    runner = CliRunner()
    service = types.SimpleNamespace(run_preprocessing=lambda: None)
    with mock.patch("sp500_analysis.interfaces.cli.main.setup_container", lambda: None):
        with mock.patch(
            "sp500_analysis.interfaces.cli.main.container.resolve",
            return_value=service,
        ) as resolve:
            with mock.patch.object(service, "run_preprocessing") as run:
                result = runner.invoke(cli, ["preprocess"])
                assert result.exit_code == 0
                resolve.assert_called_once_with("preprocessing_service")
                run.assert_called_once()


def test_train_invokes_service():
    runner = CliRunner()
    service = types.SimpleNamespace(run_training=lambda: None)
    with mock.patch("sp500_analysis.interfaces.cli.main.setup_container", lambda: None):
        with mock.patch(
            "sp500_analysis.interfaces.cli.main.container.resolve",
            return_value=service,
        ) as resolve:
            with mock.patch.object(service, "run_training") as run:
                result = runner.invoke(cli, ["train"])
                assert result.exit_code == 0
                resolve.assert_called_once_with("training_service")
                run.assert_called_once()


def test_infer_invokes_service():
    runner = CliRunner()
    service = types.SimpleNamespace(run_inference=lambda: None)
    with mock.patch("sp500_analysis.interfaces.cli.main.setup_container", lambda: None):
        with mock.patch(
            "sp500_analysis.interfaces.cli.main.container.resolve",
            return_value=service,
        ) as resolve:
            with mock.patch.object(service, "run_inference") as run:
                result = runner.invoke(cli, ["infer"])
                assert result.exit_code == 0
                resolve.assert_called_once_with("inference_service")
                run.assert_called_once()


def test_backtest_invokes_service():
    runner = CliRunner()
    service = types.SimpleNamespace(run_evaluation=lambda: None)
    with mock.patch("sp500_analysis.interfaces.cli.main.setup_container", lambda: None):
        with mock.patch(
            "sp500_analysis.interfaces.cli.main.container.resolve",
            return_value=service,
        ) as resolve:
            with mock.patch.object(service, "run_evaluation") as run:
                result = runner.invoke(cli, ["backtest"])
                assert result.exit_code == 0
                resolve.assert_called_once_with("evaluation_service")
                run.assert_called_once()
