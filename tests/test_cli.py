from unittest import mock
import types
import sys
from click.testing import CliRunner

from sp500_analysis.interfaces.cli.main import cli


def test_preprocess_invokes_service():
    runner = CliRunner()
    with mock.patch("sp500_analysis.application.services.preprocessing_service.run_preprocessing") as run:
        result = runner.invoke(cli, ["preprocess"])
        assert result.exit_code == 0
        run.assert_called_once()


def test_train_invokes_service():
    runner = CliRunner()
    dummy = types.SimpleNamespace(run_training=lambda: None)
    sys.modules["sp500_analysis.application.model_training.trainer"] = dummy
    with mock.patch.object(dummy, "run_training") as run:
        result = runner.invoke(cli, ["train"])
        assert result.exit_code == 0
        run.assert_called_once()


def test_infer_invokes_service():
    runner = CliRunner()
    with mock.patch("sp500_analysis.application.services.inference_service.run_inference") as run:
        result = runner.invoke(cli, ["infer"])
        assert result.exit_code == 0
        run.assert_called_once()


def test_backtest_invokes_service():
    runner = CliRunner()
    dummy = types.SimpleNamespace(run_backtest=lambda **kwargs: None)
    sys.modules["sp500_analysis.application.evaluation.backtester"] = dummy
    with mock.patch.object(dummy, "run_backtest") as run:
        result = runner.invoke(cli, ["backtest"])
        assert result.exit_code == 0
        run.assert_called_once()
