import logging
from unittest import mock

import pytest

import sp500_analysis.application.services.preprocessing_service as prep_service


@pytest.fixture(autouse=True)
def clear_logging_handlers():
    logging.getLogger().handlers.clear()
    yield
    logging.getLogger().handlers.clear()


def test_run_preprocessing_logs_and_invokes(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(prep_service.settings, "log_dir", tmp_path)
    with mock.patch.object(prep_service, "ejecutar_todos_los_procesadores", return_value=True) as proc:
        with caplog.at_level(logging.INFO):
            service = prep_service.PreprocessingService()
            assert service.run_preprocessing() is True
        proc.assert_called_once()
    assert any("Running preprocessing service" in m for m in caplog.text.splitlines())


def test_run_preprocessing_handles_exception(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(prep_service.settings, "log_dir", tmp_path)

    def fail():
        raise RuntimeError("boom")

    with mock.patch.object(prep_service, "ejecutar_todos_los_procesadores", side_effect=fail) as proc:
        with caplog.at_level(logging.INFO):
            service = prep_service.PreprocessingService()
            assert service.run_preprocessing() is False
        proc.assert_called_once()
    assert any("Preprocessing failed" in m for m in caplog.text.splitlines())
