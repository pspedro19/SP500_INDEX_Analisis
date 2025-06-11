import types
import pytest

try:
    import pandas as pd
except Exception:
    pd = None

from sp500_analysis.application.services.inference_service import InferenceService


def test_run_inference_returns_false_when_no_models(monkeypatch):
    if pd is None:
        pytest.skip("pandas not available")
    monkeypatch.setattr(
        "sp500_analysis.application.services.inference_service.load_all_models",
        lambda *_: {},
    )
    service = InferenceService()
    assert service.run_inference() is False


def test_run_inference_success(monkeypatch, tmp_path):
    if pd is None:
        pytest.skip("pandas not available")
    dummy_model = types.SimpleNamespace(predict=lambda X: [0])
    monkeypatch.setattr(
        "sp500_analysis.application.services.inference_service.load_all_models",
        lambda *_: {"m": dummy_model},
    )

    file_path = tmp_path / "data.xlsx"
    df = pd.DataFrame({"date": ["2020-01-01"], "x": [1], "x_Target": [1]})
    df.to_excel(file_path, index=False)

    monkeypatch.setattr(
        "sp500_analysis.application.services.inference_service.get_most_recent_file",
        lambda *_: str(file_path),
    )
    monkeypatch.setattr(pd, "read_excel", lambda *_: df)

    monkeypatch.setattr(
        "sp500_analysis.application.services.inference_service.get_inference_for_all_models",
        lambda *a, **k: ({"m": {}}, {"m": df}),
    )

    saved = {}

    def fake_save(res, df2):
        saved["called"] = True

    monkeypatch.setattr(
        "sp500_analysis.application.services.inference_service.save_all_inference_results",
        fake_save,
    )

    service = InferenceService()
    assert service.run_inference() is True
    assert saved.get("called")
