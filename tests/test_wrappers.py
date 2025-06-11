import importlib.util
from pathlib import Path
import pytest

WRAPPERS_DIR = Path('src/sp500_analysis/infrastructure/models/wrappers')


def load_wrapper(name):
    path = WRAPPERS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lightgbm_wrapper_importerror():
    mod = load_wrapper('lightgbm_wrapper')
    with pytest.raises(ImportError):
        mod.LightGBMWrapper()


def test_xgboost_wrapper_importerror():
    mod = load_wrapper('xgboost_wrapper')
    with pytest.raises(ImportError):
        mod.XGBoostWrapper()


def test_catboost_wrapper_importerror():
    mod = load_wrapper('catboost_wrapper')
    with pytest.raises(ImportError):
        mod.CatBoostWrapper()


def test_lstm_wrapper_importerror():
    mod = load_wrapper('lstm_wrapper')
    with pytest.raises(ImportError):
        mod.LSTMWrapper()


def test_registry_discovers_wrappers(monkeypatch):
    monkeypatch.syspath_prepend("src")
    import importlib, sys

    if "sp500_analysis.infrastructure.models.registry" in sys.modules:
        del sys.modules["sp500_analysis.infrastructure.models.registry"]
    importlib.import_module("sp500_analysis.infrastructure.models.registry")
    from sp500_analysis.infrastructure.models.registry import model_registry

    names = {name for name, _ in model_registry.items()}
    expected = {"CatBoost", "LightGBM", "XGBoost", "LSTM"}
    assert expected.issubset(names)
