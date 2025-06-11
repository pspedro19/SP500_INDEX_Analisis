try:
    import pandas as pd
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    pd = None
    np = None
from sp500_analysis.application.feature_engineering import (
    FeatureSelector,
    remove_correlated_features_by_frequency,
    iterative_vif_reduction_by_frequency,
    compute_vif_by_frequency,
)
import pytest

if pd is None:
    pytest.skip("pandas not available", allow_module_level=True)


def test_feature_selector_drop_constant():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "a": [1, 1, 1, 1, 1],
            "b": [1, 2, 3, 4, 5],
            "c": [0, 0, 0, 0, 0],
        }
    )
    selector = FeatureSelector(date_col="date", constant_threshold=0.95)
    result, dropped = selector.drop_constant_features(df)
    assert set(dropped) == {"a", "c"}
    assert "a" not in result.columns and "c" not in result.columns


def test_feature_selector_stationarity():
    np.random.seed(0)
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    stationary = np.random.normal(size=50)
    non_stationary = np.cumsum(np.random.normal(size=50))
    df = pd.DataFrame({"date": dates, "x": stationary, "y": non_stationary})
    selector = FeatureSelector(date_col="date")
    res = selector.test_stationarity(df)
    is_stat = dict(zip(res["column"], res["is_stationary"]))
    assert is_stat["x"] is True
    assert is_stat["y"] is False


def test_remove_correlated_features_by_frequency():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    f1 = np.arange(30)
    f2 = f1 * 2 + np.random.normal(scale=0.01, size=30)
    target = pd.Series(np.random.normal(size=30))
    df = pd.DataFrame({"date": dates, "f1": f1, "f2": f2, "f3": np.random.normal(size=30)})
    cleaned, dropped = remove_correlated_features_by_frequency(df, target, date_col="date", threshold=0.8)
    assert len(dropped) == 1
    assert ("f1" in dropped) ^ ("f2" in dropped)
    assert ("f1" not in cleaned.columns) or ("f2" not in cleaned.columns)


def test_iterative_vif_reduction_by_frequency():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    f1 = np.arange(30)
    f2 = f1 * 2
    df = pd.DataFrame({"date": dates, "f1": f1, "f2": f2})
    reduced, vif_df = iterative_vif_reduction_by_frequency(df, date_col="date", threshold=5.0)
    assert vif_df["VIF"].max() <= 5.0
    assert reduced.shape[1] < df.shape[1]
