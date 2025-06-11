import pytest

pandas = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
catboost = pytest.importorskip("catboost")
feature_engine = pytest.importorskip("feature_engine")

from sklearn.model_selection import TimeSeriesSplit

from sp500_analysis.application.feature_engineering import (
    get_most_recent_file,
    plot_cv_splits,
    plot_performance_drift,
    select_features_fpi,
)


def test_get_most_recent_file(tmp_path):
    f1 = tmp_path / "a.xlsx"
    f2 = tmp_path / "b.xlsx"
    f1.write_text("1")
    import time as _time
    _time.sleep(0.01)
    f2.write_text("2")
    assert get_most_recent_file(tmp_path) == str(f2)


def test_plot_cv_splits(tmp_path):
    df = pandas.DataFrame({"a": range(10)})
    tscv = TimeSeriesSplit(n_splits=3, gap=1)
    out = tmp_path / "cv.png"
    plot_cv_splits(df, tscv, out, cv_splits=3, gap=1)
    assert out.exists()


def test_plot_performance_drift(tmp_path):
    out = tmp_path / "drift.png"
    plot_performance_drift(["a", "b"], [0.1, 0.2], ["b"], 0.15, out)
    assert out.exists()


def test_select_features_fpi(tmp_path):
    rng = np.random.default_rng(0)
    X = pandas.DataFrame({"f1": rng.normal(size=30), "f2": rng.normal(size=30)})
    y = rng.normal(size=30)
    features, drifts = select_features_fpi(
        X,
        pandas.Series(y),
        cv_splits=3,
        gap=1,
        threshold=0.0,
        catboost_params={"iterations": 5, "verbose": False},
        plots_dir=tmp_path,
    )
    assert len(features) > 0
    assert len(drifts) == X.shape[1]
