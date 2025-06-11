import pytest

pandas = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
statsmodels = pytest.importorskip("statsmodels")

from sp500_analysis.application.time_series_training import ensemble, training_pipeline


def test_average_series():
    idx = pandas.date_range("2020-01-01", periods=3, freq="D")
    s1 = pandas.Series([1, 2, 3], index=idx)
    s2 = pandas.Series([3, 2, 1], index=idx)
    avg = ensemble.average_series([s1, s2])
    assert list(avg) == [2, 2, 2]


def test_run_training_pipeline(tmp_path):
    df = pandas.DataFrame({
        "date": pandas.date_range("2021-01-01", periods=15, freq="D"),
        "value": np.sin(np.arange(15)),
    })
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)

    forecast = training_pipeline.run_training_pipeline(csv, tmp_path)
    assert forecast.exists()

    forecast2 = training_pipeline.run_training_pipeline(
        csv,
        tmp_path / "ens",
        ensemble_inputs=[forecast],
    )
    assert (tmp_path / "ens" / "forecast_ensemble.csv").exists()
