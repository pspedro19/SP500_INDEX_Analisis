import pytest

pandas = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
statsmodels = pytest.importorskip("statsmodels")

from sp500_analysis.application.time_series_training import (
    data_loading,
    hyperparameter_search,
    model_fitting,
    result_export,
)


def test_load_and_split(tmp_path):
    csv = tmp_path / "data.csv"
    df = pandas.DataFrame({
        "date": pandas.date_range("2020-01-01", periods=10, freq="D"),
        "value": range(10),
    })
    df.to_csv(csv, index=False)

    loaded = data_loading.load_data(csv)
    assert isinstance(loaded.index, pandas.DatetimeIndex)
    train, val, test = data_loading.split_data(loaded, val_size=2, test_size=2)
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2


def test_full_flow(tmp_path):
    df = pandas.DataFrame({
        "date": pandas.date_range("2021-01-01", periods=20, freq="D"),
        "value": np.sin(np.arange(20)),
    })
    csv = tmp_path / "ts.csv"
    df.to_csv(csv, index=False)

    data = data_loading.load_data(csv)
    train, _, test = data_loading.split_data(data, val_size=5, test_size=5)
    order = hyperparameter_search.search_arima_order(train["value"], p_values=[0,1], d_values=[0], q_values=[0])
    model, preds = model_fitting.fit_model(train["value"], test["value"], order)
    assert len(preds) == len(test)

    out = result_export.export_forecast(preds, tmp_path / "out.csv")
    assert out.exists()
