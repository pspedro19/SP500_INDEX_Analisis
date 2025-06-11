try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas optional
    pd = None

import pytest

from sp500_analysis.application.preprocessing.processors.filter_20days import (
    filter_last_n_days,
)


@pytest.mark.skipif(pd is None, reason="pandas not available")
def test_filter_last_n_days_with_date():
    df1 = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=30, freq="D"),
        "A": range(30),
        "B": range(100, 130),
        "A_Return_Target": range(30),
    })
    df2 = pd.DataFrame(columns=["date", "A", "B"])

    result = filter_last_n_days(df1, df2, days=20, date_col="date", target_suffix="_Return_Target")
    assert list(result.columns) == ["date", "A", "B"]
    assert len(result) == 20
    assert result["date"].is_monotonic_increasing
    assert result["date"].iloc[0] == df1["date"].iloc[10]
    assert result["date"].iloc[-1] == df1["date"].iloc[29]


@pytest.mark.skipif(pd is None, reason="pandas not available")
def test_filter_last_n_days_without_date():
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "A_Return_Target": [7, 8, 9]})
    df2 = pd.DataFrame(columns=["A", "B"])

    result = filter_last_n_days(df1, df2, days=2, date_col="date", target_suffix="_Return_Target")
    assert list(result.columns) == ["A", "B"]
    assert len(result) == 2
    assert result["A"].tolist() == [2, 3]

