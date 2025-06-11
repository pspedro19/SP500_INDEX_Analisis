import pytest

try:
    import pandas as pd
    import numpy as np  # noqa: F401 - used indirectly
except Exception:
    pd = None

from pytest import approx

from sp500_analysis.application.inference.data_loader import load_csv, save_csv
from sp500_analysis.application.inference.calculations import (
    parse_european_number,
    compute_predicted_sp500,
    format_for_powerbi,
)


def test_parse_european_number():
    assert parse_european_number("1.234,5") == 1234.5
    assert parse_european_number("2,5") == 2.5
    assert parse_european_number("1000") == 1000


def test_compute_predicted_sp500(tmp_path):
    if pd is None:
        pytest.skip("pandas not available")
    try:
        import numpy  # noqa: F401
    except Exception:
        pytest.skip("numpy not available")
    data = {
        "ValorReal": [0.02, 0.03],
        "ValorPredicho": [0.03, 0.05],
        "ValorReal_SP500": ["1000", ""],
    }
    df = pd.DataFrame(data)
    result = compute_predicted_sp500(df)
    assert "ValorPredicho_SP500" in result
    first = result.loc[0, "ValorPredicho_SP500"]
    second = result.loc[1, "ValorPredicho_SP500"]
    assert first == approx(1000 / (1 + 0.02) * (1 + 0.03))
    assert second == approx(1000 / (1 + 0.02) * (1 + 0.05))

    out_file = tmp_path / "out.csv"
    formatted = format_for_powerbi(result)
    save_csv(formatted, out_file)
    loaded = load_csv(out_file)
    assert not loaded.empty
