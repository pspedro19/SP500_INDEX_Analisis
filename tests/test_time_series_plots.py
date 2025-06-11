import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import pandas as pd
from sp500_analysis.shared.visualization.time_series_plots import plot_real_vs_pred


def test_plot_real_vs_pred_returns_figure(tmp_path):
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="D"),
            "Valor_Real": [1.0, 2.0, 3.0],
            "Valor_Predicho": [1.1, 2.1, 2.9],
        }
    )

    fig = plot_real_vs_pred(df)
    assert fig is not None

    out_file = tmp_path / "plot.png"
    plot_real_vs_pred(df, output_path=str(out_file))
    assert out_file.exists()
