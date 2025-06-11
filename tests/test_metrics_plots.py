import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from sp500_analysis.shared.visualization.metrics_plots import plot_radar_metrics


def test_plot_radar_metrics(tmp_path):
    metrics = {
        "m1": {"RMSE": 1.0, "MAE": 0.5},
        "m2": {"RMSE": 0.8, "MAE": 0.4},
    }
    fig = plot_radar_metrics(metrics)
    assert fig is not None

    out_file = tmp_path / "radar.png"
    plot_radar_metrics(metrics, output_path=str(out_file))
    assert out_file.exists()
