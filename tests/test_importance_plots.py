import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import numpy as np

from sp500_analysis.shared.visualization.importance_plots import plot_feature_importance


def test_plot_feature_importance(tmp_path):
    importances = np.array([0.2, 0.5, 0.3])
    names = ["f1", "f2", "f3"]

    fig = plot_feature_importance(importances, names)
    assert fig is not None

    out_file = tmp_path / "imp.png"
    plot_feature_importance(importances, names, output_path=str(out_file))
    assert out_file.exists()
