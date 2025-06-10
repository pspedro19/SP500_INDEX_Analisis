import importlib
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import run_pipeline


def test_run_pipeline_steps_importable():
    executed_steps = []

    def fake_run_step(step_module, step_name=None):
        # Ensure the referenced script exists without executing heavy code
        if not Path(step_module).is_file():
            raise ImportError(f"Module {step_module} not found")
        executed_steps.append(step_module)
        return True, 0.0, None

    with mock.patch.object(run_pipeline, "run_step", side_effect=fake_run_step):
        with mock.patch.object(run_pipeline, "generate_html_report", return_value=None):
            with mock.patch.object(run_pipeline, "ensure_directories", lambda: None):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)
                    run_pipeline.REPORTS_DIR = tmp_path / "reports"
                    run_pipeline.RESULTS_DIR = tmp_path / "results"
                    run_pipeline.METRICS_DIR = tmp_path / "metrics"
                    run_pipeline.LOG_DIR = tmp_path / "logs"
                    run_pipeline.IMG_CHARTS_DIR = tmp_path / "img"
                    run_pipeline.METRICS_CHARTS_DIR = tmp_path / "metrics_charts"
                    run_pipeline.CSV_REPORTS = tmp_path / "csv"
                    for p in [
                        run_pipeline.REPORTS_DIR,
                        run_pipeline.RESULTS_DIR,
                        run_pipeline.METRICS_DIR,
                        run_pipeline.LOG_DIR,
                        run_pipeline.IMG_CHARTS_DIR,
                        run_pipeline.METRICS_CHARTS_DIR,
                        run_pipeline.CSV_REPORTS,
                    ]:
                        p.mkdir(parents=True, exist_ok=True)
                    results = run_pipeline.main()

    assert executed_steps
    assert len(executed_steps) == len(results)
    assert all(data["success"] for data in results.values())
