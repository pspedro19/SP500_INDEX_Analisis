from pathlib import Path
import pytest

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas optional
    pd = None
from unittest import mock
import tempfile

import run_pipeline


def test_run_pipeline_with_sample_data(monkeypatch, tmp_path):
    if pd is None:
        pytest.skip("pandas not available")
    sample_csv = Path('data/samples/sample.csv')
    executed = []

    def fake_run_step(step_module, step_name=None):
        # read sample data to simulate processing
        df = pd.read_csv(sample_csv)
        assert not df.empty
        executed.append(step_name)
        return True, 0.0, None

    monkeypatch.setattr(run_pipeline, 'run_step', fake_run_step)

    def fake_generate_html(t, r, s):
        path = run_pipeline.REPORTS_DIR / 'pipeline_report.html'
        with open(path, 'w') as f:
            f.write('report')
        return str(path)

    monkeypatch.setattr(run_pipeline, 'generate_html_report', fake_generate_html)
    monkeypatch.setattr(run_pipeline, 'ensure_directories', lambda: None)

    run_pipeline.REPORTS_DIR = tmp_path / 'reports'
    run_pipeline.RESULTS_DIR = tmp_path / 'results'
    run_pipeline.METRICS_DIR = tmp_path / 'metrics'
    run_pipeline.LOG_DIR = tmp_path / 'logs'
    run_pipeline.IMG_CHARTS_DIR = tmp_path / 'img'
    run_pipeline.METRICS_CHARTS_DIR = tmp_path / 'metrics_charts'
    run_pipeline.CSV_REPORTS = tmp_path / 'csv'
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
    assert len(executed) == len(results) == 11
    assert all(data['success'] for data in results.values())
    assert (run_pipeline.REPORTS_DIR / 'pipeline_timings.json').exists()
    assert (run_pipeline.REPORTS_DIR / 'pipeline_report.html').exists()
