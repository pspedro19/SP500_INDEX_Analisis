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
        if isinstance(step_module, str):
            if not Path(step_module).is_file():
                raise ImportError(f"Module {step_module} not found")
        executed_steps.append(step_module)
        return True, 0.0, None

    with mock.patch.object(run_pipeline, "run_step", side_effect=fake_run_step):
        with mock.patch.object(run_pipeline, "generate_html_report", return_value=None):
            with mock.patch.object(run_pipeline, "ensure_directories", lambda: None):
                with mock.patch.object(run_pipeline, "setup_container", lambda: None):
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


def test_run_pipeline_executes_services(monkeypatch, tmp_path):
    """Run the pipeline using stub services and ensure output files are created."""

    class DummyTrainingService:
        def run_training(self):
            (run_pipeline.RESULTS_DIR / "metrics_simple.json").write_text("{}")
            return {"model": {"mse": 0.0}}

    class DummyEnsembleBuilder:
        def build(self):
            (run_pipeline.RESULTS_DIR / "ensemble.csv").write_text("ok")
            return True

    class DummyEvaluationService:
        def run_evaluation(self):
            run_pipeline.METRICS_DIR.mkdir(parents=True, exist_ok=True)
            (run_pipeline.METRICS_DIR / "resultados_totales.csv").write_text("a,b\n1,2")
            return True

    class DummyInferenceService:
        def run_inference(self):
            (run_pipeline.RESULTS_DIR / "inference.csv").write_text("x")
            return True

    services = {
        "training_service": DummyTrainingService(),
        "ensemble_builder": DummyEnsembleBuilder(),
        "evaluation_service": DummyEvaluationService(),
        "inference_service": DummyInferenceService(),
    }

    def fake_resolve(name):
        return services[name]

    monkeypatch.setattr(run_pipeline, "setup_container", lambda: None)
    import sp500_analysis.shared.container as di_container
    monkeypatch.setattr(di_container.container, "resolve", lambda name: fake_resolve(name))

    def fake_run_step(step_module, step_name=None):
        name = step_module if isinstance(step_module, str) else getattr(step_module, "__name__", "")
        if "07_step_train_models.py" in name:
            (run_pipeline.RESULTS_DIR / "metrics_simple.json").write_text("{}")
        elif "07b_step_ensemble.py" in name:
            (run_pipeline.RESULTS_DIR / "ensemble.csv").write_text("ok")
        elif "09_step_backtest.py" in name:
            run_pipeline.METRICS_DIR.mkdir(parents=True, exist_ok=True)
            (run_pipeline.METRICS_DIR / "resultados_totales.csv").write_text("a,b\n1,2")
        elif "10_step_inference.py" in name:
            (run_pipeline.RESULTS_DIR / "inference.csv").write_text("x")
        elif callable(step_module):
            step_module()
        else:
            Path(run_pipeline.RESULTS_DIR / Path(str(name)).stem).write_text("done")
        return True, 0.0, None

    monkeypatch.setattr(run_pipeline, "run_step", fake_run_step)
    monkeypatch.setattr(run_pipeline, "generate_html_report", lambda *a, **k: None)
    monkeypatch.setattr(run_pipeline, "ensure_directories", lambda: None)

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

    assert len(results) == 15
    assert all(r["success"] for r in results.values())
    assert (run_pipeline.RESULTS_DIR / "metrics_simple.json").exists()
    assert (run_pipeline.METRICS_DIR / "resultados_totales.csv").exists()
    assert (run_pipeline.RESULTS_DIR / "inference.csv").exists()
