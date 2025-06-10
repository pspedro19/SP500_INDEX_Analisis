import types
import sys
import time
from pathlib import Path

import pytest

import run_pipeline


def test_run_step_propagates_exception(monkeypatch, tmp_path):
    package = tmp_path / "mod"
    package.mkdir()
    (package / "__init__.py").write_text("")
    step = package / "step.py"
    step.write_text("def main():\n    raise ValueError('fail')\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(run_pipeline.StepExecutionError) as excinfo:
        run_pipeline.run_step("mod/step.py", "Fail Step")

    assert isinstance(excinfo.value.original_exc, ValueError)
    assert excinfo.value.step_name == "Fail Step"


def test_generate_html_report_failed_step(monkeypatch, tmp_path):
    # Fake jinja2 template
    class FakeTemplate:
        def __init__(self, template_str):
            self.template_str = template_str

        def render(self, **ctx):
            data = list(ctx["results"].values())[0]
            return f"error:{data['error']} tb:{data['traceback']}"

    fake_module = types.SimpleNamespace(Template=FakeTemplate)
    monkeypatch.setitem(sys.modules, "jinja2", fake_module)

    # patch directories
    for attr in [
        "REPORTS_DIR",
        "IMG_CHARTS_DIR",
        "METRICS_CHARTS_DIR",
        "CSV_REPORTS",
        "RESULTS_DIR",
        "METRICS_DIR",
    ]:
        path = tmp_path / attr.lower()
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(run_pipeline, attr, path)

    monkeypatch.setattr(run_pipeline, "generate_timeline_chart", lambda *a, **k: None)

    timings = {"Step": 0.1}
    results = {
        "Step": {
            "success": False,
            "time": 0.1,
            "error": "boom",
            "traceback": "trace",
        }
    }

    html_path = run_pipeline.generate_html_report(timings, results, time.time())
    content = Path(html_path).read_text()
    assert "boom" in content
    assert "trace" in content
