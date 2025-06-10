from pathlib import Path
import pytest
from sp500_analysis.application.preprocessing.processors.eoe import EOEProcessor
from sp500_analysis.application.preprocessing.processors.fred import FredProcessor

try:  # pandas may not be installed
    from sp500_analysis.application.preprocessing.processors.investing import InvestingProcessor
except Exception:  # pragma: no cover - optional dependency missing
    InvestingProcessor = None


def test_eoe_processor_run(tmp_path):
    out_file = tmp_path / "eoe.txt"
    processor = EOEProcessor(data_root=tmp_path)
    assert processor.run(out_file)
    assert out_file.exists()


def test_fred_processor_run(tmp_path):
    out_file = tmp_path / "fred.txt"
    processor = FredProcessor(config_file="dummy.xlsx", data_root=tmp_path)
    assert processor.run(out_file)
    assert out_file.exists()


def test_investing_processor_parse_date():
    if InvestingProcessor is None:
        pytest.skip("pandas not available")
    processor = InvestingProcessor(config_file="dummy.xlsx")
    parsed = processor.robust_parse_date("Apr 01, 2025 (Mar)")
    assert parsed is not None
    assert parsed.year == 2025
