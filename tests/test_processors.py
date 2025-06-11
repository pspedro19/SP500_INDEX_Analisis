from pathlib import Path
import pytest

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency missing
    pd = None
from sp500_analysis.application.preprocessing.processors.eoe import EOEProcessor
from sp500_analysis.application.preprocessing.processors.fred import FredProcessor
from sp500_analysis.application.preprocessing.processors.banco_republica import (
    BancoRepublicaProcessor,
)
from sp500_analysis.application.preprocessing.processors.dane import DANEProcessor
from sp500_analysis.application.preprocessing.processors.economic import (
    EconomicDataProcessor,
)
from sp500_analysis.application.preprocessing.processors.fred_data import (
    FredDataProcessor,
)
from sp500_analysis.application.preprocessing.base import BaseProcessor

try:  # pandas may not be installed
    from sp500_analysis.application.preprocessing.processors.investing import InvestingProcessor
except Exception:  # pragma: no cover - optional dependency missing
    InvestingProcessor = None


def test_eoe_processor_run(tmp_path):
    out_file = tmp_path / "eoe.txt"
    processor = EOEProcessor(data_root=tmp_path)
    assert processor.run(output_file=out_file)
    assert out_file.exists()


def test_fred_processor_run(tmp_path):
    out_file = tmp_path / "fred.txt"
    processor = FredProcessor(config_file="dummy.xlsx", data_root=tmp_path)
    assert processor.run(output_file=out_file)
    assert out_file.exists()


def test_investing_processor_parse_date():
    if InvestingProcessor is None:
        pytest.skip("pandas not available")
    processor = InvestingProcessor(config_file="dummy.xlsx")
    parsed = processor.robust_parse_date("Apr 01, 2025 (Mar)")
    assert parsed is not None
    assert parsed.year == 2025


def test_economic_processor_run(tmp_path):
    out_file = tmp_path / "economic.txt"
    processor = EconomicDataProcessor(config_file="dummy.xlsx", data_root=tmp_path)
    assert processor.run(output_file=out_file)
    assert out_file.exists()


def test_fred_data_processor_run(tmp_path):
    out_file = tmp_path / "fred_data.txt"
    processor = FredDataProcessor(config_file="dummy.xlsx", data_root=tmp_path)
    assert processor.run(output_file=out_file)
    assert out_file.exists()


def test_banco_republica_processor_run(tmp_path):
    out_file = tmp_path / "banco.txt"
    processor = BancoRepublicaProcessor(data_root=tmp_path)
    assert processor.run(output_file=out_file)
    assert out_file.exists()


def test_dane_processor_run(tmp_path):
    out_file = tmp_path / "dane.txt"
    processor = DANEProcessor(data_root=tmp_path)
    assert processor.run(output_file=out_file)
    assert out_file.exists()


class DummyProcessor(BaseProcessor):
    def __init__(self):
        super().__init__(data_root=".")
        self.steps = []

    def _validate_input(self, data):
        self.steps.append("validate")
        return data == "ok"

    def _transform(self, data):
        self.steps.append("transform")
        return data

    def save(self, data, output_file):
        self.steps.append("save")
        return super().save(data, output_file)


def test_base_processor_success(tmp_path):
    out_file = tmp_path / "out.txt"
    proc = DummyProcessor()
    assert proc.run("ok", output_file=out_file)
    assert out_file.exists()
    assert proc.steps == ["validate", "transform", "save"]


def test_base_processor_validation_failure(tmp_path):
    out_file = tmp_path / "fail.txt"
    proc = DummyProcessor()
    assert proc.run("bad", output_file=out_file) is False
    assert not out_file.exists()
    assert proc.steps == ["validate"]


@pytest.mark.skipif(InvestingProcessor is None or pd is None, reason="pandas not available")
def test_investing_private_helpers():
    proc = InvestingProcessor(config_file="dummy.xlsx")

    df = pd.DataFrame({"Release Date": ["01/02/2020", "02/02/2020"]})
    parsed = proc._parse_dates(df.copy(), "dummy")
    assert "fecha" in parsed.columns
    assert parsed["fecha"].is_monotonic_increasing

    series = pd.Series(["1K", "2M", "3%", "4.5"])
    converted = proc._convert_values(series)
    assert converted.tolist() == [1000.0, 2e6, 0.03, 4.5]

    base = pd.DataFrame({"fecha": pd.date_range("2020-01-01", periods=3)})
    df_merge = pd.DataFrame(
        {
            "fecha": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03")],
            "col": [1, 3],
        }
    )
    merged = proc._merge_daily(base, df_merge, "col")
    assert merged["col"].tolist() == [1, 1, 3]
