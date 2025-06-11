from pathlib import Path

from sp500_analysis.config import Settings, get_settings, settings


def test_settings_instantiation():
    s = Settings()
    assert isinstance(s, Settings)


def test_settings_paths_resolved():
    root = Path(__file__).resolve().parents[1]
    s = settings
    assert s.project_root == root
    assert s.log_dir == root / "logs"
    assert s.reports_dir == root / "reports"
    assert s.data_dir == root / "data"
    assert s.raw_dir == s.data_dir / "0_raw"


def test_get_settings_is_singleton():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
