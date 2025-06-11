from __future__ import annotations

from pathlib import Path

from sp500_analysis.application.preprocessing.factory import ProcessorFactory
from sp500_analysis.config.settings import settings

PROJECT_ROOT = settings.project_root
DATA_ROOT = settings.raw_dir
CONFIG_FILE = PROJECT_ROOT / "pipelines" / "Data Engineering.xlsx"
LOG_DIR = settings.log_dir


def _run_processor(name: str, output_file: Path, config_file: Path | None = None) -> bool:
    log_file = LOG_DIR / f"{name}.log"
    processor = ProcessorFactory.get_processor(name, str(config_file or CONFIG_FILE), str(DATA_ROOT), str(log_file))
    return processor.run(str(output_file))


def run_investing_cp() -> bool:
    return _run_processor("investing", DATA_ROOT / "datos_economicos_procesados_cp.xlsx")


def run_investing_normal() -> bool:
    return _run_processor("investing", DATA_ROOT / "datos_economicos_normales_procesados.xlsx")


def run_fred_data_processor() -> bool:
    return _run_processor("fred", DATA_ROOT / "datos_economicos_procesados_Fred.xlsx")


def run_otherdataprocessor() -> bool:
    return _run_processor("eoe", DATA_ROOT / "datos_economicos_other_procesados.xlsx")


def run_banco_republica_processor() -> bool:
    return _run_processor("banco_republica", DATA_ROOT / "datos_banco_republica_procesados.xlsx")


def run_dane_exportaciones_processor() -> bool:
    return _run_processor("dane", DATA_ROOT / "datos_dane_exportaciones_procesados.xlsx")


def ejecutar_todos_los_procesadores() -> bool:
    steps = [
        run_investing_cp,
        run_investing_normal,
        run_fred_data_processor,
        run_otherdataprocessor,
        run_banco_republica_processor,
        run_dane_exportaciones_processor,
    ]
    results = [step() for step in steps]
    return all(results)


def main() -> bool:
    return ejecutar_todos_los_procesadores()


if __name__ == "__main__":
    import sys
    if not main():
        sys.exit(1)
