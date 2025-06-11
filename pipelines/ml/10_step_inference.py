"""Run inference using the high level service."""

from pipelines.ml.config import ensure_directories
from sp500_analysis.application.services.inference_service import run_inference


def main() -> None:
    ensure_directories()
    run_inference()


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
