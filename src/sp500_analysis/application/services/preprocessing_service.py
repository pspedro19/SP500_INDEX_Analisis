from __future__ import annotations

import logging
import importlib
from datetime import datetime
from sp500_analysis.config.settings import settings
from sp500_analysis.shared.logging.logger import configurar_logging


try:  # pragma: no cover - heavy optional dependency
    ejecutar_todos_los_procesadores = importlib.import_module(
        "pipelines.ml.00_step_preprocess"
    ).ejecutar_todos_los_procesadores
except Exception:  # pragma: no cover - step may not be available
    ejecutar_todos_los_procesadores = None


class PreprocessingService:
    """Service responsible for running the preprocessing pipeline."""

    def run_preprocessing(self) -> bool:
        log_file = settings.log_dir / f"preprocessing_service_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        configurar_logging(str(log_file))
        logging.info("Running preprocessing service")
        if ejecutar_todos_los_procesadores is None:  # pragma: no cover - safeguard
            logging.error("00_step_preprocess module not available")
            return False
        try:
            return ejecutar_todos_los_procesadores()
        except Exception as exc:  # pragma: no cover - runtime failure
            logging.error("Preprocessing failed: %s", exc)
            return False


def run_preprocessing() -> bool:
    """Entry point used by the CLI to run preprocessing via the DI container."""
    from sp500_analysis.shared.container import container, setup_container

    setup_container()
    service: PreprocessingService = container.resolve("preprocessing_service")
    return service.run_preprocessing()
