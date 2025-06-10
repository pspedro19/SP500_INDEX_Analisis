from __future__ import annotations

import logging
from datetime import datetime
from sp500_analysis.config.settings import settings
from sp500_analysis.shared.logging.logger import configurar_logging


class InferenceService:
    """Service responsible for running inference."""

    def run_inference(self) -> bool:
        log_file = settings.log_dir / f"inference_service_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        configurar_logging(str(log_file))
        logging.info("Running inference service")
        return True


def run_inference() -> bool:
    """Entry point used by the CLI to run inference via the DI container."""
    from sp500_analysis.shared.container import container, setup_container

    setup_container()
    service: InferenceService = container.resolve("inference_service")
    return service.run_inference()
