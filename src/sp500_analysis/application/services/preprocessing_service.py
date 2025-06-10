from __future__ import annotations

import logging
from datetime import datetime
from sp500_analysis.config.settings import settings
from sp500_analysis.shared.logging.logger import configurar_logging


class PreprocessingService:
    """Service responsible for running the preprocessing pipeline."""

    def run_preprocessing(self) -> bool:
        log_file = settings.log_dir / f"preprocessing_service_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        configurar_logging(str(log_file))
        logging.info("Running preprocessing service")
        return True


def run_preprocessing() -> bool:
    """Entry point used by the CLI to run preprocessing via the DI container."""
    from sp500_analysis.shared.container import container, setup_container

    setup_container()
    service: PreprocessingService = container.resolve("preprocessing_service")
    return service.run_preprocessing()
