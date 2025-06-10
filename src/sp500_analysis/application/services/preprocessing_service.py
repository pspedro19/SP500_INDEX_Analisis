from __future__ import annotations

import logging


class PreprocessingService:
    """Service responsible for running the preprocessing pipeline."""

    def run_preprocessing(self) -> bool:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("Running preprocessing service")
        return True


def run_preprocessing() -> bool:
    """Entry point used by the CLI to run preprocessing via the DI container."""
    from sp500_analysis.shared.container import container, setup_container

    setup_container()
    service: PreprocessingService = container.resolve("preprocessing_service")
    return service.run_preprocessing()
