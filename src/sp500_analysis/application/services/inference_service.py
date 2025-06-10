from __future__ import annotations

import logging


class InferenceService:
    """Service responsible for running inference."""

    def run_inference(self) -> bool:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("Running inference service")
        return True


def run_inference() -> bool:
    """Entry point used by the CLI to run inference via the DI container."""
    from sp500_analysis.shared.container import container, setup_container

    setup_container()
    service: InferenceService = container.resolve("inference_service")
    return service.run_inference()
