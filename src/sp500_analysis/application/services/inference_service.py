from __future__ import annotations

import logging
from datetime import datetime


from sp500_analysis.config.settings import settings
from sp500_analysis.shared.logging.logger import configurar_logging
from sp500_analysis.application.inference import (
    get_most_recent_file,
    load_all_models,
    get_inference_for_all_models,
    save_all_inference_results,
)


class InferenceService:
    """Service responsible for running inference."""

    def run_inference(self) -> bool:
        log_file = settings.log_dir / f"inference_service_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        configurar_logging(str(log_file))
        logging.info("Running inference service")

        try:
            import pandas as pd
        except Exception:  # pragma: no cover - optional dependency
            logging.error("pandas not available")
            return False

        models = load_all_models(settings.models_dir)
        if not models:
            return False

        dataset_path = get_most_recent_file(settings.training_dir, pattern="*FPI.xlsx")
        if not dataset_path:
            logging.error("No se encontró archivo de datos para inferencia")
            return False

        dataset = pd.read_excel(dataset_path)

        results, forecasts = get_inference_for_all_models(models, dataset)
        if not results:
            return False

        save_all_inference_results(results, forecasts)
        return True


def run_inference() -> bool:
    """Entry point used by the CLI to run inference via the DI container."""
    from sp500_analysis.shared.container import container, setup_container

    setup_container()
    service: InferenceService = container.resolve("inference_service")
    return service.run_inference()
