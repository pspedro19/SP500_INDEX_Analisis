from typing import Any

from sp500_analysis.application.preprocessing.base import BaseProcessor
from sp500_analysis.shared.logging.logger import configurar_logging


class DANEProcessor(BaseProcessor):
    """Placeholder processor for DANE export data."""

    def __init__(self, data_root: str = "data/0_raw", output_path: str = None, log_file: str = "dane_exportaciones.log"):
        super().__init__(data_root=data_root, log_file=log_file)
        self.output_path = output_path


    def _validate_input(self, data: Any) -> bool:
        return True

    def _transform(self, data: Any) -> Any:
        self.logger.info("Running DANE processor")
        return data
