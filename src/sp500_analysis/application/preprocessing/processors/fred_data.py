from typing import Any

from sp500_analysis.application.preprocessing.base import BaseProcessor
from sp500_analysis.shared.logging.logger import configurar_logging


class FredDataProcessor(BaseProcessor):
    """Placeholder processor for FRED data."""

    def __init__(self, config_file: str, data_root: str | None = "data/0_raw", log_file: str = "fred_data.log"):
        super().__init__(data_root=data_root, log_file=log_file)
        self.config_file = config_file

    def _validate_input(self, data: Any) -> bool:
        return True

    def _transform(self, data: Any) -> Any:
        self.logger.info("Running FRED Data processor")
        return data
