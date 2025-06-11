from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from sp500_analysis.shared.logging.logger import configurar_logging


class BaseProcessor(ABC):
    """Generic processor implementing the template method pattern."""

    def __init__(self, data_root: str | Path = "data/0_raw", log_file: str = "processor.log") -> None:
        self.data_root = Path(data_root)
        self.logger = configurar_logging(log_file, name=self.__class__.__name__)

    def run(self, input_data: Any | None = None, output_file: str | Path = "") -> bool:
        """Execute the full processing pipeline."""
        data = self.load(input_data)
        if not self._validate_input(data):
            self.logger.error("Input validation failed")
            return False
        transformed = self._transform(data)
        return self.save(transformed, output_file)

    def load(self, data: Any) -> Any:
        """Default load step simply returns the provided data."""
        return data

    def save(self, data: Any, output_file: str | Path) -> bool:
        """Default save step just ensures the output path exists."""
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).touch()
            return True
        except Exception as exc:  # pragma: no cover - runtime behaviour
            self.logger.error(f"Failed saving output: {exc}")
            return False

    @abstractmethod
    def _transform(self, data: Any) -> Any:
        """Transform loaded data and return the result."""

    @abstractmethod
    def _validate_input(self, data: Any) -> bool:
        """Validate loaded input and return True if valid."""
