from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Minimal interface for trainable models."""

    @abstractmethod
    def train(self, X: Any, y: Any) -> None:
        """Train the model."""

    @abstractmethod
    def predict(self, X: Any):
        """Generate predictions for ``X``."""
