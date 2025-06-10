from __future__ import annotations

from typing import Dict, Type, Iterator


class ModelRegistry:
    """Registry of available model wrappers."""

    def __init__(self) -> None:
        self._models: Dict[str, Type] = {}

    def register(self, name: str, cls: Type) -> None:
        self._models[name] = cls

    def get(self, name: str) -> Type:
        return self._models[name]

    def items(self) -> Iterator[tuple[str, Type]]:
        return self._models.items()


from .wrappers import (
    CatBoostWrapper,
    LightGBMWrapper,
    XGBoostWrapper,
    MLPWrapper,
    SVMWrapper,
    LSTMWrapper,
)

model_registry = ModelRegistry()
model_registry.register("CatBoost", CatBoostWrapper)
model_registry.register("LightGBM", LightGBMWrapper)
model_registry.register("XGBoost", XGBoostWrapper)
model_registry.register("MLP", MLPWrapper)
model_registry.register("SVM", SVMWrapper)
model_registry.register("LSTM", LSTMWrapper)

__all__ = ["ModelRegistry", "model_registry"]
