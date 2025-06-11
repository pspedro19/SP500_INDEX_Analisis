from __future__ import annotations

from typing import Any, Callable, Dict, Tuple


class Container:
    """Very small dependency injection container."""

    def __init__(self) -> None:
        self._providers: Dict[str, Tuple[Callable[[], Any], bool]] = {}
        self._singletons: Dict[str, Any] = {}

    def register(self, name: str, provider: Callable[[], Any] | type, *, singleton: bool = False) -> None:
        """Register a provider callable or class."""
        if not callable(provider):
            raise TypeError("provider must be callable")
        self._providers[name] = (provider, singleton)

    def resolve(self, name: str) -> Any:
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not registered")
        provider, singleton = self._providers[name]
        if singleton:
            if name not in self._singletons:
                self._singletons[name] = provider()
            return self._singletons[name]
        return provider()


container = Container()


def setup_container() -> None:
    """Register default application services."""
    from sp500_analysis.domain.data_repository import DataRepository
    from sp500_analysis.application.model_training.trainer import ModelTrainer
    from sp500_analysis.application.services.training_service import TrainingService
    from sp500_analysis.infrastructure.models.registry import model_registry

    container.register("data_repository", DataRepository, singleton=True)
    container.register("model_trainer", ModelTrainer, singleton=True)
    container.register("model_registry", lambda: model_registry, singleton=True)
    container.register(
        "training_service",
        lambda: TrainingService(
            container.resolve("data_repository"),
            container.resolve("model_trainer"),
            container.resolve("model_registry"),
        ),
        singleton=True,
    )
    from sp500_analysis.application.services.preprocessing_service import PreprocessingService
    from sp500_analysis.application.services.evaluation_service import EvaluationService
    from sp500_analysis.application.services.inference_service import InferenceService

    container.register("preprocessing_service", PreprocessingService, singleton=True)
    container.register("evaluation_service", EvaluationService, singleton=True)
    container.register("inference_service", InferenceService, singleton=True)
    from sp500_analysis.application.feature_engineering import FeatureEngineeringService
    container.register("feature_engineering_service", FeatureEngineeringService, singleton=True)
    from sp500_analysis.application.model_training.ensemble_builder import EnsembleBuilder
    container.register(
        "ensemble_builder",
        lambda: EnsembleBuilder(container.resolve("data_repository")),
        singleton=True,
    )

