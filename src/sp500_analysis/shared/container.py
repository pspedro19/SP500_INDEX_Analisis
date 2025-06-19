from __future__ import annotations

from typing import Dict, Any, Callable, Optional, Tuple


class SimpleContainer:
    """Simple dependency injection container."""

    def __init__(self) -> None:
        self._providers: Dict[str, Tuple[Callable[[], Any], bool]] = {}
        self._singletons: Dict[str, Any] = {}

    def register(self, name: str, provider: Callable[[], Any] | type, *, singleton: bool = False) -> None:
        """Register a provider callable or class."""
        if not callable(provider):
            raise TypeError("provider must be callable")
        self._providers[name] = (provider, singleton)

    def resolve(self, name: str) -> Any:
        """Resolve a dependency by name."""
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not registered")
        provider, singleton = self._providers[name]
        if singleton:
            if name not in self._singletons:
                self._singletons[name] = provider()
            return self._singletons[name]
        return provider()

    def clear(self) -> None:
        """Clear all registrations."""
        self._providers.clear()
        self._singletons.clear()


container = SimpleContainer()


def setup_container() -> None:
    """Set up the dependency injection container."""
    container.clear()

    # Data layer
    from sp500_analysis.domain.data_repository import DataRepository
    container.register("repository", DataRepository, singleton=True)

    # Core training components (NEW TEMPORAL ARCHITECTURE)
    from sp500_analysis.application.model_training.temporal_trainer import TemporalModelTrainer
    container.register("temporal_trainer", TemporalModelTrainer)

    from sp500_analysis.application.model_training.forecaster import ModelForecaster
    container.register("forecaster", ModelForecaster)

    from sp500_analysis.application.model_training.optimizer import HyperparameterOptimizer
    container.register("optimizer", HyperparameterOptimizer)

    from sp500_analysis.application.evaluation.temporal_evaluator import TemporalEvaluator
    container.register("temporal_evaluator", TemporalEvaluator)

    from sp500_analysis.application.data_processing.temporal_splitter import TemporalDataSplitter
    container.register("temporal_splitter", TemporalDataSplitter)

    from sp500_analysis.application.data_processing.data_validator import DataValidator
    container.register("data_validator", DataValidator)

    # Legacy components (keeping for backward compatibility)
    from sp500_analysis.application.model_training.trainer import ModelTrainer
    container.register("trainer", ModelTrainer)

    from sp500_analysis.application.evaluation.evaluator import Evaluator
    container.register("evaluator", Evaluator)

    # Infrastructure
    from sp500_analysis.infrastructure.models.registry import model_registry
    container.register("registry", lambda: model_registry, singleton=True)
    
    # TTS Model Registration (Transformer Time Series)
    from sp500_analysis.infrastructure.models.wrappers.tts_wrapper import TTSWrapper
    try:
        # Register TTS model if PyTorch is available
        model_registry.register("TTS", TTSWrapper)
        print("✅ TTS (Transformer Time Series) model registered successfully")
    except ImportError:
        print("⚠️ TTS model not registered - PyTorch not available")

    # Service layer - UPDATED TO USE NEW TEMPORAL COMPONENTS
    from sp500_analysis.application.services.training_service import TrainingService
    container.register(
        "training_service", 
        lambda: TrainingService(
            repository=container.resolve("repository"),
            trainer=container.resolve("temporal_trainer"),
            forecaster=container.resolve("forecaster"),
            optimizer=container.resolve("optimizer"),
            evaluator=container.resolve("temporal_evaluator"),
            splitter=container.resolve("temporal_splitter"),
            validator=container.resolve("data_validator"),
            registry=container.resolve("registry")
        ),
        singleton=True
    )

    # Other services
    from sp500_analysis.application.services.preprocessing_service import PreprocessingService
    container.register("preprocessing_service", PreprocessingService, singleton=True)

    from sp500_analysis.application.services.evaluation_service import EvaluationService
    container.register("evaluation_service", EvaluationService, singleton=True)

    from sp500_analysis.application.services.inference_service import InferenceService
    container.register("inference_service", InferenceService, singleton=True)

    from sp500_analysis.application.feature_engineering import FeatureEngineeringService
    container.register("feature_engineering_service", FeatureEngineeringService, singleton=True)

    from sp500_analysis.application.services.category_service import CategoryService
    container.register("category_service", CategoryService, singleton=True)

    from sp500_analysis.application.model_training.ensemble_builder import EnsembleBuilder
    container.register(
        "ensemble_builder",
        lambda: EnsembleBuilder(container.resolve("repository")),
        singleton=True
    )
