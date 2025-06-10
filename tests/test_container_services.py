from sp500_analysis.shared.container import container, setup_container
from sp500_analysis.application.services.preprocessing_service import PreprocessingService
from sp500_analysis.application.services.inference_service import InferenceService
from sp500_analysis.application.services.evaluation_service import EvaluationService
import types
import sys


def test_container_resolves_registered_services():
    dummy_repo = types.SimpleNamespace(DataRepository=object)
    dummy_trainer = types.SimpleNamespace(ModelTrainer=object)
    dummy_training_service = types.SimpleNamespace(TrainingService=lambda *a, **k: None)
    dummy_registry = types.SimpleNamespace(model_registry={})

    sys.modules['sp500_analysis.domain.data_repository'] = dummy_repo
    sys.modules['sp500_analysis.application.model_training.trainer'] = dummy_trainer
    sys.modules['sp500_analysis.application.services.training_service'] = dummy_training_service
    sys.modules['sp500_analysis.infrastructure.models.registry'] = dummy_registry

    setup_container()
    assert isinstance(container.resolve("preprocessing_service"), PreprocessingService)
    assert isinstance(container.resolve("inference_service"), InferenceService)
    assert isinstance(container.resolve("evaluation_service"), EvaluationService)
