from __future__ import annotations

import json
import logging
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from sp500_analysis.shared.logging.logger import configurar_logging
from sp500_analysis.config.settings import settings
from sp500_analysis.application.model_training.temporal_trainer import TemporalModelTrainer
from sp500_analysis.application.model_training.forecaster import ModelForecaster
from sp500_analysis.application.model_training.optimizer import HyperparameterOptimizer
from sp500_analysis.application.evaluation.temporal_evaluator import TemporalEvaluator
from sp500_analysis.domain.data_repository import DataRepository
from sp500_analysis.infrastructure.compute.gpu_manager import configure_gpu
from sp500_analysis.infrastructure.models.registry import ModelRegistry
from sp500_analysis.application.data_processing.temporal_splitter import TemporalDataSplitter
from sp500_analysis.application.data_processing.data_validator import DataValidator

RESULTS_DIR = settings.results_dir


class TrainingService:
    """Service orchestrating temporal model training with the same logic as step_7_0_train_models.py."""

    def __init__(
        self, 
        repository: DataRepository, 
        trainer: TemporalModelTrainer,
        forecaster: ModelForecaster,
        optimizer: HyperparameterOptimizer,
        evaluator: TemporalEvaluator,
        splitter: TemporalDataSplitter,
        validator: DataValidator,
        registry: ModelRegistry
    ) -> None:
        self.repository = repository
        self.trainer = trainer
        self.forecaster = forecaster
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.splitter = splitter
        self.validator = validator
        self.registry = registry

    def run_training(self) -> Dict[str, Dict[str, float]]:
        """
        Execute complete temporal training pipeline with advanced features.
        Now delegates to AdvancedTrainingService for full step_7_0_train_models (1).py functionality.
        """
        
        # Import here to avoid circular imports
        from sp500_analysis.application.services.advanced_training_service import AdvancedTrainingService
        
        # Create advanced training service with all dependencies
        advanced_service = AdvancedTrainingService(
            repository=self.repository,
            trainer=self.trainer,
            forecaster=self.forecaster,
            optimizer=self.optimizer,
            evaluator=self.evaluator,
            splitter=self.splitter,
            validator=self.validator,
            registry=self.registry
        )
        
        # Execute advanced training pipeline
        return advanced_service.run_advanced_training()

    def _load_and_validate_data(self) -> Optional[pd.DataFrame]:
        """Load and validate training data with same logic as step_7_0_train_models.py."""
        
        # Try to load FPI processed file first
        fpi_file = settings.training_dir / "datos_economicos_1month_SP500_TRAINING_FPI.xlsx"
        if fpi_file.exists():
            logging.info(f"✅ Loading FPI processed file: {fpi_file}")
            df = pd.read_excel(fpi_file)
        else:
            # Fallback to latest training file
            df = self.repository.load_latest()
            if df is None:
                logging.error("❌ No training data available")
                return None
            logging.info("⚠️ Using latest training file (not FPI processed)")

        # Validate data
        validation_result = self.validator.validate_temporal_data(df)
        if not validation_result.is_valid:
            logging.error(f"❌ Data validation failed: {validation_result.errors}")
            return None
            
        logging.info(f"✅ Data loaded and validated: {df.shape}")
        return df

    def _save_comprehensive_results(self, results: Dict[str, Any], models: Dict[str, Any]) -> None:
        """Save comprehensive results matching step_7_0_train_models.py output."""
        
        # Save metrics
        metrics_file = RESULTS_DIR / "comprehensive_metrics.json"
        with open(metrics_file, "w") as f:
            # Convert any non-serializable objects to strings
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=4)
        
        # Save trained models
        models_file = RESULTS_DIR / "trained_models.joblib" 
        import joblib
        joblib.dump(models, models_file)
        
        # Save hyperparameters history
        self._save_hyperparameters_history(results)
        
        logging.info(f"📊 Results saved to {metrics_file}")
        logging.info(f"🤖 Models saved to {models_file}")

    def _generate_predictions_csv(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive predictions CSV."""
        try:
            predictions_data = []
            
            for model_name, model_results in results.items():
                if 'forecast_results' in model_results:
                    forecast_data = model_results['forecast_results']
                    if forecast_data and isinstance(forecast_data, dict):
                        # Add model predictions to the list
                        for date, prediction in forecast_data.get('predictions', {}).items():
                            predictions_data.append({
                                'Model': model_name,
                                'Date': date,
                                'Prediction': prediction,
                                'Type': 'Forecast'
                            })
            
            if predictions_data:
                df_predictions = pd.DataFrame(predictions_data)
                predictions_file = RESULTS_DIR / "all_models_predictions.csv"
                df_predictions.to_csv(predictions_file, index=False)
                logging.info(f"📈 Predictions saved to {predictions_file}")
            
        except Exception as e:
            logging.error(f"❌ Error generating predictions CSV: {e}")

    def _save_hyperparameters_history(self, results: Dict[str, Any]) -> None:
        """Save hyperparameters history for future optimization."""
        try:
            hp_data = []
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for model_name, model_results in results.items():
                if 'best_params' in model_results and 'holdout_metrics' in model_results:
                    hp_data.append({
                        'timestamp': current_time,
                        'model': model_name,
                        'parameters': json.dumps(model_results['best_params']),
                        'rmse': model_results['holdout_metrics'].get('RMSE', None),
                        'mae': model_results['holdout_metrics'].get('MAE', None),
                        'r2': model_results['holdout_metrics'].get('R2', None)
                    })
            
            if hp_data:
                df_hp = pd.DataFrame(hp_data)
                hp_file = RESULTS_DIR / "hyperparameters_history.csv"
                
                # Append to existing file or create new
                if hp_file.exists():
                    existing_df = pd.read_csv(hp_file)
                    df_combined = pd.concat([existing_df, df_hp], ignore_index=True)
                else:
                    df_combined = df_hp
                    
                df_combined.to_csv(hp_file, index=False)
                logging.info(f"📋 Hyperparameters history saved to {hp_file}")
                
        except Exception as e:
            logging.error(f"❌ Error saving hyperparameters history: {e}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert non-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return obj


def run_training() -> Dict[str, Dict[str, float]]:
    """Entry point used by the CLI to run training via the DI container."""
    from sp500_analysis.shared.container import container, setup_container

    setup_container()
    service: TrainingService = container.resolve("training_service")
    return service.run_training()
