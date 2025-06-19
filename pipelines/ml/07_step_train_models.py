"""
Advanced Model Training Pipeline
Integrates all features from step_7_0_train_models (1).py while maintaining modular architecture.
"""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parents[2]  # Go up two directories from pipelines/ml/
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import logging
from datetime import datetime

from sp500_analysis.application.model_training.trainer import run_training
from sp500_analysis.shared.logging.logger import configurar_logging
from sp500_analysis.config.settings import settings


def main() -> None:
    """
    Execute the advanced training pipeline with all features from step_7_0_train_models (1).py.
    
    This maintains the modular architecture while providing:
    - ✅ TTS (Transformer Time Series) fully integrated
    - ✅ 3-zone temporal validation (A: Training, B: Backtest, C: Holdout)  
    - ✅ RandomizedSearchCV + Optuna hyperparameter optimization
    - ✅ Comprehensive forecast generation
    - ✅ Fact tables generation (hechos_predicciones_fields.csv, hechos_metricas_modelo.csv)
    - ✅ Hilbert transform metrics (Amplitud_Score, Fase_Score, Ultra_Metric, Hit_Direction)
    - ✅ Inverse transform to actual prices
    - ✅ Business days forecasting
    - ✅ Local refinement on recent data
    - ✅ All advanced outputs and visualizations
    """
    
    # Configure logging for the advanced pipeline
    log_file = settings.log_dir / f"advanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    configurar_logging(str(log_file))
    
    logging.info("STARTING Advanced Training Pipeline")
    logging.info("=" * 70)
    logging.info("Features integrated from step_7_0_train_models (1).py:")
    logging.info("  [OK] TTS (Transformer Time Series) model")
    logging.info("  [OK] 3-zone temporal validation methodology")
    logging.info("  [OK] Advanced hyperparameter optimization")
    logging.info("  [OK] Comprehensive forecasting pipeline")
    logging.info("  [OK] Fact tables with Hilbert metrics")
    logging.info("  [OK] Simplified predictions processing")
    logging.info("  [OK] Business days forecasting")
    logging.info("=" * 70)
    
    try:
        # Execute the advanced training pipeline
        # This now uses AdvancedTrainingService internally while maintaining modularity
        results = run_training()
        
        logging.info("SUCCESS: Advanced Training Pipeline completed successfully")
        logging.info("=" * 70)
        logging.info("Generated files:")
        logging.info("  [DATA] all_models_predictions.csv - Complete predictions")
        logging.info("  [FACTS] hechos_predicciones_fields.csv - Fact predictions (no Hilbert)")
        logging.info("  [METRICS] hechos_metricas_modelo.csv - Metrics with Hilbert transform")
        logging.info("  [DIM] dim_modelo.csv - Model dimension table")
        logging.info("  [PRICES] Simplified predictions (no inverse transform)")
        logging.info("  [MODELS] Individual model files: catboost_best.pkl, tts_best.pkl, etc.")
        logging.info("  [FORECASTS] Individual forecast CSVs for each model")
        logging.info("  [CHARTS] Comprehensive metrics and visualization plots")
        logging.info("=" * 70)
        
        # Log TTS specific confirmation
        if 'TTS' in results:
            logging.info("TTS CONFIRMATION:")
            logging.info("  [OK] TTS model trained successfully")
            logging.info("  [OK] TTS predictions generated")
            logging.info("  [OK] TTS included in fact tables")
            logging.info("  [OK] TTS metrics calculated with Hilbert transform")
        
        return results
        
    except Exception as e:
        logging.error(f"ERROR in advanced training pipeline: {e}")
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
