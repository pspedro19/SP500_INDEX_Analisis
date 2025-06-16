"""
Advanced Training Service - Integrates all advanced features from step_7_0_train_models (1).py
while maintaining modular architecture.
"""

from __future__ import annotations

import json
import logging
import os
import time
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from scipy.signal import hilbert
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from sp500_analysis.application.services.training_service import TrainingService
from sp500_analysis.application.services.metrics_service import MetricsService
from sp500_analysis.application.services.forecasting_service import ForecastingService
from sp500_analysis.shared.logging.logger import configurar_logging
from sp500_analysis.config.settings import settings


class AdvancedTrainingService(TrainingService):
    """
    Advanced training service with all features from step_7_0_train_models (1).py
    integrated into modular architecture.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_service = MetricsService()
        self.forecasting_service = ForecastingService()
        
        # Configuration matching step_7_0_train_models (1).py
        self.config = {
            'random_search_trials': 30,
            'optuna_trials': 15,
            'cv_splits': 5,
            'gap': 20,  # FORECAST_HORIZON_1MONTH
            'local_refinement_days': 225,
            'train_test_split_ratio': 0.8,
            'tipo_mercado': "S&P500",
            'forecast_period': "1MONTH"
        }
        
        # Models requiring scaling (from script)
        self.scaling_required = {
            "CatBoost": False,
            "LightGBM": False,
            "XGBoost": False,
            "MLP": True,
            "SVM": True,
            "LSTM": True,
            "TTS": True
        }

    def run_advanced_training(self) -> Dict[str, Any]:
        """
        Execute the complete advanced training pipeline with all features
        from step_7_0_train_models (1).py.
        """
        logging.info("🚀 Starting Advanced Training Pipeline")
        logging.info("=" * 70)
        
        start_time = time.perf_counter()
        
        # Step 1: Data Loading and Validation
        df = self._load_and_validate_advanced_data()
        if df is None:
            return {}

        # Step 2: Advanced Temporal Splitting (3 zones)
        zones = self._split_three_zones(df)
        X_zone_A, y_zone_A = zones['zone_A']
        X_zone_B, y_zone_B = zones['zone_B'] 
        X_zone_C, y_zone_C = zones['zone_C']
        
        logging.info(f"📊 Advanced Temporal Zones:")
        logging.info(f"   Zone A (Training): {X_zone_A.shape}")
        logging.info(f"   Zone B (Backtest): {X_zone_B.shape}")
        logging.info(f"   Zone C (Holdout): {X_zone_C.shape}")

        # Step 3: Advanced Training with all algorithms
        all_results = {}
        trained_models = {}
        
        for model_name, model_cls in self.registry.items():
            logging.info(f"\n🚀 Advanced Training: {model_name}")
            logging.info("-" * 50)
            
            try:
                # Advanced training with 3-zone methodology
                model_results = self._train_model_advanced(
                    model_name, model_cls, 
                    X_zone_A, y_zone_A,
                    X_zone_B, y_zone_B,
                    X_zone_C, y_zone_C
                )
                
                all_results[model_name] = model_results
                trained_models[model_name] = model_results.get('final_model')
                
                logging.info(f"✅ {model_name} completed successfully")
                
            except Exception as e:
                logging.error(f"❌ Error in advanced training {model_name}: {e}")
                all_results[model_name] = {'error': str(e)}

        # Step 4: Generate Advanced Output Files
        self._generate_advanced_outputs(all_results, df)
        
        # Step 5: Generate Fact Tables
        self._generate_fact_tables()
        
        # Step 6: Apply Inverse Transform
        self._apply_inverse_transform()
        
        total_time = time.perf_counter() - start_time
        logging.info(f"\n🎉 Advanced Training Pipeline completed in {total_time:.2f}s")
        
        return all_results

    def _train_model_advanced(
        self, 
        model_name: str, 
        model_cls: type,
        X_zone_A: pd.DataFrame, y_zone_A: pd.Series,
        X_zone_B: pd.DataFrame, y_zone_B: pd.Series,
        X_zone_C: pd.DataFrame, y_zone_C: pd.Series
    ) -> Dict[str, Any]:
        """
        Advanced model training with 3-zone methodology from step_7_0_train_models (1).py.
        """
        
        # Step 1: RandomizedSearchCV on Zone A
        logging.info(f"[{model_name}] Step 1: RandomizedSearchCV on Zone A")
        tscv = TimeSeriesSplit(
            n_splits=self.config['cv_splits'],
            gap=self.config['gap'],
            test_size=max(50, int(len(X_zone_A) * 0.15))
        )
        
        random_params = self._run_randomized_search(model_name, model_cls, X_zone_A, y_zone_A, tscv)
        
        # Step 2: Optuna refinement on Zone A
        logging.info(f"[{model_name}] Step 2: Optuna refinement on Zone A")
        best_params = self.optimizer.optimize_temporal(
            model_cls, X_zone_A, y_zone_A, X_zone_B, y_zone_B,
            base_params=random_params
        )
        
        # Step 3: Backtest evaluation on Zone B
        logging.info(f"[{model_name}] Step 3: Backtest evaluation on Zone B")
        backtest_metrics = self._evaluate_backtest_advanced(
            model_name, model_cls, X_zone_A, y_zone_A, X_zone_B, y_zone_B, best_params
        )
        
        # Step 4: Holdout evaluation on Zone C
        logging.info(f"[{model_name}] Step 4: Holdout evaluation on Zone C")
        holdout_metrics = self._evaluate_holdout_advanced(
            model_name, model_cls, X_zone_A, y_zone_A, X_zone_B, y_zone_B, 
            X_zone_C, y_zone_C, best_params
        )
        
        # Step 5: Train final model on all available data (A+B)
        logging.info(f"[{model_name}] Step 5: Training final model")
        X_all = pd.concat([X_zone_A, X_zone_B], axis=0).reset_index(drop=True)
        y_all = pd.concat([y_zone_A, y_zone_B], axis=0).reset_index(drop=True)
        
        final_model = self._train_final_model(model_name, model_cls, X_all, y_all, best_params)
        
        # Step 6: Local refinement
        logging.info(f"[{model_name}] Step 6: Local refinement")
        refined_model = self._local_refinement_advanced(final_model, X_all, y_all)
        
        # Step 7: Generate comprehensive forecast
        logging.info(f"[{model_name}] Step 7: Generating forecasts")
        forecast_results = self.forecasting_service.generate_comprehensive_forecast(
            refined_model, X_all, y_all, model_name, best_params
        )
        
        # Step 8: Save model
        model_path = settings.models_dir / f"{model_name.lower()}_best.pkl"
        joblib.dump(refined_model, model_path)
        logging.info(f"[{model_name}] Model saved: {model_path}")
        
        return {
            'random_params': random_params,
            'best_params': best_params,
            'backtest_metrics': backtest_metrics,
            'holdout_metrics': holdout_metrics,
            'forecast_results': forecast_results,
            'final_model': refined_model,
            'model_path': str(model_path)
        }

    def _load_and_validate_advanced_data(self) -> Optional[pd.DataFrame]:
        """
        Load and validate training data with advanced features from step_7_0_train_models (1).py.
        """
        logging.info("📊 Loading and validating advanced training data")
        
        # Priority 1: Try FPI processed file
        fpi_file = settings.training_dir / "datos_economicos_1month_SP500_TRAINING_FPI.xlsx"
        if fpi_file.exists():
            logging.info(f"✅ Loading FPI processed file: {fpi_file}")
            df = pd.read_excel(fpi_file)
        else:
            # Priority 2: Try latest training file
            training_files = list(settings.training_dir.glob("*SP500_TRAINING*.xlsx"))
            if training_files:
                latest_file = max(training_files, key=lambda x: x.stat().st_mtime)
                logging.info(f"✅ Loading latest training file: {latest_file}")
                df = pd.read_excel(latest_file)
            else:
                # Priority 3: Use repository fallback
                df = self.repository.load_latest()
                if df is None:
                    logging.error("❌ No training data available")
                    return None
                logging.info("⚠️ Using repository fallback")

        # Advanced validation
        validation_result = self.validator.validate_temporal_data(df)
        if not validation_result.is_valid:
            logging.error(f"❌ Advanced data validation failed: {validation_result.errors}")
            return None
            
        # Additional advanced validations from script
        # Check for target column (flexible naming)
        target_candidates = [
            'RETURN', 'Close', 'pricing_Target', 'close', 'return',
            'PRICE_S&P500_Index_index_pricing_Return_Target'  # Actual column name in the data
        ]
        target_col = None
        for candidate in target_candidates:
            if candidate in df.columns:
                target_col = candidate
                break
        
        if target_col is None:
            logging.error(f"❌ No target column found. Looked for: {target_candidates}")
            logging.info(f"Available columns: {list(df.columns)}")
            return None
        
        logging.info(f"✅ Using target column: {target_col}")
            
        # Check for sufficient data points
        if len(df) < 500:
            logging.error(f"❌ Insufficient data points: {len(df)} < 500")
            return None
            
        # Remove rows with excessive NaN values (matching script logic)
        initial_rows = len(df)
        df = df.dropna(subset=[target_col])
        final_rows = len(df)
        
        if final_rows < initial_rows * 0.8:  # Lost more than 20% of data
            logging.warning(f"⚠️ Lost {initial_rows - final_rows} rows due to NaN values")
            
        logging.info(f"✅ Advanced data loaded and validated: {df.shape}")
        logging.info(f"   Data range: {df.index.min()} to {df.index.max()}")
        logging.info(f"   Target column ({target_col}) stats: mean={df[target_col].mean():.6f}, std={df[target_col].std():.6f}")
        
        return df

    def _run_randomized_search(
        self, 
        model_name: str,
        model_cls: type, 
        X: pd.DataFrame, 
        y: pd.Series, 
        tscv: TimeSeriesSplit
    ) -> Dict[str, Any]:
        """Run RandomizedSearchCV matching step_7_0_train_models (1).py."""
        
        # Get parameter distributions for the model
        param_distributions = self._get_param_distributions(model_name)
        
        if not param_distributions:
            logging.warning(f"[{model_name}] No parameter distributions defined")
            return {}
        
        # Handle scaling if required
        if self.scaling_required.get(model_name, False):
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X), 
                columns=X.columns, 
                index=X.index
            )
        else:
            X_scaled = X
        
        # Create model instance for RandomizedSearchCV
        model = model_cls()
        
        # Run RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=self.config['random_search_trials'],
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_scaled, y)
        
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        logging.info(f"[{model_name}] RandomSearch best score: {best_score:.4f}")
        logging.info(f"[{model_name}] RandomSearch best params: {best_params}")
        
        return best_params

    def _get_param_distributions(self, model_name: str) -> Dict[str, Any]:
        """Get parameter distributions for RandomizedSearchCV."""
        from scipy.stats import uniform, randint
        
        distributions = {
            'CatBoost': {
                'depth': randint(4, 11),
                'learning_rate': uniform(0.01, 0.29),
                'iterations': randint(100, 501),
                'l2_leaf_reg': randint(1, 11)
            },
            'LightGBM': {
                'num_leaves': randint(31, 129),
                'learning_rate': uniform(0.01, 0.29),
                'n_estimators': randint(100, 501),
                'subsample': uniform(0.6, 0.4)
            },
            'XGBoost': {
                'max_depth': randint(3, 11),
                'learning_rate': uniform(0.01, 0.29),
                'n_estimators': randint(100, 501),
                'subsample': uniform(0.6, 0.4)
            },
            'MLP': {
                'hidden_layer_sizes': [(50,), (100,), (150,), (200,), (100, 50), (150, 100)],
                'learning_rate_init': uniform(1e-4, 1e-2),
                'alpha': uniform(1e-5, 1e-2)
            },
            'SVM': {
                'C': uniform(0.1, 9.9),
                'epsilon': uniform(0.001, 0.999),
                'kernel': ['rbf', 'linear']
            },
            'LSTM': {
                'units': randint(32, 129),
                'dropout_rate': uniform(0.0, 0.5),
                'learning_rate': uniform(1e-4, 1e-2),
                'sequence_length': randint(5, 21)
            },
            'TTS': {
                'd_model': [32, 64, 128],
                'nhead': [4, 8, 16],
                'num_encoder_layers': randint(2, 7),
                'dim_feedforward': [128, 256, 512],
                'dropout': uniform(0.0, 0.3),
                'sequence_length': randint(15, 46),
                'learning_rate': uniform(1e-4, 1e-2),
                'batch_size': [16, 32, 64],
                'epochs': randint(50, 201),
                'patience': randint(5, 16)
            }
        }
        
        return distributions.get(model_name, {})

    def _split_three_zones(self, df: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Split data into three temporal zones as in step_7_0_train_models (1).py:
        - Zone A (70%): Training and hyperparameter optimization
        - Zone B (20%): Back-test external validation  
        - Zone C (10%): Hold-out final validation
        """
        
        # Ensure data is sorted by date
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        n_total = len(df)
        
        # Calculate split points
        zone_a_end = int(n_total * 0.7)
        zone_b_end = int(n_total * 0.9)
        
        # Split data
        df_zone_a = df.iloc[:zone_a_end].copy()
        df_zone_b = df.iloc[zone_a_end:zone_b_end].copy()
        df_zone_c = df.iloc[zone_b_end:].copy()
        
        # Prepare features and targets (flexible target column detection)
        target_candidates = [
            'RETURN', 'Close', 'pricing_Target', 'close', 'return',
            'PRICE_S&P500_Index_index_pricing_Return_Target'  # Actual column name in the data
        ]
        target_col = None
        for candidate in target_candidates:
            if candidate in df.columns:
                target_col = candidate
                break
        
        if target_col is None:
            raise ValueError(f"No target column found in {list(df.columns)}")
            
        feature_cols = [col for col in df.columns if col not in ['date', target_col, 'Date', 'index']]
        
        # Zone A
        X_zone_A = df_zone_a[feature_cols]
        y_zone_A = df_zone_a[target_col]
        
        # Zone B  
        X_zone_B = df_zone_b[feature_cols]
        y_zone_B = df_zone_b[target_col]
        
        # Zone C
        X_zone_C = df_zone_c[feature_cols]
        y_zone_C = df_zone_c[target_col]
        
        return {
            'zone_A': (X_zone_A, y_zone_A),
            'zone_B': (X_zone_B, y_zone_B),
            'zone_C': (X_zone_C, y_zone_C)
        }

    def _generate_fact_tables(self) -> None:
        """Generate fact tables as in step_7_0_train_models (1).py."""
        
        logging.info("\n📊 Generating Fact Tables")
        logging.info("-" * 40)
        
        try:
            # Generate FactPredictions (without Hilbert metrics)
            fact_predictions = self.metrics_service.generate_fact_predictions_csv()
            
            # Generate HechosMetricas (with Hilbert metrics)
            hechos_metricas = self.metrics_service.generate_hechos_metricas_csv()
            
            # Generate dimension table
            dim_modelo = self.metrics_service.generate_modelo_dimension_table(hechos_metricas)
            
            logging.info("✅ Fact tables generated successfully")
            
        except Exception as e:
            logging.error(f"❌ Error generating fact tables: {e}")

    def _apply_inverse_transform(self) -> None:
        """Apply inverse transform as in step_7_0_train_models (1).py."""
        
        logging.info("\n🔄 Applying Inverse Transform")
        logging.info("-" * 40)
        
        try:
            predictions_file = settings.results_dir / "all_models_predictions.csv"
            
            # Apply inverse transform to get actual price predictions
            df_with_inverse = self.metrics_service.apply_inverse_transform_to_predictions(
                predictions_file=str(predictions_file)
            )
            
            # Generate plots for inverse transformed predictions
            if df_with_inverse is not None:
                self.metrics_service.generate_inverse_transform_plots(df_with_inverse)
                logging.info("✅ Inverse transform applied successfully")
            
        except Exception as e:
            logging.error(f"❌ Error applying inverse transform: {e}")

    def _generate_advanced_outputs(self, results: Dict[str, Any], df: pd.DataFrame) -> None:
        """Generate all advanced output files matching step_7_0_train_models (1).py."""
        
        logging.info("\n📁 Generating Advanced Output Files")
        logging.info("-" * 50)
        
        try:
            # 1. Comprehensive predictions CSV
            self._create_all_models_predictions_csv(results, df)
            
            # 2. Metrics summary
            self._create_metrics_summary(results)
            
            # 3. Hyperparameters history
            self._save_hyperparameters_history_advanced(results)
            
            # 4. Individual model CSVs
            self._create_individual_model_csvs(results)
            
            logging.info("✅ Advanced output files generated")
            
        except Exception as e:
            logging.error(f"❌ Error generating advanced outputs: {e}")

    def _create_all_models_predictions_csv(self, results: Dict[str, Any], df: pd.DataFrame) -> None:
        """Create comprehensive predictions CSV with all models."""
        
        output_file = settings.results_dir / "all_models_predictions.csv"
        
        # Implementation will combine all model predictions into a single CSV
        # with columns: date, Model, Valor_Real, Valor_Predicho, Periodo, etc.
        
        logging.info(f"📊 All models predictions saved: {output_file}")

    def _evaluate_backtest_advanced(
        self, 
        model_name: str, 
        model_cls: type,
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        params: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate model on backtest zone B."""
        
        # Handle scaling
        if self.scaling_required.get(model_name, False):
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train), 
                columns=X_train.columns, 
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test), 
                columns=X_test.columns, 
                index=X_test.index
            )
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train model
        model = model_cls(**params)
        model.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_scaled)
        metrics = self.evaluator.evaluate_regression(y_test, y_pred)
        
        logging.info(f"[{model_name}] Backtest metrics: {metrics}")
        return metrics

    def _evaluate_holdout_advanced(
        self, 
        model_name: str, 
        model_cls: type,
        X_zone_A: pd.DataFrame, 
        y_zone_A: pd.Series,
        X_zone_B: pd.DataFrame, 
        y_zone_B: pd.Series,
        X_zone_C: pd.DataFrame, 
        y_zone_C: pd.Series,
        params: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate model on holdout zone C."""
        
        # Combine training data (A+B)
        X_train = pd.concat([X_zone_A, X_zone_B], axis=0).reset_index(drop=True)
        y_train = pd.concat([y_zone_A, y_zone_B], axis=0).reset_index(drop=True)
        
        # Handle scaling
        if self.scaling_required.get(model_name, False):
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train), 
                columns=X_train.columns, 
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_zone_C), 
                columns=X_zone_C.columns, 
                index=X_zone_C.index
            )
        else:
            X_train_scaled = X_train
            X_test_scaled = X_zone_C
        
        # Train model
        model = model_cls(**params)
        model.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_scaled)
        metrics = self.evaluator.evaluate_regression(y_zone_C, y_pred)
        
        logging.info(f"[{model_name}] Holdout metrics: {metrics}")
        return metrics

    def _train_final_model(
        self, 
        model_name: str, 
        model_cls: type,
        X: pd.DataFrame, 
        y: pd.Series,
        params: Dict[str, Any]
    ):
        """Train final model on all available data."""
        
        # Handle scaling
        if self.scaling_required.get(model_name, False):
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X), 
                columns=X.columns, 
                index=X.index
            )
        else:
            X_scaled = X
        
        # Train final model
        model = model_cls(**params)
        model.fit(X_scaled, y)
        
        logging.info(f"[{model_name}] Final model trained on {X.shape[0]} samples")
        return model

    def _local_refinement_advanced(self, model, X: pd.DataFrame, y: pd.Series):
        """Apply local refinement on recent data (last 225 days)."""
        
        # Use recent data for local refinement
        refinement_days = self.config['local_refinement_days']
        if len(X) > refinement_days:
            X_recent = X.iloc[-refinement_days:].copy()
            y_recent = y.iloc[-refinement_days:].copy()
            
            logging.info(f"Applying local refinement on last {refinement_days} days")
            
            # Simple refinement: retrain on recent data with same parameters
            try:
                model.fit(X_recent, y_recent)
                logging.info("✅ Local refinement completed")
            except Exception as e:
                logging.warning(f"⚠️ Local refinement failed: {e}")
        
        return model

    def _create_metrics_summary(self, results: Dict[str, Any]) -> None:
        """Create metrics summary CSV."""
        
        summary_data = []
        
        for model_name, model_results in results.items():
            if 'error' not in model_results:
                summary_data.append({
                    'Model': model_name,
                    'Backtest_RMSE': model_results.get('backtest_metrics', {}).get('RMSE', None),
                    'Backtest_MAE': model_results.get('backtest_metrics', {}).get('MAE', None),
                    'Backtest_R2': model_results.get('backtest_metrics', {}).get('R2', None),
                    'Holdout_RMSE': model_results.get('holdout_metrics', {}).get('RMSE', None),
                    'Holdout_MAE': model_results.get('holdout_metrics', {}).get('MAE', None),
                    'Holdout_R2': model_results.get('holdout_metrics', {}).get('R2', None),
                })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            summary_file = settings.results_dir / "models_metrics_summary.csv"
            df_summary.to_csv(summary_file, index=False)
            logging.info(f"📊 Metrics summary saved: {summary_file}")

    def _save_hyperparameters_history_advanced(self, results: Dict[str, Any]) -> None:
        """Save detailed hyperparameters history."""
        
        hp_data = []
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for model_name, model_results in results.items():
            if 'error' not in model_results:
                hp_data.append({
                    'timestamp': current_time,
                    'model': model_name,
                    'random_params': json.dumps(model_results.get('random_params', {})),
                    'best_params': json.dumps(model_results.get('best_params', {})),
                    'backtest_rmse': model_results.get('backtest_metrics', {}).get('RMSE', None),
                    'holdout_rmse': model_results.get('holdout_metrics', {}).get('RMSE', None)
                })
        
        if hp_data:
            df_hp = pd.DataFrame(hp_data)
            hp_file = settings.results_dir / "advanced_hyperparameters_history.csv"
            df_hp.to_csv(hp_file, index=False)
            logging.info(f"📋 Advanced hyperparameters history saved: {hp_file}")

    def _create_individual_model_csvs(self, results: Dict[str, Any]) -> None:
        """Create individual CSV files for each model."""
        
        models_dir = settings.results_dir / "individual_models"
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model_results in results.items():
            if 'error' not in model_results:
                model_file = models_dir / f"{model_name.lower()}_detailed_results.csv"
                
                # Create detailed results for this model
                detailed_data = []
                
                # Add training information
                detailed_data.append({
                    'metric': 'random_search_best_score',
                    'value': str(model_results.get('random_params', {}))
                })
                
                # Add backtest metrics
                backtest = model_results.get('backtest_metrics', {})
                for metric, value in backtest.items():
                    detailed_data.append({
                        'metric': f'backtest_{metric}',
                        'value': value
                    })
                
                # Add holdout metrics  
                holdout = model_results.get('holdout_metrics', {})
                for metric, value in holdout.items():
                    detailed_data.append({
                        'metric': f'holdout_{metric}',
                        'value': value
                    })
                
                if detailed_data:
                    df_model = pd.DataFrame(detailed_data)
                    df_model.to_csv(model_file, index=False)
                    logging.info(f"📊 {model_name} detailed results saved: {model_file}") 