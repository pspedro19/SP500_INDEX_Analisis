from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import optuna
from typing import Type, Dict, Any, Callable
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import uniform, randint
from math import sqrt

from sp500_analysis.config.settings import settings


class HyperparameterOptimizer:
    """Hyperparameter optimizer implementing step_7_0_train_models.py logic with RandomSearch + Optuna."""

    def __init__(
        self, 
        random_search_trials: int = 3,
        optuna_trials: int = 2,
        cv_splits: int = 3,
        gap: int = 20,
        random_seed: int = 42
    ) -> None:
        self.random_search_trials = random_search_trials
        self.optuna_trials = optuna_trials
        self.cv_splits = cv_splits
        self.gap = gap
        self.random_seed = random_seed
        
        # Configure optuna to be less verbose
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def optimize_temporal(
        self, 
        model_cls: Type, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame, 
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using RandomSearch + Optuna approach.
        
        Args:
            model_cls: Model class to optimize
            X_train: Training features (Zone A)
            y_train: Training target (Zone A)
            X_val: Validation features (Zone B)
            y_val: Validation target (Zone B)
            
        Returns:
            Best hyperparameters
        """
        model_name = model_cls.__name__.replace('Wrapper', '')
        logging.info(f"🔍 Optimizing hyperparameters for {model_name}")
        
        try:
            # Step 1: RandomizedSearch for initial exploration
            random_params = self._run_randomized_search(
                model_cls, X_train, y_train, model_name
            )
            logging.info(f"RandomSearch completed for {model_name}")
            
            # Step 2: Optuna refinement based on RandomSearch results
            optuna_params = self._run_optuna_optimization(
                model_cls, X_train, y_train, X_val, y_val, model_name, random_params
            )
            logging.info(f"Optuna optimization completed for {model_name}")
            
            return optuna_params
            
        except Exception as e:
            logging.error(f"Optimization failed for {model_name}: {e}")
            # Return default parameters
            return self._get_default_params(model_name)

    def _run_randomized_search(
        self, 
        model_cls: Type, 
        X: pd.DataFrame, 
        y: pd.Series, 
        model_name: str
    ) -> Dict[str, Any]:
        """Run RandomizedSearch for initial parameter exploration."""
        
        # Define parameter distributions
        param_distributions = self._get_random_search_distributions(model_name)
        
        if not param_distributions:
            return self._get_default_params(model_name)
        
        try:
            # Configure TimeSeriesSplit for temporal validation
            tscv = TimeSeriesSplit(n_splits=self.cv_splits, gap=self.gap)
            
            # RMSE scorer (negative because sklearn maximizes)
            rmse_scorer = make_scorer(
                lambda y_true, y_pred: -sqrt(mean_squared_error(y_true, y_pred)),
                greater_is_better=True
            )
            
            # RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=model_cls(),
                param_distributions=param_distributions,
                n_iter=self.random_search_trials,
                cv=tscv,
                scoring=rmse_scorer,
                random_state=self.random_seed,
                n_jobs=1  # Single job to avoid conflicts
            )
            
            random_search.fit(X, y)
            
            best_params = random_search.best_params_
            best_score = -random_search.best_score_  # Convert back to positive RMSE
            
            logging.info(f"RandomSearch {model_name} - Best RMSE: {best_score:.6f}")
            return best_params
            
        except Exception as e:
            logging.error(f"RandomizedSearch failed for {model_name}: {e}")
            return self._get_default_params(model_name)

    def _run_optuna_optimization(
        self,
        model_cls: Type,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_name: str,
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run Optuna optimization for parameter refinement."""
        
        try:
            # Create study
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=self.random_seed)
            )
            
            # Define objective function
            def objective(trial):
                return self._get_objective_function(model_name)(
                    trial, X_train, y_train, X_val, y_val, base_params
                )
            
            # Optimize
            study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)
            
            best_params = study.best_params
            best_value = study.best_value
            
            logging.info(f"Optuna {model_name} - Best RMSE: {best_value:.6f}")
            return best_params
            
        except Exception as e:
            logging.error(f"Optuna optimization failed for {model_name}: {e}")
            return base_params

    def _get_random_search_distributions(self, model_name: str) -> Dict[str, Any]:
        """Get parameter distributions for RandomizedSearch."""
        
        distributions = {
            'CatBoost': {
                'depth': randint(4, 11),
                'learning_rate': uniform(0.01, 0.29),
                'iterations': randint(100, 501),
                'l2_leaf_reg': uniform(1, 10),
                'verbose': [False]
            },
            'LightGBM': {
                'num_leaves': randint(31, 129),
                'learning_rate': uniform(0.01, 0.29),
                'n_estimators': randint(100, 501),
                'feature_fraction': uniform(0.6, 0.4),
                'bagging_fraction': uniform(0.6, 0.4),
                'verbose': [-1]
            },
            'XGBoost': {
                'max_depth': randint(3, 11),
                'learning_rate': uniform(0.01, 0.29),
                'n_estimators': randint(100, 501),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'verbosity': [0]
            },
            'MLP': {
                'hidden_layer_sizes': [(50,), (100,), (150,), (200,), (100, 50), (150, 75)],
                'learning_rate_init': uniform(0.0001, 0.0099),
                'alpha': uniform(0.0001, 0.01),
                'max_iter': [300, 500, 800]
            },
            'SVM': {
                'C': uniform(0.1, 9.9),
                'epsilon': uniform(0.001, 0.999),
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        return distributions.get(model_name, {})

    def _get_objective_function(self, model_name: str) -> Callable:
        """Get the appropriate objective function for Optuna optimization."""
        
        objective_functions = {
            'CatBoost': self._objective_catboost,
            'LightGBM': self._objective_lightgbm,
            'XGBoost': self._objective_xgboost,
            'MLP': self._objective_mlp,
            'SVM': self._objective_svm,
            'LSTM': self._objective_lstm,
            'TTS': self._objective_tts
        }
        
        return objective_functions.get(model_name, self._objective_default)

    def _objective_catboost(self, trial, X_train, y_train, X_val, y_val, base_params):
        """CatBoost objective function for Optuna."""
        from sp500_analysis.infrastructure.models.wrappers.catboost_wrapper import CatBoostWrapper
        
        params = {
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'iterations': trial.suggest_int('iterations', 100, 500),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'verbose': False
        }
        
        model = CatBoostWrapper(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        return sqrt(mean_squared_error(y_val, y_pred))

    def _objective_lightgbm(self, trial, X_train, y_train, X_val, y_val, base_params):
        """LightGBM objective function for Optuna."""
        from sp500_analysis.infrastructure.models.wrappers.lightgbm_wrapper import LightGBMWrapper
        
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 31, 128),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'verbose': -1
        }
        
        model = LightGBMWrapper(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        return sqrt(mean_squared_error(y_val, y_pred))

    def _objective_xgboost(self, trial, X_train, y_train, X_val, y_val, base_params):
        """XGBoost objective function for Optuna."""
        from sp500_analysis.infrastructure.models.wrappers.xgboost_wrapper import XGBoostWrapper
        
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'verbosity': 0
        }
        
        model = XGBoostWrapper(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        return sqrt(mean_squared_error(y_val, y_pred))

    def _objective_mlp(self, trial, X_train, y_train, X_val, y_val, base_params):
        """MLP objective function for Optuna."""
        from sp500_analysis.infrastructure.models.wrappers.mlp_wrapper import MLPWrapper
        from sklearn.preprocessing import StandardScaler
        
        # Scale features for MLP
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        params = {
            'hidden_layer_sizes': (trial.suggest_int('hidden_units', 50, 200),),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
            'alpha': trial.suggest_float('alpha', 1e-4, 1e-2, log=True),
            'max_iter': 300
        }
        
        model = MLPWrapper(**params)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        
        return sqrt(mean_squared_error(y_val, y_pred))

    def _objective_svm(self, trial, X_train, y_train, X_val, y_val, base_params):
        """SVM objective function for Optuna."""
        from sp500_analysis.infrastructure.models.wrappers.svm_wrapper import SVMWrapper
        from sklearn.preprocessing import StandardScaler
        
        # Scale features for SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.001, 1.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear'])
        }
        
        model = SVMWrapper(**params)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        
        return sqrt(mean_squared_error(y_val, y_pred))

    def _objective_lstm(self, trial, X_train, y_train, X_val, y_val, base_params):
        """LSTM objective function for Optuna."""
        try:
            from sp500_analysis.infrastructure.models.wrappers.lstm_wrapper import LSTMWrapper
            from sklearn.preprocessing import StandardScaler
            
            # Scale features for LSTM
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            params = {
                'units': trial.suggest_int('units', 32, 128),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.3),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'sequence_length': trial.suggest_int('sequence_length', 10, 20),
                'epochs': trial.suggest_int('epochs', 20, 50),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32]),
                'patience': trial.suggest_int('patience', 5, 10)
            }
            
            model = LSTMWrapper(**params)
            model.fit(pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index), y_train)
            y_pred = model.predict(pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index))
            
            return sqrt(mean_squared_error(y_val, y_pred))
            
        except Exception as e:
            logging.warning(f"LSTM optimization failed: {e}")
            return float('inf')

    def _objective_tts(self, trial, X_train, y_train, X_val, y_val, base_params):
        """TTS (Transformer Time Series) objective function for Optuna."""
        try:
            from sp500_analysis.infrastructure.models.wrappers.tts_wrapper import TTSWrapper
            
            params = {
                'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
                'nhead': trial.suggest_categorical('nhead', [4, 8, 16]),
                'num_encoder_layers': trial.suggest_int('num_encoder_layers', 2, 4),
                'dim_feedforward': trial.suggest_categorical('dim_feedforward', [128, 256]),
                'dropout': trial.suggest_float('dropout', 0.0, 0.3),
                'sequence_length': trial.suggest_int('sequence_length', 15, 30),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32]),
                'epochs': trial.suggest_int('epochs', 30, 80),  # Reduced for optimization
                'patience': trial.suggest_int('patience', 5, 10)
            }
            
            # Ensure nhead divides d_model
            if params['d_model'] % params['nhead'] != 0:
                params['nhead'] = 4  # Safe fallback
            
            model = TTSWrapper(**params)
            model.fit(X_train, y_train, X_val, y_val)
            y_pred = model.predict(X_val)
            
            return sqrt(mean_squared_error(y_val, y_pred))
            
        except Exception as e:
            logging.warning(f"TTS optimization failed: {e}")
            return float('inf')

    def _objective_default(self, trial, X_train, y_train, X_val, y_val, base_params):
        """Default objective function."""
        return float('inf')

    def _get_default_params(self, model_name: str) -> Dict[str, Any]:
        """Get default parameters for models."""
        
        defaults = {
            'CatBoost': {
                'depth': 6,
                'learning_rate': 0.1,
                'iterations': 100,
                'verbose': False
            },
            'LightGBM': {
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'verbose': -1
            },
            'XGBoost': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'verbosity': 0
            },
            'MLP': {
                'hidden_layer_sizes': (100,),
                'learning_rate_init': 0.001,
                'max_iter': 300
            },
            'SVM': {
                'C': 1.0,
                'epsilon': 0.1,
                'kernel': 'rbf'
            },
            'LSTM': {
                'units': 50,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'sequence_length': 10
            },
            'TTS': {
                'd_model': 64,
                'nhead': 8,
                'num_encoder_layers': 3,
                'dim_feedforward': 256,
                'dropout': 0.1,
                'sequence_length': 30,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'patience': 10
            }
        }
        
        return defaults.get(model_name, {}) 