from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Type, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from sp500_analysis.config.settings import settings


class TemporalModelTrainer:
    """Temporal model trainer implementing step_7_0_train_models.py logic."""

    def __init__(self, cv_splits: int = 3, gap: int = 20, random_seed: int = 42) -> None:
        self.cv_splits = cv_splits
        self.gap = gap
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def train_temporal_model(
        self, 
        model_cls: Type, 
        X: pd.DataFrame, 
        y: pd.Series, 
        params: Dict[str, Any],
        scaling_required: bool = None
    ) -> Any:
        """
        Train a model with temporal considerations.
        
        Args:
            model_cls: Model class to instantiate
            X: Features
            y: Target
            params: Best hyperparameters
            scaling_required: Whether to apply scaling (auto-detect if None)
            
        Returns:
            Trained model
        """
        if scaling_required is None:
            scaling_required = self._requires_scaling(model_cls.__name__)
            
        # Apply scaling if required
        if scaling_required:
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X), 
                columns=X.columns, 
                index=X.index
            )
            logging.info(f"Applied scaling for {model_cls.__name__}")
        else:
            X_scaled = X.copy()
            scaler = None

        # Handle special cases for different model types
        if 'LSTM' in model_cls.__name__:
            model = self._train_lstm_model(model_cls, X_scaled, y, params)
        elif 'TTS' in model_cls.__name__:
            model = self._train_tts_model(model_cls, X_scaled, y, params)
        else:
            # Standard model training
            model = model_cls(**params)
            model.fit(X_scaled, y)
            
        # Store scaler for later use
        if hasattr(model, '__dict__'):
            model._scaler = scaler
            
        logging.info(f"✅ {model_cls.__name__} trained successfully")
        return model

    def local_refinement(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_days: int = 225
    ) -> Any:
        """
        Perform local refinement on the most recent data.
        
        Args:
            model: Trained model
            X: Full feature set
            y: Full target
            n_days: Number of recent days to use for refinement
            
        Returns:
            Refined model
        """
        try:
            # Get the most recent n_days of data
            if len(X) > n_days:
                X_recent = X.tail(n_days).copy()
                y_recent = y.tail(n_days).copy()
                logging.info(f"Local refinement using last {n_days} days: {X_recent.shape}")
            else:
                X_recent = X.copy()
                y_recent = y.copy()
                logging.info(f"Using all available data for refinement: {X_recent.shape}")

            # Apply same scaling if model was scaled
            if hasattr(model, '_scaler') and model._scaler is not None:
                X_recent_scaled = pd.DataFrame(
                    model._scaler.transform(X_recent),
                    columns=X_recent.columns,
                    index=X_recent.index
                )
            else:
                X_recent_scaled = X_recent

            # Split recent data for refinement (80/20)
            split_idx = int(len(X_recent_scaled) * 0.8)
            X_train_ref = X_recent_scaled.iloc[:split_idx]
            y_train_ref = y_recent.iloc[:split_idx]
            X_val_ref = X_recent_scaled.iloc[split_idx:]
            y_val_ref = y_recent.iloc[split_idx:]

            if len(X_train_ref) > 5:  # Minimum data for refinement
                # Create a new model instance with same parameters
                if hasattr(model, 'get_params'):
                    # Sklearn-style models
                    refined_model = model.__class__(**model.get_params())
                else:
                    # Custom models - try to copy configuration
                    refined_model = model.__class__()
                    
                refined_model.fit(X_train_ref, y_train_ref)
                
                # Copy scaler to refined model
                if hasattr(model, '_scaler'):
                    refined_model._scaler = model._scaler
                
                # Validate refinement
                val_pred = refined_model.predict(X_val_ref)
                val_rmse = np.sqrt(mean_squared_error(y_val_ref, val_pred))
                logging.info(f"Local refinement validation RMSE: {val_rmse:.6f}")
                
                return refined_model
            else:
                logging.warning("Not enough data for local refinement, returning original model")
                return model
                
        except Exception as e:
            logging.error(f"Local refinement failed: {e}, returning original model")
            return model

    def _train_lstm_model(
        self, 
        model_cls: Type, 
        X: pd.DataFrame, 
        y: pd.Series, 
        params: Dict[str, Any]
    ) -> Any:
        """Special handling for LSTM models."""
        try:
            # Create sequences for LSTM
            sequence_length = params.get('sequence_length', 10)
            X_sequences, y_sequences = self._create_sequences(X, y, sequence_length)
            
            if len(X_sequences) == 0:
                raise ValueError("Not enough data to create LSTM sequences")
            
            # Create and train LSTM model
            model = model_cls(**params)
            model.fit(X_sequences, y_sequences)
            
            # Store sequence length for prediction
            model._sequence_length = sequence_length
            
            return model
            
        except Exception as e:
            logging.error(f"LSTM training failed: {e}")
            raise

    def _create_sequences(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        sequence_length: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        if len(X) < sequence_length:
            return np.array([]), np.array([])
            
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X.iloc[i-sequence_length:i].values)
            y_sequences.append(y.iloc[i])
            
        return np.array(X_sequences), np.array(y_sequences)

    def _train_tts_model(
        self, 
        model_cls: Type, 
        X: pd.DataFrame, 
        y: pd.Series, 
        params: Dict[str, Any]
    ) -> Any:
        """Special handling for TTS (Transformer Time Series) models."""
        try:
            # Create and train TTS model
            model = model_cls(**params)
            
            # Split data for validation during training
            split_idx = int(len(X) * 0.8)
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            X_val = X.iloc[split_idx:]
            y_val = y.iloc[split_idx:]
            
            if len(X_val) > 0:
                model.fit(X_train, y_train, X_val, y_val)
            else:
                model.fit(X_train, y_train)
            
            return model
            
        except Exception as e:
            logging.error(f"TTS training failed: {e}")
            raise

    def _requires_scaling(self, model_name: str) -> bool:
        """Determine if a model requires feature scaling."""
        scaling_required_models = {
            "MLP", "MLPRegressor", "SVM", "SVR", "LSTM", "TTS"
        }
        return any(req_model in model_name for req_model in scaling_required_models) 