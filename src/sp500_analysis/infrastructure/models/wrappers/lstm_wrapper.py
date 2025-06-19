try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pandas as pd
    import logging
    TENSORFLOW_AVAILABLE = True
    
    # Configuración para evitar warnings de GPU
    tf.config.set_visible_devices([], 'GPU')  # Forzar uso de CPU
    
except Exception as e:  # pragma: no cover
    logging.warning(f"TensorFlow not available: {e}")
    Sequential = None
    BaseEstimator = object
    RegressorMixin = object
    TENSORFLOW_AVAILABLE = False


class LSTMWrapper(BaseEstimator, RegressorMixin):
    def __init__(
        self, units: int = 64, dropout_rate: float = 0.2, learning_rate: float = 0.001, 
        sequence_length: int = 10, epochs: int = 50, batch_size: int = 32, **kwargs
    ):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not installed")
            
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Initialize components
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.n_features_ = None

    def _build_model(self, n_features):
        """Build the LSTM model with explicit input shape."""
        try:
            # Clear any existing session
            tf.keras.backend.clear_session()
            
            # Validate input parameters
            if n_features <= 0:
                raise ValueError(f"Invalid number of features: {n_features}")
            if self.sequence_length <= 0:
                raise ValueError(f"Invalid sequence length: {self.sequence_length}")
            if self.units <= 0:
                raise ValueError(f"Invalid units: {self.units}")
            
            # Build model with explicit input shape
            self.model = Sequential([
                LSTM(
                    units=self.units, 
                    input_shape=(self.sequence_length, n_features),
                    return_sequences=False,
                    dropout=self.dropout_rate,
                    recurrent_dropout=0.1
                ),
                Dropout(self.dropout_rate),
                Dense(1, activation='linear')
            ])
            
            # Compile with stable configuration
            optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
            self.model.compile(
                optimizer=optimizer, 
                loss="mse",
                metrics=['mae']
            )
            
            logging.info(f"LSTM model built successfully: input_shape=({self.sequence_length}, {n_features})")
            
        except Exception as e:
            logging.error(f"Error building LSTM model: {e}")
            raise

    def _prepare_sequences(self, X, y=None, fit_scaler=True):
        """Prepare sequences for LSTM training with robust validation."""
        try:
            # Convert to numpy if pandas
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            # Validate input
            if X.shape[0] == 0:
                return np.array([]), np.array([]) if y is not None else None
                
            # Scale features
            if fit_scaler:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
            
            # Validate sufficient data for sequences
            if len(X_scaled) < self.sequence_length:
                logging.warning(f"Insufficient data for sequences: {len(X_scaled)} < {self.sequence_length}")
                return np.array([]), np.array([]) if y is not None else None
            
            # Create sequences
            X_sequences = []
            y_sequences = [] if y is not None else None
            
            for i in range(self.sequence_length, len(X_scaled)):
                X_sequences.append(X_scaled[i-self.sequence_length:i])
                if y is not None:
                    y_sequences.append(y[i])
            
            # Convert to numpy arrays with explicit dtype
            X_sequences = np.array(X_sequences, dtype=np.float32)
            
            if y is not None:
                y_sequences = np.array(y_sequences, dtype=np.float32)
                logging.info(f"LSTM sequences created: X_shape={X_sequences.shape}, y_shape={y_sequences.shape}")
                return X_sequences, y_sequences
            else:
                logging.info(f"LSTM sequences created: X_shape={X_sequences.shape}")
                return X_sequences, None
                
        except Exception as e:
            logging.error(f"Error preparing LSTM sequences: {e}")
            return np.array([]), np.array([]) if y is not None else None

    def fit(self, X, y, **kwargs):
        """Fit the LSTM model with robust error handling."""
        try:
            logging.info(f"Training LSTM with data shape: X={X.shape}, y={y.shape}")
            
            X_seq, y_seq = self._prepare_sequences(X, y, fit_scaler=True)
            
            if len(X_seq) == 0:
                raise ValueError(f"Not enough data to create sequences. Need at least {self.sequence_length} samples, got {len(X)}")
            
            # Store number of features
            self.n_features_ = X_seq.shape[2]
            logging.info(f"LSTM features detected: {self.n_features_}")
            
            # Build model if not exists
            if self.model is None:
                self._build_model(self.n_features_)
            
            # Prepare training with early stopping
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=0
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=0
                )
            ]
            
            # Train model with validation split
            history = self.model.fit(
                X_seq, y_seq, 
                epochs=self.epochs, 
                batch_size=min(self.batch_size, len(X_seq)),  # Adjust batch size if needed
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0,  # Silent training
                **kwargs
            )
            
            self.is_fitted = True
            logging.info(f"LSTM trained successfully - Final loss: {history.history['loss'][-1]:.6f}")
            return self
            
        except Exception as e:
            logging.error(f"Error training LSTM: {e}")
            # Set a flag for fallback behavior
            self.is_fitted = False
            raise

    def predict(self, X):
        """Make predictions with the LSTM model."""
        try:
            if not self.is_fitted:
                logging.warning("LSTM model not fitted, returning zeros")
                return np.zeros(len(X) if hasattr(X, '__len__') else 1)
            
            X_seq, _ = self._prepare_sequences(X, fit_scaler=False)
            
            if len(X_seq) == 0:
                logging.warning("No sequences created for prediction, returning zeros")
                return np.zeros(len(X) if hasattr(X, '__len__') else 1)
            
            # Make predictions
            predictions = self.model.predict(X_seq, verbose=0).flatten()
            
            # Extend predictions to match input length
            if hasattr(X, '__len__'):
                full_predictions = np.zeros(len(X))
                start_idx = self.sequence_length - 1
                if start_idx < len(X):
                    end_idx = min(start_idx + len(predictions), len(X))
                    full_predictions[start_idx:end_idx] = predictions[:end_idx-start_idx]
                    # Fill beginning with first prediction
                    if start_idx > 0 and len(predictions) > 0:
                        full_predictions[:start_idx] = predictions[0]
                return full_predictions
            else:
                return predictions[0] if len(predictions) > 0 else 0.0
                
        except Exception as e:
            logging.error(f"Error in LSTM prediction: {e}")
            return np.zeros(len(X) if hasattr(X, '__len__') else 1)

    def get_params(self, deep=True):
        """Get parameters for scikit-learn compatibility."""
        return {
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'sequence_length': self.sequence_length,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }

    def set_params(self, **params):
        """Set parameters for scikit-learn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        # Reset model if parameters changed
        self.model = None
        self.is_fitted = False
        return self
