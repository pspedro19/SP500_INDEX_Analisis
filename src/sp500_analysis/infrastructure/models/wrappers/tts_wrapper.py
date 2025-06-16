# -*- coding: utf-8 -*-
"""
Transformer Time Series (TTS) Wrapper
Implementa un modelo Transformer especializado para predicción de series temporales.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from typing import Optional, Tuple
    TORCH_AVAILABLE = True
except ImportError as e:
    torch = None
    nn = None
    _import_error = e
    TORCH_AVAILABLE = False

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple


if TORCH_AVAILABLE:
    class PositionalEncoding(nn.Module):
        """Positional Encoding para secuencias temporales."""
        
        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)
            
        def forward(self, x):
            x = x + self.pe[:x.size(1), :].transpose(0, 1)
            return self.dropout(x)


    class TransformerTimeSeriesModel(nn.Module):
        """Modelo Transformer para Series Temporales."""
        
        def __init__(self, d_model=64, nhead=8, num_encoder_layers=3, dim_feedforward=256, 
                     dropout=0.1, sequence_length=30, n_features=1, activation="relu"):
            super().__init__()
            
            self.d_model = d_model
            self.sequence_length = sequence_length
            self.n_features = n_features
            
            # Input projection
            self.input_projection = nn.Linear(n_features, d_model)
            
            # Positional encoding
            self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=sequence_length)
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation=activation, batch_first=True
            )
            
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
            
            # Output layers
            self.output_projection = nn.Sequential(
                nn.Linear(d_model, dim_feedforward // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward // 2, 1)
            )
            
            # Global average pooling for sequence aggregation
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            
        def forward(self, x):
            # Input projection
            x = self.input_projection(x)  # [batch, seq_len, d_model]
            
            # Add positional encoding
            x = self.positional_encoding(x)
            
            # Transformer encoding
            x = self.transformer_encoder(x)  # [batch, seq_len, d_model]
            
            # Global pooling over sequence dimension
            x = x.transpose(1, 2)  # [batch, d_model, seq_len]
            x = self.global_pool(x)  # [batch, d_model, 1]
            x = x.squeeze(-1)  # [batch, d_model]
            
            # Output projection
            output = self.output_projection(x)  # [batch, 1]
            
            return output

else:
    # Stubs para cuando PyTorch no está disponible
    class PositionalEncoding:
        pass
    
    class TransformerTimeSeriesModel:
        pass


class TTSWrapper:
    """Wrapper para el modelo Transformer Time Series (TTS)."""
    
    def __init__(self, d_model=64, nhead=8, num_encoder_layers=3, dim_feedforward=256,
                 dropout=0.1, sequence_length=30, learning_rate=0.001, batch_size=32,
                 epochs=100, patience=10, device=None, **kwargs):
        
        if not TORCH_AVAILABLE:
            raise ImportError(f"PyTorch no está instalado: {_import_error}")
            
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Initialize components
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        print(f"TTS Wrapper initialized - Device: {self.device}")
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Entrenar el modelo TTS."""
        print(f"Entrenando TTS con datos: {X.shape}")
        
        # Preparar datos
        X_sequences, y_sequences = self._prepare_sequences(X, y)
        
        if len(X_sequences) == 0:
            raise ValueError(f"No se pudieron crear secuencias. Datos insuficientes: {len(X)} < {self.sequence_length}")
        
        # Validación opcional
        X_val_seq, y_val_seq = None, None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val, fit_scaler=False)
        
        # Crear modelo
        n_features = X_sequences.shape[2]
        self.model = TransformerTimeSeriesModel(
            d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward, dropout=self.dropout,
            sequence_length=self.sequence_length, n_features=n_features
        ).to(self.device)
        
        # Setup entrenamiento
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_sequences).to(self.device),
            torch.FloatTensor(y_sequences).to(self.device)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_loader = None
        if X_val_seq is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_seq).to(self.device),
                torch.FloatTensor(y_val_seq).to(self.device)
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Entrenamiento
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Entrenando por {self.epochs} épocas...")
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = self.model(X_batch)
                        loss = criterion(outputs.squeeze(), y_batch)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    print(f"Early stopping en época {epoch+1}")
                    break
            
            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                if val_loader is not None:
                    print(f"Época {epoch+1:3d}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Época {epoch+1:3d}: Train Loss: {train_loss:.6f}")
        
        # Load best model if validation was used
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        self.is_fitted = True
        print(f"TTS entrenado exitosamente")
    
    def predict(self, X):
        """Realizar predicciones con el modelo entrenado."""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        X_sequences, _ = self._prepare_sequences(X, None, fit_scaler=False)
        
        if len(X_sequences) == 0:
            return np.zeros(len(X))
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            dataset = TensorDataset(torch.FloatTensor(X_sequences))
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            
            for (X_batch,) in dataloader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions.extend(outputs.cpu().numpy().flatten())
        
        # Extender predicciones para cubrir todas las muestras
        full_predictions = np.zeros(len(X))
        start_idx = self.sequence_length - 1
        if start_idx < len(X):
            end_idx = min(start_idx + len(predictions), len(X))
            full_predictions[start_idx:end_idx] = predictions[:end_idx-start_idx]
            
            if start_idx > 0 and len(predictions) > 0:
                full_predictions[:start_idx] = predictions[0]
        
        return full_predictions
    
    def _prepare_sequences(self, X, y=None, fit_scaler=True):
        """Preparar secuencias temporales para el transformer."""
        # Scaling
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Crear secuencias
        if len(X_scaled) < self.sequence_length:
            return np.array([]), np.array([]) if y is not None else None
        
        X_sequences = []
        y_sequences = [] if y is not None else None
        
        for i in range(self.sequence_length, len(X_scaled) + 1):
            X_sequences.append(X_scaled[i-self.sequence_length:i])
            if y is not None:
                y_sequences.append(y.iloc[i-1])
        
        X_sequences = np.array(X_sequences)
        
        if y is not None:
            y_sequences = np.array(y_sequences)
            return X_sequences, y_sequences
        else:
            return X_sequences, None
    
    def get_params(self, deep=True):
        """Obtener parámetros del modelo (compatible con sklearn)."""
        return {
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_encoder_layers': self.num_encoder_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'sequence_length': self.sequence_length,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience
        }
    
    def set_params(self, **params):
        """Establecer parámetros del modelo (compatible con sklearn)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
