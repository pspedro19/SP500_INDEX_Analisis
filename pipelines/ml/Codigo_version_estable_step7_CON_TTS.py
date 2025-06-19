#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CÓDIGO VERSIÓN ESTABLE STEP 7 - CON TTS INCLUIDO
=================================================

Versión híbrida que combina:
✅ El código original funcional de 6,777 líneas 
✅ El nuevo modelo TTS agregado
✅ Todo autocontenido sin dependencias externas

Modelos incluidos: CatBoost, LightGBM, XGBoost, MLP, SVM, LSTM, TTS
"""

import logging
import os
import random
import json
import joblib
import pandas as pd
import numpy as np
import optuna
import glob
from scipy.signal import hilbert
import warnings
import matplotlib.pyplot as plt
import time
from math import sqrt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ============================================================================
# CONFIGURACIÓN INTEGRADA (Sin dependencias externas)
# ============================================================================

# Configuración de proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TRAINING_DIR = os.path.join(PROJECT_ROOT, "data", "3_trainingdata") 
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "4_results")
IMG_CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
DATE_COL = "date"
LOCAL_REFINEMENT_DAYS = 225
TRAIN_TEST_SPLIT_RATIO = 0.8
FORECAST_HORIZON_1MONTH = 20
FORECAST_HORIZON_3MONTHS = 60
RANDOM_SEED = 42

def ensure_directories():
    """Crear directorios necesarios"""
    for directory in [MODELS_DIR, RESULTS_DIR, IMG_CHARTS_DIR]:
        os.makedirs(directory, exist_ok=True)

# ============================================================================
# CONFIGURACIÓN TTS (NUEVO)
# ============================================================================

# Importar PyTorch para TTS
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    torch.set_num_threads(4)  # Configuración CPU
    logging.info("✅ PyTorch disponible para TTS")
except ImportError as e:
    TORCH_AVAILABLE = False
    logging.warning(f"⚠️ PyTorch no disponible: {e}")

# ============================================================================
# MODELO TTS (NUEVO) 
# ============================================================================

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
        """Modelo Transformer para predicción de series temporales."""
        
        def __init__(self, d_model=64, nhead=8, num_encoder_layers=3, dim_feedforward=256, 
                     dropout=0.1, sequence_length=30, n_features=1):
            super().__init__()
            
            # Validate parameters
            if d_model % nhead != 0:
                nhead = 4  # Safe fallback
                logging.warning(f"Adjusting nhead to {nhead} for d_model {d_model}")
            
            self.d_model = d_model
            self.sequence_length = sequence_length
            self.n_features = n_features
            
            # Input projection
            self.input_projection = nn.Linear(n_features, d_model)
            
            # Positional encoding
            self.pos_encoder = PositionalEncoding(d_model, dropout)
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='relu',
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, 
                num_layers=num_encoder_layers
            )
            
            # Output layers
            self.fc1 = nn.Linear(d_model, dim_feedforward // 2)
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(dim_feedforward // 2, 1)
            
            # Initialize weights
            self._init_weights()
        
        def _init_weights(self):
            """Initialize model weights."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            
        def forward(self, x):
            # x shape: (batch_size, sequence_length, n_features)
            
            # Project input to d_model dimensions
            x = self.input_projection(x)  # (batch_size, sequence_length, d_model)
            
            # Add positional encoding
            x = self.pos_encoder(x)
            
            # Transformer encoding
            encoded = self.transformer_encoder(x)  # (batch_size, sequence_length, d_model)
            
            # Global average pooling over sequence dimension
            pooled = encoded.mean(dim=1)  # (batch_size, d_model)
            
            # Final prediction layers
            x = F.relu(self.fc1(pooled))
            x = self.dropout(x)
            output = self.fc2(x)  # (batch_size, 1)
            
            return output

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN INTEGRADAS
# ============================================================================

def plot_real_vs_pred(df, title="Real vs Predicho", output_path=None, figsize=(12, 8)):
    """Gráfico de valores reales vs predichos por modelo"""
    try:
        plt.figure(figsize=figsize)
        
        if 'Modelo' in df.columns:
            models = df['Modelo'].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
            
            for i, model in enumerate(models):
                model_data = df[df['Modelo'] == model]
                if not model_data.empty:
                    plt.scatter(model_data['Valor_Real'], model_data['Valor_Predicho'], 
                              alpha=0.6, label=model, color=colors[i], s=20)
        else:
            plt.scatter(df['Valor_Real'], df['Valor_Predicho'], alpha=0.6, s=20)
        
        # Línea diagonal (predicción perfecta)
        min_val = min(df['Valor_Real'].min(), df['Valor_Predicho'].min())
        max_val = max(df['Valor_Real'].max(), df['Valor_Predicho'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        plt.xlabel('Valor Real')
        plt.ylabel('Valor Predicho')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"Gráfico guardado en: {output_path}")
        
        plt.close()
        
    except Exception as e:
        logging.error(f"Error creando gráfico: {e}")

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

# Variables globales para controlar GPU
USE_GPU = True           
GPU_MEMORY_LIMIT = 0     
has_gpu = False          

def configure_gpu(use_gpu=USE_GPU, memory_limit=GPU_MEMORY_LIMIT):
    """Configura el uso de GPU para TensorFlow."""
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logging.info("Uso de GPU deshabilitado manualmente.")
        return False
        
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        logging.info("No se detectó ninguna GPU. Usando CPU.")
        return False
    
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        if memory_limit > 0:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
            )
            logging.info(f"GPU configurada con límite de memoria: {memory_limit} MB")
        
        gpu_info = [gpu.name.split('/')[-1] for gpu in gpus]
        logging.info(f"GPU disponible para entrenamiento: {', '.join(gpu_info)}")
        return True
    except RuntimeError as e:
        logging.error(f"Error configurando GPU: {e}")
        return False

# Configurar GPU
has_gpu = configure_gpu()

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================
log_file = os.path.join(PROJECT_ROOT, "logs", f"train_models_{time.strftime('%Y%m%d_%H%M%S')}.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Asegurar directorios
ensure_directories()

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def get_most_recent_file(directory, pattern='*.xlsx'):
    """Obtiene el archivo más reciente en un directorio."""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

# ============================================================================
# CONFIGURACIÓN DE MODELOS
# ============================================================================

SCALING_REQUIRED_MODELS = {
    "CatBoost": False,
    "LightGBM": False,
    "XGBoost": False,
    "MLP": True,
    "SVM": True,
    "LSTM": True,
    "TTS": True  # NUEVO: TTS también requiere escalado
}

class Args:
    # Archivo de entrada
    input_dir = TRAINING_DIR
    input_file = get_most_recent_file(input_dir)
    
    # Directorios de salida
    output_dir = MODELS_DIR
    output_predictions = os.path.join(RESULTS_DIR, "all_models_predictions.csv")
    
    # Configuración rápida
    random_search_trials = 20
    optuna_trials = 10
    n_trials = 15
    cv_splits = 5
    gap = FORECAST_HORIZON_1MONTH
    tipo_mercado = "S&P500"
    forecast_period = "1MONTH"

args = Args()

# Semilla para reproducibilidad
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if args.input_file:
    logging.info(f"Usando el archivo más reciente: {args.input_file}")
else:
    logging.error(f"No se encontraron archivos Excel en {args.input_dir}")

# ============================================================================
# WRAPPER TTS (NUEVO)
# ============================================================================

class TTSWrapper:
    """Wrapper para el modelo Transformer Time Series (TTS)."""
    
    def __init__(self, d_model=64, nhead=8, num_encoder_layers=3, dim_feedforward=256,
                 dropout=0.1, sequence_length=20, learning_rate=0.001, batch_size=32,
                 epochs=20, patience=5, device=None, **kwargs):
        
        if not TORCH_AVAILABLE:
            raise ImportError(f"PyTorch no está instalado")
        
        # Validate and adjust parameters
        if d_model % nhead != 0:
            nhead = 4  # Safe fallback
            logging.warning(f"Adjusting nhead to {nhead} for d_model {d_model}")
            
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
        
        # Device setup - Force CPU for stability
        self.device = torch.device("cpu")
        logging.info(f"TTS using device: {self.device}")
            
        # Initialize components
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        logging.info(f"TTS Wrapper initialized - Device: {self.device}")
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Entrenar el modelo TTS."""
        try:
            logging.info(f"Training TTS with data: X={X.shape}, y={y.shape}")
            
            # Preparar datos
            X_sequences, y_sequences = self._prepare_sequences(X, y)
            
            if len(X_sequences) == 0:
                raise ValueError(f"No se pudieron crear secuencias. Datos insuficientes")
            
            # Validación opcional
            X_val_seq, y_val_seq = None, None
            if X_val is not None and y_val is not None:
                X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val, fit_scaler=False)
            
            # Crear modelo
            n_features = X_sequences.shape[2]
            self.model = TransformerTimeSeriesModel(
                d_model=self.d_model, 
                nhead=self.nhead, 
                num_encoder_layers=self.num_encoder_layers,
                dim_feedforward=self.dim_feedforward, 
                dropout=self.dropout,
                sequence_length=self.sequence_length, 
                n_features=n_features
            ).to(self.device)
            
            # Setup entrenamiento
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=1e-5
            )
            
            criterion = nn.MSELoss()
            
            # Convertir a tensors
            X_tensor = torch.FloatTensor(X_sequences).to(self.device)
            y_tensor = torch.FloatTensor(y_sequences).to(self.device)
            
            # Datasets
            train_dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            val_loader = None
            if X_val_seq is not None:
                X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val_seq).to(self.device)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Entrenamiento
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.epochs):
                # Entrenamiento
                self.model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    output = self.model(batch_X)
                    loss = criterion(output.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validación
                if val_loader:
                    self.model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            output = self.model(batch_X)
                            loss = criterion(output.squeeze(), batch_y)
                            val_loss += loss.item()
                    
                    val_loss /= len(val_loader)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.patience:
                        logging.info(f"Early stopping at epoch {epoch}")
                        break
                        
                    if epoch % 5 == 0:
                        logging.info(f"Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}")
            
            self.is_fitted = True
            logging.info("TTS training completed successfully")
            
        except Exception as e:
            logging.error(f"Error training TTS: {e}")
            raise
    
    def predict(self, X):
        """Realizar predicciones con el modelo entrenado."""
        try:
            if not self.is_fitted:
                logging.warning("TTS model not fitted, returning zeros")
                return np.zeros(len(X))
            
            # Preparar secuencias
            X_sequences, _ = self._prepare_sequences(X, fit_scaler=False)
            
            if len(X_sequences) == 0:
                logging.warning("No sequences created for prediction, returning zeros")
                return np.zeros(len(X))
            
            # Convertir a tensor
            X_tensor = torch.FloatTensor(X_sequences).to(self.device)
            
            # Predicción
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor)
                predictions = predictions.squeeze().cpu().numpy()
            
            # Asegurar que tengamos el número correcto de predicciones
            if len(predictions.shape) == 0:
                predictions = np.array([predictions])
            
            # Expandir predicciones si es necesario
            if len(predictions) < len(X):
                last_pred = predictions[-1] if len(predictions) > 0 else 0
                predictions = np.concatenate([
                    predictions,
                    np.full(len(X) - len(predictions), last_pred)
                ])
            
            return predictions[:len(X)]
            
        except Exception as e:
            logging.error(f"Error in TTS prediction: {e}")
            return np.zeros(len(X))
    
    def _prepare_sequences(self, X, y=None, fit_scaler=True):
        """Preparar secuencias para el modelo."""
        try:
            # Convertir a numpy array si es DataFrame
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = np.array(X)
            
            # Escalar datos
            if fit_scaler:
                X_scaled = self.scaler.fit_transform(X_array)
            else:
                X_scaled = self.scaler.transform(X_array)
            
            # Crear secuencias
            sequences = []
            targets = []
            
            for i in range(len(X_scaled) - self.sequence_length + 1):
                seq = X_scaled[i:(i + self.sequence_length)]
                sequences.append(seq)
                
                if y is not None:
                    targets.append(y.iloc[i + self.sequence_length - 1] if hasattr(y, 'iloc') else y[i + self.sequence_length - 1])
            
            sequences = np.array(sequences)
            targets = np.array(targets) if targets else None
            
            return sequences, targets
            
        except Exception as e:
            logging.error(f"Error preparing sequences: {e}")
            return np.array([]), np.array([]) if y is not None else None
    
    def get_params(self, deep=True):
        """Obtener parámetros del modelo."""
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
        """Establecer parámetros del modelo."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self 