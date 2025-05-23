import logging
import os
import random
import json
import joblib
import pandas as pd
import numpy as np
import optuna
import glob
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

# Variables globales para controlar GPU
# Puedes cambiar estos valores directamente en el notebook
USE_GPU = True           # Cambiar a False para deshabilitar GPU
GPU_MEMORY_LIMIT = 0     # Establecer a un valor en MB para limitar memoria (0 = sin límite)
has_gpu = False          # Estado (se actualizará en configure_gpu)

def configure_gpu(use_gpu=USE_GPU, memory_limit=GPU_MEMORY_LIMIT):
    """
    Configura el uso de GPU para TensorFlow.
    
    Args:
        use_gpu (bool): Si es False, desactiva completamente el uso de GPU
        memory_limit (int): Límite de memoria en MB (0 = sin límite)
        
    Returns:
        bool: True si la GPU está disponible y habilitada, False en caso contrario
    """
    if not use_gpu:
        # Deshabilitar GPU completamente
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logging.info("Uso de GPU deshabilitado manualmente.")
        return False
        
    # Verificar si hay GPUs disponibles
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        logging.info("No se detectó ninguna GPU. Usando CPU.")
        return False
    
    try:
        # Permitir crecimiento de memoria según necesidad
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Limitar memoria GPU si se especificó un límite
        if memory_limit > 0:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
            )
            logging.info(f"GPU configurada con límite de memoria: {memory_limit} MB")
        
        # Mostrar información sobre las GPUs disponibles
        gpu_info = [gpu.name.split('/')[-1] for gpu in gpus]
        logging.info(f"GPU disponible para entrenamiento: {', '.join(gpu_info)}")
        return True
    except RuntimeError as e:
        logging.error(f"Error configurando GPU: {e}")
        return False

# Configurar GPU con los valores predeterminados
has_gpu = configure_gpu()

# Importar configuraciones centralizadas
from config import (
    PROJECT_ROOT, MODELS_DIR, TRAINING_DIR, RESULTS_DIR, IMG_CHARTS_DIR,
    DATE_COL, LOCAL_REFINEMENT_DAYS, TRAIN_TEST_SPLIT_RATIO,
    FORECAST_HORIZON_1MONTH, FORECAST_HORIZON_3MONTHS, RANDOM_SEED,
    ensure_directories
)

# Importar funciones de visualización
from utils.plots import (
    plot_real_vs_pred, plot_training_curves, plot_feature_importance
)

# ------------------------------
# CONFIGURACIÓN DE LOGGING
# ------------------------------
log_file = os.path.join(PROJECT_ROOT, "logs", f"train_models_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Asegurar que existen los directorios
ensure_directories()

# ------------------------------
# FUNCIONES AUXILIARES
# ------------------------------
def get_most_recent_file(directory, pattern='*.xlsx'):
    """
    Obtiene el archivo más reciente en un directorio con la extensión especificada.
    
    Args:
        directory (str): Ruta al directorio
        pattern (str): Patrón para buscar archivos
        
    Returns:
        str: Ruta completa al archivo más reciente
    """
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

# ------------------------------
# CONFIGURACIÓN GLOBAL
# ------------------------------
# Constantes de horizonte para alinear con pipeline de series temporales
FORECAST_HORIZON_1MONTH = 20  # Exactamente 20 días hábiles
FORECAST_HORIZON_3MONTHS = 60  # Exactamente 60 días hábiles
LOCAL_REFINEMENT_DAYS = 225  # Número de días para refinamiento local
TRAIN_TEST_SPLIT_RATIO = 0.8  # 80/20 para training/test en refinamiento local

SCALING_REQUIRED_MODELS = {
    "CatBoost": False,
    "LightGBM": False,
    "XGBoost": False,
    "MLP": True,
    "SVM": True,
    "LSTM": True  # LSTM también requiere escalado
}

class Args:
    # Obtener el archivo de entrada más reciente de la carpeta 3_trainingdata
    input_dir = TRAINING_DIR
    input_file = get_most_recent_file(input_dir)
    
    # Directorios de salida
    output_dir = MODELS_DIR
    output_predictions = os.path.join(RESULTS_DIR, "all_models_predictions.csv")
    
    # Numero de entrenamientos - CONFIGURACIÓN PARA PRUEBAS RÁPIDAS ⚡
    # Para el enfoque de tres zonas
    random_search_trials = 3   # Era 30 ⭐ CAMBIO PRINCIPAL (90% más rápido)
    optuna_trials = 2          # Era 15 ⭐ CAMBIO PRINCIPAL (87% más rápido)
    
    n_trials = 20  # Original (no se usa mucho)
    cv_splits = 3              # Era 5 ⭐ CAMBIO (40% más rápido)
    gap = FORECAST_HORIZON_1MONTH  # Alineado con horizonte de 20 días (1MONTH)
    tipo_mercado = "S&P500"
    forecast_period = "1MONTH"  # Puede ser "1MONTH" o "3MONTHS"

args = Args()


# Asegurar que existen los directorios
os.makedirs(os.path.dirname(args.output_predictions), exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)

# Semilla para reproducibilidad
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if args.input_file:
    logging.info(f"Usando el archivo más reciente: {args.input_file}")
else:
    logging.error(f"No se encontraron archivos Excel en {args.input_dir}")


def save_hyperparameters_to_csv(algo_name, random_params, optuna_params, backtest_metrics, holdout_metrics):
    """
    Guarda el historial de hiperparámetros en un CSV que se retroalimenta con cada ejecución.
    
    Args:
        algo_name: Nombre del algoritmo
        random_params: Parámetros encontrados por RandomSearch
        optuna_params: Parámetros refinados por Optuna
        backtest_metrics: Métricas de back-test
        holdout_metrics: Métricas de hold-out
    """
    hp_history_file = os.path.join(RESULTS_DIR, "hyperparameters_history.csv")
    
    # Preparar datos para el CSV
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Datos básicos
    entry_data = {
        'date': current_time,
        'model': algo_name,
        'method': 'ThreeZones',
        'random_rmse': backtest_metrics.get('RMSE', np.nan),
        'optuna_rmse': holdout_metrics.get('RMSE', np.nan)
    }
    
    # Añadir parámetros de RandomSearch
    if random_params and isinstance(random_params, dict):
        for param_name, param_value in random_params.items():
            entry_data[f'random_{param_name}'] = param_value
    
    # Añadir parámetros de Optuna
    if optuna_params and isinstance(optuna_params, dict):
        for param_name, param_value in optuna_params.items():
            entry_data[f'optuna_{param_name}'] = param_value
    
    # Crear DataFrame para esta entrada
    new_entry = pd.DataFrame([entry_data])
    
    # Verificar si el archivo existe y cargarlo
    if os.path.exists(hp_history_file):
        try:
            existing_history = pd.read_csv(hp_history_file)
            # Concatenar con el historial existente
            combined_history = pd.concat([existing_history, new_entry], ignore_index=True)
        except Exception as e:
            logging.warning(f"Error al cargar historial existente: {e}")
            combined_history = new_entry
    else:
        combined_history = new_entry
    
    # Guardar el historial actualizado
    combined_history.to_csv(hp_history_file, index=False)
    logging.info(f"Historial de hiperparámetros actualizado y guardado en: {hp_history_file}")

def plot_hyperparameter_evolution(output_dir=IMG_CHARTS_DIR):
    """
    Genera gráficos de la evolución de los hiperparámetros a lo largo del tiempo.
    """
    hp_history_file = os.path.join(RESULTS_DIR, "hyperparameters_history.csv")
    
    if not os.path.exists(hp_history_file):
        logging.warning("No se encontró archivo de historial de hiperparámetros.")
        return
    
    try:
        history_df = pd.read_csv(hp_history_file)
        history_df['date'] = pd.to_datetime(history_df['date'])
        
        # Obtener lista de modelos
        models = history_df['model'].unique()
        
        # Para cada modelo, graficar la evolución de los hiperparámetros
        for model in models:
            model_df = history_df[history_df['model'] == model].sort_values('date')
            
            if len(model_df) < 2:
                logging.info(f"No hay suficientes datos para {model}. Se necesitan al menos 2 ejecuciones.")
                continue
            
            # Identificar columnas de parámetros para RandomSearch y Optuna
            random_params = [col for col in model_df.columns if col.startswith('random_') and col != 'random_rmse']
            optuna_params = [col for col in model_df.columns if col.startswith('optuna_') and col != 'optuna_rmse']
            
            # Calcular número total de subgráficos (RMSE + parámetros)
            n_plots = 1 + len(random_params) + len(optuna_params)
            fig_height = max(10, n_plots * 2)  # Altura dinámica según número de parámetros
            
            # Crear figura
            plt.figure(figsize=(12, fig_height))
            
            # Subgráfico para RMSE
            plt.subplot(n_plots, 1, 1)
            if 'random_rmse' in model_df.columns:
                plt.plot(model_df['date'], model_df['random_rmse'], marker='o', label='RandomSearch')
            if 'optuna_rmse' in model_df.columns:
                plt.plot(model_df['date'], model_df['optuna_rmse'], marker='x', label='Optuna')
            plt.title(f"Evolución de RMSE - {model}")
            plt.grid(True, alpha=0.3)
            plt.ylabel("RMSE")
            plt.legend()
            
            # Subgráficos para parámetros de RandomSearch
            for i, param in enumerate(random_params):
                plt.subplot(n_plots, 1, i + 2)
                plt.plot(model_df['date'], model_df[param], marker='o', color='blue')
                param_name = param.replace('random_', '')
                plt.title(f"RandomSearch: {param_name}")
                plt.grid(True, alpha=0.3)
                plt.ylabel(param_name)
            
            # Subgráficos para parámetros de Optuna
            for i, param in enumerate(optuna_params):
                plt.subplot(n_plots, 1, i + 2 + len(random_params))
                plt.plot(model_df['date'], model_df[param], marker='x', color='red')
                param_name = param.replace('optuna_', '')
                plt.title(f"Optuna: {param_name}")
                plt.grid(True, alpha=0.3)
                plt.ylabel(param_name)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model.lower()}_hyperparameter_evolution.png"), dpi=300)
            plt.close()
            
        logging.info(f"Gráficos de evolución de hiperparámetros guardados en: {output_dir}")
    except Exception as e:
        logging.error(f"Error al generar gráficos de evolución de hiperparámetros: {e}")

def create_sequences(X, y, sequence_length=10):
    """
    Transforma los datos en formato de secuencias para LSTM.
    
    Args:
        X (DataFrame): Datos de características
        y (Series): Variable objetivo
        sequence_length (int): Longitud de la secuencia (ventana de tiempo)
        
    Returns:
        tuple: (X_seq, y_seq) - Datos transformados en formato de secuencia
    """
    X_values = X.values
    y_values = y.values
    
    X_seq = []
    y_seq = []
    
    for i in range(len(X_values) - sequence_length):
        X_seq.append(X_values[i:i + sequence_length])
        y_seq.append(y_values[i + sequence_length])
    
    return np.array(X_seq), np.array(y_seq)

# -----------------------------------------------------------------
# FUNCIONES DE OBJETIVO (usando únicamente datos de Training)
# -----------------------------------------------------------------
def create_lagged_target(df, target_col, lag=20):
    """
    Crea una nueva columna que contiene la variable target retrocedida 'lag' días.
    Esto permite entrenar modelos para predecir el valor futuro del target.
    
    Args:
        df (DataFrame): DataFrame con los datos, incluyendo la columna target
        target_col (str): Nombre de la columna target
        lag (int): Número de días hábiles a retroceder (por defecto: 20)
        
    Returns:
        DataFrame: DataFrame con la columna target original y la nueva columna target_lagged
    """
    logging.info(f"Creando target con lag de {lag} días para '{target_col}'")
    
    # Hacer una copia para no modificar el original
    df_copy = df.copy()
    
    # Asegurar que el DataFrame está ordenado por fecha
    if 'date' in df_copy.columns:
        df_copy = df_copy.sort_values('date')
    
    # Crear la columna target_lagged (target retrocedido)
    target_lagged_col = f"{target_col}_lagged_{lag}"
    df_copy[target_lagged_col] = df_copy[target_col].shift(-lag)
    
    # Eliminar las filas donde target_lagged es NaN (últimos 'lag' días)
    df_filtered = df_copy.dropna(subset=[target_lagged_col])
    
    logging.info(f"Filas originales: {len(df_copy)}, Filas después del lag: {len(df_filtered)}")
    logging.info(f"Se eliminaron las últimas {len(df_copy) - len(df_filtered)} filas (sin valores futuros)")
    
    return df_filtered

def objective_lstm(trial, X, y, base_params=None):
    """
    Optimización de hiperparámetros para LSTM.
    
    Args:
        trial: Instancia de trial de Optuna
        X: Features de entrenamiento
        y: Target de entrenamiento
        base_params: Parámetros base de RandomSearch (opcional)
        
    Returns:
        float: RMSE medio de validación cruzada
    """
    start_time = time.perf_counter()
    
    # Si tenemos parámetros base de RandomSearch, usarlos para refinar rangos
    if base_params and isinstance(base_params, dict):
        units = trial.suggest_int(
            "units", 
            max(base_params.get("units", 64) - 32, 16),
            min(base_params.get("units", 64) + 32, 256)
        )
        learning_rate = trial.suggest_float(
            "learning_rate", 
            max(base_params.get("learning_rate", 0.001) * 0.5, 0.0001),
            min(base_params.get("learning_rate", 0.001) * 2.0, 0.01),
            log=True
        )
        dropout_rate = trial.suggest_float(
            "dropout_rate", 
            max(base_params.get("dropout_rate", 0.2) - 0.1, 0.0),
            min(base_params.get("dropout_rate", 0.2) + 0.2, 0.5)
        )
        sequence_length = trial.suggest_int(
            "sequence_length",
            max(base_params.get("sequence_length", 10) - 5, 5),
            min(base_params.get("sequence_length", 10) + 5, 20)
        )
    else:
        units = trial.suggest_int("units", 32, 128)
        learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.01, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        sequence_length = trial.suggest_int("sequence_length", 5, 20)
    
    # Usar el gap correcto según el horizonte de predicción
    tscv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    rmse_scores = []
    
    # Determinar batch_size óptimo según disponibilidad de GPU
    batch_size = 64 if has_gpu else 32
    
    # Modo rápido está habilitado? (Podemos definir esto globalmente en el notebook)
    fast_mode = getattr(args, 'fast_mode', False)
    max_epochs = 50 if fast_mode else 100
    patience = 5 if fast_mode else 10
    
    # Función para crear y compilar el modelo LSTM
    def create_lstm_model(n_features, units, dropout_rate, learning_rate):
        """Crea un modelo LSTM con configuración optimizada."""
        # Crear modelo con la estrategia de dispositivo adecuada
        if has_gpu:
            try:
                with tf.device('/gpu:0'):
                    model = Sequential([
                        LSTM(units=units, 
                             input_shape=(sequence_length, n_features), 
                             return_sequences=False),
                        Dropout(dropout_rate),
                        Dense(1)
                    ])
            except RuntimeError:
                # Fallback a CPU si hay problemas con GPU
                logging.warning("Error al crear modelo en GPU. Fallback a CPU.")
                model = Sequential([
                    LSTM(units=units, 
                         input_shape=(sequence_length, n_features), 
                         return_sequences=False),
                    Dropout(dropout_rate),
                    Dense(1)
                ])
        else:
            model = Sequential([
                LSTM(units=units, 
                     input_shape=(sequence_length, n_features), 
                     return_sequences=False),
                Dropout(dropout_rate),
                Dense(1)
            ])
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse'
        )
        return model
    
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        train_idx = sorted(train_idx)
        val_idx = sorted(val_idx)
        
        # Asegurar que hay suficientes datos para la secuencia
        if len(train_idx) <= sequence_length or len(val_idx) <= sequence_length:
            logging.warning(f"[LSTM][fold {fold_i}] Datos insuficientes para crear secuencias con length={sequence_length}")
            continue
        
        X_train_cv, y_train_cv = X.iloc[train_idx], y.iloc[train_idx]
        X_val_cv, y_val_cv = X.iloc[val_idx], y.iloc[val_idx]
        
        # Crear secuencias para LSTM
        X_train_seq, y_train_seq = create_sequences(X_train_cv, y_train_cv, sequence_length)
        X_val_seq, y_val_seq = create_sequences(X_val_cv, y_val_cv, sequence_length)
        
        # Si no hay suficientes secuencias, saltar este fold
        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            logging.warning(f"[LSTM][fold {fold_i}] No se pudieron crear secuencias suficientes")
            continue
        
        # Crear y entrenar modelo LSTM
        n_features = X.shape[1]
        model = create_lstm_model(n_features, units, dropout_rate, learning_rate)
        
        # Reducir verbosidad para acelerar
        verbose_level = 0
        
        # Definir early stopping para evitar overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            min_delta=0.001  # Umbral mínimo de mejora
        )
        
        # Entrenar modelo con early stopping
        try:
            if has_gpu:
                with tf.device('/gpu:0'):
                    model.fit(
                        X_train_seq, y_train_seq,
                        validation_data=(X_val_seq, y_val_seq),
                        epochs=max_epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping],
                        verbose=verbose_level
                    )
            else:
                model.fit(
                    X_train_seq, y_train_seq,
                    validation_data=(X_val_seq, y_val_seq),
                    epochs=max_epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=verbose_level
                )
            
            # Evaluar en el conjunto de validación
            if has_gpu:
                with tf.device('/gpu:0'):
                    y_pred = model.predict(X_val_seq, verbose=0).flatten()
            else:
                y_pred = model.predict(X_val_seq, verbose=0).flatten()
                
            # Calcular RMSE para este fold
            rmse = sqrt(mean_squared_error(y_val_seq, y_pred))
            rmse_scores.append(rmse)
            
            # Para modo más rápido, podemos usar solo algunos folds
            if fast_mode and fold_i >= min(3, args.cv_splits - 1):
                logging.info(f"[LSTM] Modo rápido: limitando a {fold_i+1} folds")
                break
                
        except (ValueError, tf.errors.ResourceExhaustedError) as e:
            # Manejar errores de memoria u otros problemas
            logging.warning(f"[LSTM][fold {fold_i}] Error durante entrenamiento: {e}")
            # Si es el primer fold, fallar; de lo contrario, usar los scores existentes
            if fold_i == 0:
                return 9999.0
        
        finally:
            # Liberar memoria explícitamente
            tf.keras.backend.clear_session()
    
    # Calcular RMSE medio (si hay algún score válido)
    if rmse_scores:
        mean_rmse = np.mean(rmse_scores)
    else:
        # Si no hay scores válidos, asignar un valor alto
        mean_rmse = 9999.0
        
    elapsed_time = time.perf_counter() - start_time
    gpu_status = "✓" if has_gpu else "✗"
    logging.info(f"[LSTM] Trial {trial.number}: RMSE={mean_rmse:.4f}, Time={elapsed_time:.2f}s, GPU={gpu_status}")
    return mean_rmse


def run_randomized_search(algo_name, X, y, tscv):
    """
    Ejecuta RandomizedSearchCV para exploración inicial de hiperparámetros.
    
    IMPORTANTE PARA DESPLIEGUE:
    - Esta función tiene configuraciones optimizadas para pruebas rápidas
    - Ver comentarios marcados con ⚠️ PRODUCCIÓN para ajustes finales
    """
    search_start_time = time.perf_counter()
    logging.info(f"[{algo_name}] ========== Iniciando RandomizedSearchCV ==========")
    
    # Determinar modo de ejecución (normal(full) o rápido(fast))
    execution_mode = os.environ.get("EXECUTION_MODE", "fast")
    logging.info(f"[{algo_name}] Modo de ejecución detectado: {execution_mode}")
    
    # Ajustar número de trials según modo
    n_iter = args.random_search_trials
    if execution_mode == "fast":
        n_iter = max(3, args.random_search_trials // 3)  # Reducir a 1/3 en modo rápido
        logging.info(f"[{algo_name}] Modo rápido activado: usando {n_iter} trials en lugar de {args.random_search_trials}")
    else:
        logging.info(f"[{algo_name}] Modo completo: usando {n_iter} trials")
    
    # ============================================================================
    # DISTRIBUCIONES DE PARÁMETROS - CONFIGURADAS PARA PRUEBAS RÁPIDAS
    # ⚠️ PRODUCCIÓN: Cambiar estos valores para el despliegue final
    # ============================================================================
    
    if execution_mode == "fast":
        # 🏃‍♂️ CONFIGURACIÓN RÁPIDA PARA DESARROLLO Y TESTING
        logging.info(f"[{algo_name}] Usando parámetros optimizados para velocidad")
        param_distributions = {
            "CatBoost": {
                "learning_rate": uniform(0.01, 0.04),      # Rango: 0.01-0.05
                "depth": randint(4, 7),                     # Rango: 4-6 (menos profundidad)
                "iterations": randint(50, 151)             # ⭐ CRÍTICO: 50-150 (era 300-701)
                # ⚠️ PRODUCCIÓN: Cambiar a randint(500, 2001) para mejor precisión
            },
            "LightGBM": {
                "learning_rate": uniform(0.01, 0.04),      # Rango: 0.01-0.05
                "max_depth": randint(4, 7),                 # Rango: 4-6
                "n_estimators": randint(50, 151),          # ⭐ CRÍTICO: 50-150 (era 300-701)
                "subsample": uniform(0.7, 0.3)             # Rango: 0.7-1.0
                # ⚠️ PRODUCCIÓN: Cambiar n_estimators a randint(500, 2001)
            },
            "XGBoost": {
                "learning_rate": uniform(0.01, 0.04),      # Rango: 0.01-0.05
                "max_depth": randint(4, 7),                 # Rango: 4-6
                "n_estimators": randint(50, 151),          # ⭐ CRÍTICO: 50-150 (era 300-701)
                "subsample": uniform(0.7, 0.3)             # Rango: 0.7-1.0
                # ⚠️ PRODUCCIÓN: Cambiar n_estimators a randint(500, 2001)
            },
            "MLP": {
                "hidden_neurons": randint(50, 101),        # Rango: 50-100 (era 50-201)
                "learning_rate_init": uniform(0.001, 0.009), # Rango: 0.001-0.01
                "max_iter": randint(50, 201)               # ⭐ CRÍTICO: 50-200 (era 200-1001)
                # ⚠️ PRODUCCIÓN: Cambiar max_iter a randint(200, 1001)
            },
            "SVM": {
                "C": uniform(0.5, 4.5),                    # Rango: 0.5-5.0 (era 0.1-10)
                "epsilon": uniform(0.05, 0.15)             # Rango: 0.05-0.2 (era 0.01-0.5)
                # ⚠️ PRODUCCIÓN: Ampliar rangos - C: uniform(0.1, 9.9), epsilon: uniform(0.01, 0.49)
            },
            "LSTM": {
                "units": randint(32, 65),                  # Rango: 32-64 (era 32-129)
                "learning_rate": uniform(0.001, 0.009),   # Rango: 0.001-0.01
                "dropout_rate": uniform(0.1, 0.3),        # Rango: 0.1-0.4 (era 0.0-0.5)
                "sequence_length": randint(5, 11)         # Rango: 5-10 (era 5-21)
                # ⚠️ PRODUCCIÓN: Ampliar rangos según comentarios
            }
        }
    else:
        # 🏭 CONFIGURACIÓN COMPLETA PARA PRODUCCIÓN
        logging.info(f"[{algo_name}] Usando parámetros completos para producción")
        param_distributions = {
            "CatBoost": {
                "learning_rate": uniform(0.001, 0.049),    # Rango completo: 0.001-0.05
                "depth": randint(4, 11),                    # Rango completo: 4-10
                "iterations": randint(500, 2001)           # Rango completo: 500-2000
            },
            "LightGBM": {
                "learning_rate": uniform(0.001, 0.049),
                "max_depth": randint(4, 11),
                "n_estimators": randint(500, 2001),
                "subsample": uniform(0.5, 0.5)  # 0.5 a 1.0
            },
            "XGBoost": {
                "learning_rate": uniform(0.001, 0.049),
                "max_depth": randint(4, 11),
                "n_estimators": randint(500, 2001),
                "subsample": uniform(0.5, 0.5)  # 0.5 a 1.0
            },
            "MLP": {
                "hidden_neurons": randint(50, 201),
                "learning_rate_init": uniform(0.0001, 0.01),
                "max_iter": randint(200, 1001)
            },
            "SVM": {
                "C": uniform(0.1, 9.9),  # 0.1 a 10
                "epsilon": uniform(0.01, 0.49)  # 0.01 a 0.5
            },
            "LSTM": {
                "units": randint(32, 129),  # 32 a 128 unidades
                "learning_rate": uniform(0.0001, 0.01),  # 0.0001 a 0.01
                "dropout_rate": uniform(0.0, 0.5),  # 0 a 0.5
                "sequence_length": randint(5, 21)  # 5 a 20
            }
        }
    
    if algo_name not in param_distributions:
        logging.warning(f"[{algo_name}] No se encontraron distribuciones de parámetros. Saltando RandomizedSearchCV.")
        return None
    
    # Log de parámetros que se van a usar
    current_params = param_distributions[algo_name]
    logging.info(f"[{algo_name}] Parámetros a optimizar:")
    for param_name, param_dist in current_params.items():
        if hasattr(param_dist, 'low') and hasattr(param_dist, 'high'):
            # Para randint
            logging.info(f"  - {param_name}: {param_dist.low} a {param_dist.high-1}")
        elif hasattr(param_dist, 'loc') and hasattr(param_dist, 'scale'):
            # Para uniform
            low = param_dist.loc
            high = param_dist.loc + param_dist.scale
            logging.info(f"  - {param_name}: {low:.4f} a {high:.4f}")
        else:
            logging.info(f"  - {param_name}: {param_dist}")
    
    # Para LSTM usamos un enfoque especial (no RandomizedSearchCV)
    if algo_name == "LSTM":
        logging.info(f"[{algo_name}] Generando parámetros iniciales aleatorios para LSTM")
        
        # Para LSTM, generamos parámetros aleatorios manualmente ya que no es compatible con RandomizedSearchCV
        lstm_params = {}
        
        # Generar parámetros aleatorios según las distribuciones definidas
        for param_name, distribution in param_distributions["LSTM"].items():
            try:
                if hasattr(distribution, 'rvs'):
                    # Si tiene método rvs, es una distribución de scipy.stats
                    param_value = distribution.rvs()
                elif hasattr(distribution, 'low') and hasattr(distribution, 'high'):
                    # Para randint manualmente
                    param_value = np.random.randint(distribution.low, distribution.high)
                elif hasattr(distribution, 'loc') and hasattr(distribution, 'scale'):
                    # Para uniform manualmente  
                    param_value = np.random.uniform(distribution.loc, distribution.loc + distribution.scale)
                else:
                    # Valor por defecto si no reconocemos la distribución
                    param_value = None
                    
                if param_value is not None:
                    # Convertir a tipos apropiados
                    if param_name in ['units', 'sequence_length']:
                        lstm_params[param_name] = int(param_value)
                    else:
                        lstm_params[param_name] = float(param_value)
                        
            except Exception as e:
                logging.warning(f"[{algo_name}] Error al generar {param_name}: {e}")
                # Valores por defecto
                default_values = {
                    'units': 64,
                    'learning_rate': 0.001,
                    'dropout_rate': 0.2,
                    'sequence_length': 10
                }
                if param_name in default_values:
                    lstm_params[param_name] = default_values[param_name]
        
        # Asegurar que tenemos todos los parámetros necesarios
        if "units" not in lstm_params:
            lstm_params["units"] = 64
        if "learning_rate" not in lstm_params:
            lstm_params["learning_rate"] = 0.001
        if "dropout_rate" not in lstm_params:
            lstm_params["dropout_rate"] = 0.2
        if "sequence_length" not in lstm_params:
            lstm_params["sequence_length"] = 10
        
        logging.info(f"[{algo_name}] Parámetros iniciales generados:")
        for param_name, param_value in lstm_params.items():
            logging.info(f"  - {param_name}: {param_value}")
        
        return lstm_params
    
    # Crear modelo base según algoritmo (para los modelos que no son LSTM)
    logging.info(f"[{algo_name}] Creando modelo base para RandomizedSearchCV")
    
    if algo_name == "CatBoost":
        model = CatBoostRegressor(verbose=0, random_seed=RANDOM_SEED)
    elif algo_name == "LightGBM":
        model = lgb.LGBMRegressor(random_state=RANDOM_SEED, verbose=-1)  # Silenciar logs de LightGBM
    elif algo_name == "XGBoost":
        model = xgb.XGBRegressor(random_state=RANDOM_SEED, verbosity=0)  # Silenciar logs de XGBoost
    elif algo_name == "MLP":
        model = MLPRegressor(random_state=RANDOM_SEED)
    elif algo_name == "SVM":
        model = SVR()
    else:
        logging.error(f"[{algo_name}] Algoritmo no soportado para RandomizedSearchCV.")
        return None
    
    # Configuración óptima para paralelismo
    # Limitar el número de trabajos para evitar saturación de recursos
    n_jobs = min(4, os.cpu_count() or 1)  # Usa hasta 4 cores o los disponibles si son menos
    logging.info(f"[{algo_name}] Configuración de paralelismo: {n_jobs} cores")
    
    # Configurar y ejecutar RandomizedSearchCV para los modelos que no son LSTM
    logging.info(f"[{algo_name}] Configurando RandomizedSearchCV...")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions[algo_name],
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_SEED,
        n_jobs=n_jobs,           # Paralelismo controlado
        pre_dispatch="2*n_jobs", # Mejor gestión de memoria
        error_score=np.nan,      # No fallar si hay errores en evaluaciones individuales
        verbose=0                # Reducir salida para menos overhead
    )
    
    # Registrar inicio de búsqueda
    logging.info(f"[{algo_name}] 🚀 Iniciando búsqueda con {n_iter} trials usando {n_jobs} cores")
    logging.info(f"[{algo_name}] Tamaño del dataset: {X.shape[0]} filas, {X.shape[1]} características")
    logging.info(f"[{algo_name}] CV splits: {tscv.n_splits}, Gap: {args.gap}")
    
    # Establecer un límite de tiempo para la búsqueda
    max_search_time = 1800 if execution_mode == "fast" else 3600  # 30 min vs 1 hora
    logging.info(f"[{algo_name}] Tiempo máximo permitido: {max_search_time/60:.1f} minutos")
    
    try:
        search_start = time.perf_counter()
        
        # Ejecutar la búsqueda directamente
        search.fit(X, y)
        
        # Verificar si excedió el tiempo límite
        search_duration = time.perf_counter() - search_start
        
        if search_duration > max_search_time:
            logging.warning(f"[{algo_name}] ⚠️ RandomizedSearchCV tomó demasiado tiempo: {search_duration:.2f}s")
        else:
            logging.info(f"[{algo_name}] ✅ Búsqueda completada en tiempo razonable: {search_duration:.2f}s")
        
        # Extraer resultados
        best_params = search.best_params_
        best_score = -search.best_score_
        
        # Logs detallados de resultados
        logging.info(f"[{algo_name}] ========== Resultados RandomizedSearchCV ==========")
        logging.info(f"[{algo_name}] ✅ Mejor RMSE encontrado: {best_score:.4f}")
        logging.info(f"[{algo_name}] 📊 Mejores parámetros encontrados:")
        for param_name, param_value in best_params.items():
            logging.info(f"[{algo_name}]   - {param_name}: {param_value}")
        
        # Información adicional sobre la búsqueda
        if hasattr(search, 'cv_results_'):
            cv_results = search.cv_results_
            mean_scores = cv_results['mean_test_score']
            std_scores = cv_results['std_test_score']
            
            # Convertir a RMSE (scores son negativos)
            rmse_scores = -mean_scores
            best_idx = np.argmax(mean_scores)
            
            logging.info(f"[{algo_name}] 📈 Estadísticas de la búsqueda:")
            logging.info(f"[{algo_name}]   - Mejor RMSE: {rmse_scores[best_idx]:.4f} ± {std_scores[best_idx]:.4f}")
            logging.info(f"[{algo_name}]   - RMSE promedio: {np.mean(rmse_scores):.4f}")
            logging.info(f"[{algo_name}]   - RMSE mínimo: {np.min(rmse_scores):.4f}")
            logging.info(f"[{algo_name}]   - RMSE máximo: {np.max(rmse_scores):.4f}")
        
        # Registro de tiempo de búsqueda
        search_time = time.perf_counter() - search_start_time
        logging.info(f"[{algo_name}] ⏱️ Tiempo total de RandomizedSearchCV: {search_time:.2f}s")
        
        # Estimación de tiempo para Optuna
        estimated_optuna_time = (search_time / n_iter) * args.optuna_trials
        logging.info(f"[{algo_name}] 🔮 Tiempo estimado para Optuna ({args.optuna_trials} trials): {estimated_optuna_time:.2f}s")
        
        return best_params
        
    except Exception as e:
        logging.error(f"[{algo_name}] ❌ Error en RandomizedSearchCV: {e}")
        logging.error(f"[{algo_name}] Traceback: ", exc_info=True)
        
        # En caso de error, devolver parámetros por defecto
        logging.info(f"[{algo_name}] 🔄 Usando parámetros por defecto como respaldo")
        
        default_params = {
            "CatBoost": {"learning_rate": 0.01, "depth": 6, "iterations": 100 if execution_mode == "fast" else 1000},
            "LightGBM": {"learning_rate": 0.01, "max_depth": 6, "n_estimators": 100 if execution_mode == "fast" else 1000, "subsample": 0.8},
            "XGBoost": {"learning_rate": 0.01, "max_depth": 6, "n_estimators": 100 if execution_mode == "fast" else 1000, "subsample": 0.8},
            "MLP": {"hidden_neurons": 100, "learning_rate_init": 0.001, "max_iter": 100 if execution_mode == "fast" else 500},
            "SVM": {"C": 1.0, "epsilon": 0.1}
        }
        
        if algo_name in default_params:
            fallback_params = default_params[algo_name]
            logging.info(f"[{algo_name}] 📋 Parámetros por defecto:")
            for param_name, param_value in fallback_params.items():
                logging.info(f"[{algo_name}]   - {param_name}: {param_value}")
            return fallback_params
        else:
            logging.error(f"[{algo_name}] No se encontraron parámetros por defecto para {algo_name}")
            return None


def objective_catboost(trial, X, y, base_params=None):
    """Optimización de hiperparámetros para CatBoost con rangos refinados."""
    start_time = time.perf_counter()
    
    # Si tenemos parámetros base de RandomSearch, usarlos para refinar rangos
    if base_params and isinstance(base_params, dict):
        learning_rate = trial.suggest_float(
            "learning_rate", 
            max(base_params.get("learning_rate", 0.01) * 0.5, 0.0005),
            min(base_params.get("learning_rate", 0.01) * 2.0, 0.1), 
            log=True
        )
        depth = trial.suggest_int(
            "depth", 
            max(base_params.get("depth", 6) - 2, 3),
            min(base_params.get("depth", 6) + 2, 12)
        )
        iterations = trial.suggest_int(
            "iterations", 
            max(base_params.get("iterations", 1000) - 300, 300),
            min(base_params.get("iterations", 1000) + 300, 3000)
        )
    else:
        # Si no tenemos parámetros base, usar rangos amplios (como antes)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.05, log=True)
        depth = trial.suggest_int("depth", 4, 10)
        iterations = trial.suggest_int("iterations", 500, 2000)
    
    # Usar el gap correcto según el horizonte de predicción
    tscv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    rmse_scores = []
    
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # El resto igual que antes
        train_idx = sorted(train_idx)
        val_idx = sorted(val_idx)
        
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = CatBoostRegressor(
            learning_rate=learning_rate,
            depth=depth,
            iterations=iterations,
            random_seed=RANDOM_SEED,
            verbose=0,
            use_best_model=True
        )
        model.fit(X_train_cv, y_train_cv, eval_set=(X_val_cv, y_val_cv), early_stopping_rounds=50)
        y_pred = model.predict(X_val_cv)
        rmse_scores.append(sqrt(mean_squared_error(y_val_cv, y_pred)))
        logging.debug(f"[CatBoost][fold {fold_i}] best_iteration={model.get_best_iteration()}")
    
    mean_rmse = np.mean(rmse_scores)
    elapsed_time = time.perf_counter() - start_time
    logging.info(f"[CatBoost] Trial {trial.number}: RMSE={mean_rmse:.4f}, Time={elapsed_time:.2f}s")
    return mean_rmse

def objective_lgbm(trial, X, y, base_params=None):
    """Optimización de hiperparámetros para LightGBM con rangos refinados."""
    start_time = time.perf_counter()
    
    # Si tenemos parámetros base de RandomSearch, usarlos para refinar rangos
    if base_params and isinstance(base_params, dict):
        learning_rate = trial.suggest_float(
            "learning_rate", 
            max(base_params.get("learning_rate", 0.01) * 0.5, 0.0005),
            min(base_params.get("learning_rate", 0.01) * 2.0, 0.1), 
            log=True
        )
        max_depth = trial.suggest_int(
            "max_depth", 
            max(base_params.get("max_depth", 6) - 2, 3),
            min(base_params.get("max_depth", 6) + 2, 12)
        )
        n_estimators = trial.suggest_int(
            "n_estimators", 
            max(base_params.get("n_estimators", 1000) - 300, 300),
            min(base_params.get("n_estimators", 1000) + 300, 3000)
        )
        subsample = trial.suggest_float(
            "subsample", 
            max(base_params.get("subsample", 0.8) - 0.2, 0.5),
            min(base_params.get("subsample", 0.8) + 0.2, 1.0)
        )
    else:
        # Si no tenemos parámetros base, usar rangos amplios
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.05, log=True)
        max_depth = trial.suggest_int("max_depth", 4, 10)
        n_estimators = trial.suggest_int("n_estimators", 500, 2000)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
    
    # Usar el gap correcto según el horizonte de predicción
    tscv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    rmse_scores = []
    
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Asegurar que los índices se mantienen ordenados temporalmente
        train_idx = sorted(train_idx)
        val_idx = sorted(val_idx)
        
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = lgb.LGBMRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=subsample,
            random_state=RANDOM_SEED
        )
        model.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], eval_metric="rmse",
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)])
        y_pred = model.predict(X_val_cv)
        rmse_scores.append(sqrt(mean_squared_error(y_val_cv, y_pred)))
        if hasattr(model, "best_iteration_"):
            logging.debug(f"[LightGBM][fold {fold_i}] best_iteration={model.best_iteration_}")
    
    mean_rmse = np.mean(rmse_scores)
    elapsed_time = time.perf_counter() - start_time
    logging.info(f"[LightGBM] Trial {trial.number}: RMSE={mean_rmse:.4f}, Time={elapsed_time:.2f}s")
    return mean_rmse

def objective_xgboost(trial, X, y, base_params=None):
    """Optimización de hiperparámetros para XGBoost con rangos refinados."""
    start_time = time.perf_counter()
    
    # Si tenemos parámetros base de RandomSearch, usarlos para refinar rangos
    if base_params and isinstance(base_params, dict):
        learning_rate = trial.suggest_float(
            "learning_rate", 
            max(base_params.get("learning_rate", 0.01) * 0.5, 0.0005),
            min(base_params.get("learning_rate", 0.01) * 2.0, 0.1), 
            log=True
        )
        max_depth = trial.suggest_int(
            "max_depth", 
            max(base_params.get("max_depth", 6) - 2, 3),
            min(base_params.get("max_depth", 6) + 2, 12)
        )
        n_estimators = trial.suggest_int(
            "n_estimators", 
            max(base_params.get("n_estimators", 1000) - 300, 300),
            min(base_params.get("n_estimators", 1000) + 300, 3000)
        )
        subsample = trial.suggest_float(
            "subsample", 
            max(base_params.get("subsample", 0.8) - 0.2, 0.5),
            min(base_params.get("subsample", 0.8) + 0.2, 1.0)
        )
    else:
        # Si no tenemos parámetros base, usar rangos amplios
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.05, log=True)
        max_depth = trial.suggest_int("max_depth", 4, 10)
        n_estimators = trial.suggest_int("n_estimators", 500, 2000)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
    
    # Usar el gap correcto según el horizonte de predicción
    tscv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    rmse_scores = []
    
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Asegurar que los índices se mantienen ordenados temporalmente
        train_idx = sorted(train_idx)
        val_idx = sorted(val_idx)
        
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv)
        dval   = xgb.DMatrix(X_val_cv, label=y_val_cv)
        params = {
            "objective": "reg:squarederror",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "seed": RANDOM_SEED
        }
        evals_result = {}
        xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=n_estimators,
                              evals=[(dval, "eval")], early_stopping_rounds=50,
                              evals_result=evals_result, verbose_eval=False)
        best_iter = xgb_model.best_iteration
        y_pred = xgb_model.predict(dval, iteration_range=(0, best_iter+1))
        rmse_scores.append(sqrt(mean_squared_error(y_val_cv, y_pred)))
        logging.debug(f"[XGBoost][fold {fold_i}] best_iter={best_iter}")
    
    mean_rmse = np.mean(rmse_scores)
    elapsed_time = time.perf_counter() - start_time
    logging.info(f"[XGBoost] Trial {trial.number}: RMSE={mean_rmse:.4f}, Time={elapsed_time:.2f}s")
    return mean_rmse

def objective_mlp(trial, X, y, base_params=None):
    """Optimización de hiperparámetros para MLP con rangos refinados."""
    start_time = time.perf_counter()
    
    # Si tenemos parámetros base de RandomSearch, usarlos para refinar rangos
    if base_params and isinstance(base_params, dict):
        hidden_neurons = trial.suggest_int(
            "hidden_neurons", 
            max(base_params.get("hidden_neurons", 100) - 30, 50),
            min(base_params.get("hidden_neurons", 100) + 50, 250)
        )
        learning_rate_init = trial.suggest_float(
            "learning_rate_init", 
            max(base_params.get("learning_rate_init", 0.001) * 0.5, 0.0001),
            min(base_params.get("learning_rate_init", 0.001) * 2.0, 0.01), 
            log=True
        )
        max_iter = trial.suggest_int(
            "max_iter", 
            max(base_params.get("max_iter", 500) - 200, 200),
            min(base_params.get("max_iter", 500) + 200, 1200)
        )
    else:
        # Si no tenemos parámetros base, usar rangos amplios
        hidden_neurons = trial.suggest_int("hidden_neurons", 50, 200)
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
        max_iter = trial.suggest_int("max_iter", 200, 1000)
    
    hidden_layer_sizes = (hidden_neurons, hidden_neurons)
    
    # Usar el gap correcto según el horizonte de predicción
    tscv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    rmse_scores = []
    
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Asegurar que los índices se mantienen ordenados temporalmente
        train_idx = sorted(train_idx)
        val_idx = sorted(val_idx)
        
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             learning_rate_init=learning_rate_init,
                             max_iter=max_iter,
                             random_state=RANDOM_SEED,
                             early_stopping=True,
                             validation_fraction=0.2,
                             n_iter_no_change=25)
        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_val_cv)
        rmse_scores.append(sqrt(mean_squared_error(y_val_cv, y_pred)))
        logging.debug(f"[MLP][fold {fold_i}] n_iter_={model.n_iter_}")
    
    mean_rmse = np.mean(rmse_scores)
    elapsed_time = time.perf_counter() - start_time
    logging.info(f"[MLP] Trial {trial.number}: RMSE={mean_rmse:.4f}, Time={elapsed_time:.2f}s")
    return mean_rmse

def objective_svm(trial, X, y, base_params=None):
    """Optimización de hiperparámetros para SVM con rangos refinados."""
    start_time = time.perf_counter()
    
    # Si tenemos parámetros base de RandomSearch, usarlos para refinar rangos
    if base_params and isinstance(base_params, dict):
        C = trial.suggest_float(
            "C", 
            max(base_params.get("C", 1.0) * 0.5, 0.1),
            min(base_params.get("C", 1.0) * 2.0, 20.0), 
            log=True
        )
        epsilon = trial.suggest_float(
            "epsilon", 
            max(base_params.get("epsilon", 0.1) * 0.5, 0.01),
            min(base_params.get("epsilon", 0.1) * 2.0, 1.0)
        )
    else:
        # Si no tenemos parámetros base, usar rangos amplios
        C = trial.suggest_float("C", 0.1, 10, log=True)
        epsilon = trial.suggest_float("epsilon", 0.01, 0.5)
    
    # Usar el gap correcto según el horizonte de predicción
    tscv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    rmse_scores = []
    
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Asegurar que los índices se mantienen ordenados temporalmente
        train_idx = sorted(train_idx)
        val_idx = sorted(val_idx)
        
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = SVR(C=C, epsilon=epsilon)
        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_val_cv)
        rmse_scores.append(sqrt(mean_squared_error(y_val_cv, y_pred)))
    
    mean_rmse = np.mean(rmse_scores)
    elapsed_time = time.perf_counter() - start_time
    logging.info(f"[SVM] Trial {trial.number}: RMSE={mean_rmse:.4f}, Time={elapsed_time:.2f}s")
    return mean_rmse


def evaluate_backtest(algo_name, X_zone_A, y_zone_A, X_zone_B, y_zone_B, best_params, scaling_required=False):
    """Evalúa el modelo entrenado en Zona A sobre la Zona B (back-test)."""
    backtest_start = time.perf_counter()
    logging.info(f"[{algo_name}] Iniciando back-test en Zona B")
    
    # Escalar si es necesario
    scaler = None
    if scaling_required:
        scaler = StandardScaler()
        X_zone_A_scaled = pd.DataFrame(
            scaler.fit_transform(X_zone_A),
            columns=X_zone_A.columns,
            index=X_zone_A.index
        )
        X_zone_B_scaled = pd.DataFrame(
            scaler.transform(X_zone_B),
            columns=X_zone_B.columns,
            index=X_zone_B.index
        )
    else:
        X_zone_A_scaled = X_zone_A.copy()
        X_zone_B_scaled = X_zone_B.copy()
    
    # Crear y entrenar modelo
    if algo_name == "CatBoost":
        # Usar GPU para CatBoost si está disponible
        task_type = "GPU" if has_gpu else "CPU"
        devices = '0' if has_gpu else None
        model = CatBoostRegressor(
            learning_rate=best_params.get("learning_rate", 0.01),
            depth=best_params.get("depth", 6),
            iterations=best_params.get("iterations", 1000),
            random_seed=RANDOM_SEED,
            verbose=0,
            task_type=task_type,
            devices=devices
        )
        model.fit(X_zone_A_scaled, y_zone_A)
        preds_zone_B = model.predict(X_zone_B_scaled)
    elif algo_name == "LightGBM":
        model = lgb.LGBMRegressor(
            learning_rate=best_params.get("learning_rate", 0.01),
            max_depth=best_params.get("max_depth", 6),
            n_estimators=best_params.get("n_estimators", 1000),
            random_state=RANDOM_SEED
        )
        model.fit(X_zone_A_scaled, y_zone_A)
        preds_zone_B = model.predict(X_zone_B_scaled)
    elif algo_name == "XGBoost":
        # Usar GPU para XGBoost si está disponible
        tree_method = "gpu_hist" if has_gpu else "auto"
        gpu_id = 0 if has_gpu else None

        model = xgb.XGBRegressor(
            learning_rate=best_params.get("learning_rate", 0.01),
            max_depth=best_params.get("max_depth", 6),
            n_estimators=best_params.get("n_estimators", 1000),
            subsample=best_params.get("subsample", 0.8),
            random_state=RANDOM_SEED,
            tree_method=tree_method,
            gpu_id=gpu_id
        )
        model.fit(X_zone_A_scaled, y_zone_A)
        preds_zone_B = model.predict(X_zone_B_scaled)
    elif algo_name == "MLP":
        hidden_neurons = best_params.get("hidden_neurons", 100)
        model = MLPRegressor(
            hidden_layer_sizes=(hidden_neurons, hidden_neurons),
            learning_rate_init=best_params.get("learning_rate_init", 0.001),
            max_iter=best_params.get("max_iter", 500),
            random_state=RANDOM_SEED
        )
        model.fit(X_zone_A_scaled, y_zone_A)
        preds_zone_B = model.predict(X_zone_B_scaled)
    elif algo_name == "SVM":
        model = SVR(
            C=best_params.get("C", 1.0),
            epsilon=best_params.get("epsilon", 0.1)
        )
        model.fit(X_zone_A_scaled, y_zone_A)
        preds_zone_B = model.predict(X_zone_B_scaled)
    elif algo_name == "LSTM":
        # Parámetros LSTM
        units = best_params.get("units", 64)
        learning_rate = best_params.get("learning_rate", 0.001)
        dropout_rate = best_params.get("dropout_rate", 0.2)
        sequence_length = best_params.get("sequence_length", 10)
        
        # Crear secuencias para entrenamiento
        X_train_seq, y_train_seq = create_sequences(X_zone_A_scaled, y_zone_A, sequence_length)
        
        # Crear modelo LSTM
        n_features = X_zone_A_scaled.shape[1]
        model = Sequential([
            LSTM(units=units, input_shape=(sequence_length, n_features), return_sequences=False),
            Dropout(dropout_rate),
            Dense(1)
        ])
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse'
        )
        
        # Entrenar modelo
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Dividir datos de entrenamiento para validación interna
        val_split = 0.1
        val_samples = int(len(X_train_seq) * val_split)
        if val_samples > 0:
            X_train_lstm = X_train_seq[:-val_samples]
            y_train_lstm = y_train_seq[:-val_samples]
            X_val_lstm = X_train_seq[-val_samples:]
            y_val_lstm = y_train_seq[-val_samples:]
            
            model.fit(
                X_train_lstm, y_train_lstm,
                validation_data=(X_val_lstm, y_val_lstm),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
        else:
            model.fit(
                X_train_seq, y_train_seq,
                epochs=50,
                batch_size=32,
                verbose=0
            )
        
        # Crear secuencias para Zona B
        if len(X_zone_B_scaled) > sequence_length:
            # Crear secuencias completas para predicción
            X_zone_B_seq, y_zone_B_actual = create_sequences(X_zone_B_scaled, y_zone_B, sequence_length)
            
            # Predecir
            if len(X_zone_B_seq) > 0:
                preds_seq = model.predict(X_zone_B_seq, verbose=0).flatten()
                
                # Ajustar las predicciones al formato original
                preds_zone_B = np.full(len(y_zone_B), np.nan)
                preds_zone_B[sequence_length:sequence_length+len(preds_seq)] = preds_seq
                
                # Para las primeras sequence_length filas, usar el último modelo para predecir
                if sequence_length < len(X_zone_B_scaled):
                    for i in range(sequence_length):
                        # Usar datos de Zona A + inicio de Zona B para secuencias iniciales
                        if i < len(X_zone_A_scaled):
                            combined_data = pd.concat([
                                X_zone_A_scaled.iloc[-sequence_length+i:],
                                X_zone_B_scaled.iloc[:i]
                            ])
                            if len(combined_data) == sequence_length:
                                combined_seq = combined_data.values.reshape(1, sequence_length, n_features)
                                preds_zone_B[i] = model.predict(combined_seq, verbose=0)[0][0]
            else:
                # Si no hay suficientes datos para crear secuencias, usar nulos
                preds_zone_B = np.full(len(y_zone_B), np.nan)
        else:
            # Si Zona B es más pequeña que sequence_length, usar nulos
            preds_zone_B = np.full(len(y_zone_B), np.nan)
        
        # Convertir NaN a 0 para cálculos de métricas
        preds_zone_B = np.nan_to_num(preds_zone_B)
        
        # Limpiar sesión de Keras
        tf.keras.backend.clear_session()
    else:
        logging.error(f"[{algo_name}] Algoritmo no soportado para back-test.")
        return None
    
    # Calcular métricas
    metrics = calcular_metricas_basicas(y_zone_B, preds_zone_B)
    
    backtest_time = time.perf_counter() - backtest_start
    logging.info(f"[{algo_name}] Back-test completado en {backtest_time:.2f}s")
    logging.info(f"[{algo_name}] Métricas Back-test (Zona B):")
    logging.info(f"  - RMSE: {metrics['RMSE']:.4f}")
    logging.info(f"  - MAE: {metrics['MAE']:.4f}")
    logging.info(f"  - R2: {metrics['R2']:.4f}")
    logging.info(f"  - SMAPE: {metrics['SMAPE']:.4f}")
    
    return {
        'model': model,
        'predictions': preds_zone_B,
        'metrics': metrics,
        'scaler': scaler
    }

def evaluate_holdout(algo_name, X_zone_A, y_zone_A, X_zone_B, y_zone_B, X_zone_C, y_zone_C, best_params, scaling_required=False):
    """Evalúa el modelo entrenado en Zona A + B sobre la Zona C (hold-out final)."""
    holdout_start = time.perf_counter()
    logging.info(f"[{algo_name}] Iniciando hold-out final en Zona C")
    
    # Combinar Zona A + B para entrenamiento
    X_zone_AB = pd.concat([X_zone_A, X_zone_B])
    y_zone_AB = pd.concat([y_zone_A, y_zone_B])
    
    # Escalar si es necesario
    scaler = None
    if scaling_required:
        scaler = StandardScaler()
        X_zone_AB_scaled = pd.DataFrame(
            scaler.fit_transform(X_zone_AB),
            columns=X_zone_AB.columns,
            index=X_zone_AB.index
        )
        X_zone_C_scaled = pd.DataFrame(
            scaler.transform(X_zone_C),
            columns=X_zone_C.columns,
            index=X_zone_C.index
        )
    else:
        X_zone_AB_scaled = X_zone_AB.copy()
        X_zone_C_scaled = X_zone_C.copy()
    
    # Crear y entrenar modelo
    if algo_name == "CatBoost":
        # Usar GPU para CatBoost si está disponible
        task_type = "GPU" if has_gpu else "CPU"
        devices = '0' if has_gpu else None
        model = CatBoostRegressor(
            learning_rate=best_params.get("learning_rate", 0.01),
            depth=best_params.get("depth", 6),
            iterations=best_params.get("iterations", 1000),
            random_seed=RANDOM_SEED,
            verbose=0,
            task_type=task_type,
            devices=devices
        )
        model.fit(X_zone_AB_scaled, y_zone_AB)
        preds_zone_C = model.predict(X_zone_C_scaled)
    elif algo_name == "LightGBM":
        model = lgb.LGBMRegressor(
            learning_rate=best_params.get("learning_rate", 0.01),
            max_depth=best_params.get("max_depth", 6),
            n_estimators=best_params.get("n_estimators", 1000),
            random_state=RANDOM_SEED
        )
        model.fit(X_zone_AB_scaled, y_zone_AB)
        preds_zone_C = model.predict(X_zone_C_scaled)
    elif algo_name == "XGBoost":
        # Usar GPU para XGBoost si está disponible
        tree_method = "gpu_hist" if has_gpu else "auto"
        gpu_id = 0 if has_gpu else None
        # Crear modelo XGBoost
        model = xgb.XGBRegressor(
            learning_rate=best_params.get("learning_rate", 0.01),
            max_depth=best_params.get("max_depth", 6),
            n_estimators=best_params.get("n_estimators", 1000),
            subsample=best_params.get("subsample", 0.8),
            random_state=RANDOM_SEED,
            tree_method=tree_method,
            gpu_id=gpu_id
        )
        model.fit(X_zone_AB_scaled, y_zone_AB)
        preds_zone_C = model.predict(X_zone_C_scaled)
    elif algo_name == "MLP":
        hidden_neurons = best_params.get("hidden_neurons", 100)
        model = MLPRegressor(
            hidden_layer_sizes=(hidden_neurons, hidden_neurons),
            learning_rate_init=best_params.get("learning_rate_init", 0.001),
            max_iter=best_params.get("max_iter", 500),
            random_state=RANDOM_SEED
        )
        model.fit(X_zone_AB_scaled, y_zone_AB)
        preds_zone_C = model.predict(X_zone_C_scaled)
    elif algo_name == "SVM":
        model = SVR(
            C=best_params.get("C", 1.0),
            epsilon=best_params.get("epsilon", 0.1)
        )
        model.fit(X_zone_AB_scaled, y_zone_AB)
        preds_zone_C = model.predict(X_zone_C_scaled)
    elif algo_name == "LSTM":
        # Parámetros LSTM
        units = best_params.get("units", 64)
        learning_rate = best_params.get("learning_rate", 0.001)
        dropout_rate = best_params.get("dropout_rate", 0.2)
        sequence_length = best_params.get("sequence_length", 10)
        
        # Crear secuencias para entrenamiento
        X_train_seq, y_train_seq = create_sequences(X_zone_AB_scaled, y_zone_AB, sequence_length)
        
        # Crear modelo LSTM
        n_features = X_zone_AB_scaled.shape[1]
        model = Sequential([
            LSTM(units=units, input_shape=(sequence_length, n_features), return_sequences=False),
            Dropout(dropout_rate),
            Dense(1)
        ])
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse'
        )
        
        # Entrenar modelo
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Dividir datos de entrenamiento para validación interna
        val_split = 0.1
        val_samples = int(len(X_train_seq) * val_split)
        if val_samples > 0:
            X_train_lstm = X_train_seq[:-val_samples]
            y_train_lstm = y_train_seq[:-val_samples]
            X_val_lstm = X_train_seq[-val_samples:]
            y_val_lstm = y_train_seq[-val_samples:]
            
            model.fit(
                X_train_lstm, y_train_lstm,
                validation_data=(X_val_lstm, y_val_lstm),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
        else:
            model.fit(
                X_train_seq, y_train_seq,
                epochs=50,
                batch_size=32,
                verbose=0
            )
        
        # Crear secuencias para Zona C
        if len(X_zone_C_scaled) > sequence_length:
            # Crear secuencias completas para predicción
            X_zone_C_seq, y_zone_C_actual = create_sequences(X_zone_C_scaled, y_zone_C, sequence_length)
            
            # Predecir
            if len(X_zone_C_seq) > 0:
                preds_seq = model.predict(X_zone_C_seq, verbose=0).flatten()
                
                # Ajustar las predicciones al formato original
                preds_zone_C = np.full(len(y_zone_C), np.nan)
                preds_zone_C[sequence_length:sequence_length+len(preds_seq)] = preds_seq
                
                # Para las primeras sequence_length filas, usar el último modelo para predecir
                if sequence_length < len(X_zone_C_scaled):
                    for i in range(sequence_length):
                        # Usar datos de Zona AB + inicio de Zona C para secuencias iniciales
                        if i < len(X_zone_AB_scaled):
                            combined_data = pd.concat([
                                X_zone_AB_scaled.iloc[-sequence_length+i:],
                                X_zone_C_scaled.iloc[:i]
                            ])
                            if len(combined_data) == sequence_length:
                                combined_seq = combined_data.values.reshape(1, sequence_length, n_features)
                                preds_zone_C[i] = model.predict(combined_seq, verbose=0)[0][0]
            else:
                # Si no hay suficientes datos para crear secuencias, usar nulos
                preds_zone_C = np.full(len(y_zone_C), np.nan)
        else:
            # Si Zona C es más pequeña que sequence_length, usar nulos
            preds_zone_C = np.full(len(y_zone_C), np.nan)
        
        # Convertir NaN a 0 para cálculos de métricas
        preds_zone_C = np.nan_to_num(preds_zone_C)
        
        # Limpiar sesión de Keras
        tf.keras.backend.clear_session()
    else:
        logging.error(f"[{algo_name}] Algoritmo no soportado para hold-out.")
        return None
    
    # Calcular métricas
    metrics = calcular_metricas_basicas(y_zone_C, preds_zone_C)
    
    holdout_time = time.perf_counter() - holdout_start
    logging.info(f"[{algo_name}] Hold-out completado en {holdout_time:.2f}s")
    logging.info(f"[{algo_name}] Métricas Hold-out (Zona C):")
    logging.info(f"  - RMSE: {metrics['RMSE']:.4f}")
    logging.info(f"  - MAE: {metrics['MAE']:.4f}")
    logging.info(f"  - R2: {metrics['R2']:.4f}")
    logging.info(f"  - SMAPE: {metrics['SMAPE']:.4f}")
    
    return {
        'model': model,
        'predictions': preds_zone_C,
        'metrics': metrics,
        'scaler': scaler
    }

# -----------------------------------------------------------------
# FUNCIONES PARA FORECAST Y ENTRENAMIENTO EXTENDIDO
# -----------------------------------------------------------------
def forecast_future(model, last_row, forecast_horizon=FORECAST_HORIZON_1MONTH, algo_name=None, sequence_length=None, X_recent=None):
    """
    Genera predicciones para los próximos 'forecast_horizon' días.
    Para modelos estándar, se asume que las características permanecen constantes.
    Para LSTM, utiliza un enfoque de ventana deslizante para generar predicciones secuenciales.
    
    Args:
        model: Modelo entrenado
        last_row: Última fila de features disponible
        forecast_horizon: Horizonte de predicción en días hábiles (20 para 1MONTH, 60 para 3MONTHS)
        algo_name: Nombre del algoritmo (para manejar LSTM específicamente)
        sequence_length: Longitud de secuencia para LSTM
        X_recent: Datos recientes para crear ventana deslizante en LSTM
        
    Returns:
        Lista de predicciones para cada día del horizonte
    """
    future_predictions = []
    
    if algo_name == "LSTM" and sequence_length is not None and X_recent is not None:
        # Para modelos LSTM necesitamos un enfoque especial con ventana deslizante
        logging.info(f"Generando pronóstico para LSTM con sequence_length={sequence_length}")
        
        # Obtener los últimos 'sequence_length' datos como ventana inicial
        if len(X_recent) >= sequence_length:
            # Tomar los últimos 'sequence_length' registros como ventana inicial
            window = X_recent[-sequence_length:].copy()
            
            # Pronóstico paso a paso
            for step in range(forecast_horizon):
                # Reshape para formato LSTM [samples, timesteps, features]
                window_reshaped = window.reshape(1, sequence_length, window.shape[1])
                
                # Predecir el siguiente valor
                try:
                    next_pred = model.predict(window_reshaped, verbose=0)[0][0]
                except Exception as e:
                    logging.error(f"Error al predecir con LSTM en paso {step}: {e}")
                    next_pred = np.nan
                
                future_predictions.append(float(next_pred))
                
                # Actualizar la ventana deslizante:
                # Eliminar el primer registro y añadir una nueva fila con last_row como características
                # (asumimos que las características permanecen constantes, pero el target cambia)
                window = np.vstack([window[1:], last_row.values.reshape(1, -1)])
        else:
            # Si no tenemos suficientes datos para la secuencia, devolvemos valores nulos
            logging.warning(f"No hay suficientes datos para generar pronóstico LSTM (se necesitan al menos {sequence_length} registros)")
            future_predictions = [np.nan] * forecast_horizon
            
    elif hasattr(model, 'predict'):
        # Para modelos estándar (no secuenciales)
        logging.info(f"Generando pronóstico para modelo estándar")
        current_features = last_row.values.reshape(1, -1)
        
        for step in range(forecast_horizon):
            try:
                # La mayoría de modelos sklearn usan este formato
                pred = model.predict(current_features)[0]
                if isinstance(pred, np.ndarray) and len(pred) > 0:
                    pred = pred[0]
            except Exception as e:
                logging.error(f"Error al predecir en paso {step}: {e}")
                # Intentar formato alternativo (como algunos modelos de keras)
                try:
                    pred = model.predict(current_features, verbose=0)[0][0]
                except:
                    pred = np.nan
                    
            future_predictions.append(float(pred))
    else:
        # Si el modelo no tiene método predict, devolvemos valores nulos
        logging.warning(f"El modelo no tiene método predict")
        future_predictions = [np.nan] * forecast_horizon
    
    # Verificar que tenemos las predicciones esperadas
    if len(future_predictions) != forecast_horizon:
        logging.warning(f"Número incorrecto de predicciones: {len(future_predictions)} en lugar de {forecast_horizon}")
        # Completar o recortar según sea necesario
        if len(future_predictions) < forecast_horizon:
            future_predictions.extend([np.nan] * (forecast_horizon - len(future_predictions)))
        else:
            future_predictions = future_predictions[:forecast_horizon]
    
    # Reemplazar cualquier valor no finito (inf, -inf, nan) con np.nan para consistencia
    future_predictions = [np.nan if not np.isfinite(x) else x for x in future_predictions]
    
    logging.info(f"Pronóstico generado con {sum(~np.isnan(future_predictions))} valores válidos de {forecast_horizon}")
    
    return future_predictions

def refinamiento_local(model, X, y, n_days=LOCAL_REFINEMENT_DAYS):
    """
    Crea una copia del modelo original y lo refina con los datos más recientes
    """
    # Crear copia del modelo original para no afectar al original
    if hasattr(model, 'get_params'):
        refined_model = model.__class__(**model.get_params())
    else:
        # Si no se puede clonar directamente, intentar otra aproximación
        import copy
        try:
            refined_model = copy.deepcopy(model)
        except:
            # Si todo falla, usar el modelo original pero advertir
            logging.warning(f"No se pudo clonar el modelo. Se usará el original.")
            refined_model = model
    
    # Seleccionar los últimos n_days para refinamiento
    if len(X) <= n_days:
        local_X = X.copy()
        local_y = y.copy()
    else:
        local_X = X.tail(n_days).copy()
        local_y = y.tail(n_days).copy()
    
    # Split 80/20 para refinamiento
    train_size = int(len(local_X) * TRAIN_TEST_SPLIT_RATIO)
    X_local_train = local_X.iloc[:train_size]
    y_local_train = local_y.iloc[:train_size]
    
    # Entrenar modelo refinado
    if hasattr(refined_model, 'fit'):
        refined_model.fit(X_local_train, y_local_train)
    
    return refined_model  # Devolvemos el modelo refinado, no el original


def calcular_metricas_basicas(y_true, y_pred):
    """
    Calcula métricas básicas de regresión.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        
    Returns:
        dict: Diccionario con métricas calculadas
    """
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # SMAPE (Symmetric Mean Absolute Percentage Error)
    smape = 100 * np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'SMAPE': smape
    }

def optimize_and_train_extended(algo_name, objective_func, X_zone_A, y_zone_A, X_zone_B, y_zone_B, X_zone_C, y_zone_C, 
                               forecast_horizon=FORECAST_HORIZON_1MONTH, top_n=3):
    """
    Optimiza el modelo usando el enfoque de tres zonas:
      - Zona A (70%): RandomSearch + Optuna para ajuste de hiperparámetros
      - Zona B (20%): Back-test externo
      - Zona C (10%): Hold-out final
    """
    start_time = time.perf_counter()
    
    # 1. RANDOMIZEDSEARCH EN ZONA A
    logging.info(f"[{algo_name}] Paso 1: RandomizedSearchCV en Zona A")
    tscv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    best_params_random = run_randomized_search(algo_name, X_zone_A, y_zone_A, tscv)
    
    # 2. OPTUNA EN ZONA A (REFINAMIENTO)
    logging.info(f"[{algo_name}] Paso 2: Optuna en Zona A (refinamiento)")
    study = optuna.create_study(direction="minimize")
    
    # Adaptar función objetivo para este algoritmo
    def objective(trial):
        return objective_func(trial, X_zone_A, y_zone_A, best_params_random)
    
    # Optimización con Optuna
    logging.info(f"[{algo_name}] Iniciando optimización con {args.optuna_trials} trials")
    study.optimize(objective, n_trials=args.optuna_trials)
    logging.info(f"[{algo_name}] Mejor RMSE en CV (Optuna): {study.best_value:.4f}")
    logging.info(f"[{algo_name}] Mejores hiperparámetros: {study.best_params}")
    
    # 3. BACK-TEST EN ZONA B
    logging.info(f"[{algo_name}] Paso 3: Back-test en Zona B")
    backtest_results = evaluate_backtest(
        algo_name, 
        X_zone_A, y_zone_A, 
        X_zone_B, y_zone_B, 
        study.best_params, 
        scaling_required=SCALING_REQUIRED_MODELS.get(algo_name, False)
    )
    
    # 4. HOLD-OUT EN ZONA C
    logging.info(f"[{algo_name}] Paso 4: Hold-out en Zona C")
    holdout_results = evaluate_holdout(
        algo_name, 
        X_zone_A, y_zone_A, 
        X_zone_B, y_zone_B, 
        X_zone_C, y_zone_C, 
        study.best_params, 
        scaling_required=SCALING_REQUIRED_MODELS.get(algo_name, False)
    )
    
    # 5. MODELO FINAL (100% DE LOS DATOS)
    logging.info(f"[{algo_name}] Paso 5: Modelo final (100% de los datos)")
    
    # Combinar todos los datos
    X_all = pd.concat([X_zone_A, X_zone_B, X_zone_C])
    y_all = pd.concat([y_zone_A, y_zone_B, y_zone_C])
    
    # Escalar si es necesario
    scaling_required = SCALING_REQUIRED_MODELS.get(algo_name, False)
    scaler = None
    if scaling_required:
        scaler = StandardScaler()
        X_all_scaled = pd.DataFrame(
            scaler.fit_transform(X_all),
            columns=X_all.columns,
            index=X_all.index
        )
    else:
        X_all_scaled = X_all.copy()
    
    # Crear modelo final con mejores parámetros
    if algo_name == "CatBoost":
        # Usar GPU para CatBoost si está disponible
        task_type = "GPU" if has_gpu else "CPU"
        devices = '0' if has_gpu else None
        final_model = CatBoostRegressor(
            learning_rate=study.best_params.get("learning_rate", 0.01),
            depth=study.best_params.get("depth", 6),
            iterations=study.best_params.get("iterations", 1000),
            random_seed=RANDOM_SEED,
            verbose=0,
            task_type=task_type,
            devices=devices
        )
    elif algo_name == "LightGBM":
        final_model = lgb.LGBMRegressor(
            learning_rate=study.best_params.get("learning_rate", 0.01),
            max_depth=study.best_params.get("max_depth", 6),
            n_estimators=study.best_params.get("n_estimators", 1000),
            random_state=RANDOM_SEED
        )
    elif algo_name == "XGBoost":
        # Usar GPU para XGBoost si está disponible
        tree_method = "gpu_hist" if has_gpu else "auto"
        gpu_id = 0 if has_gpu else None
        # Crear modelo XGBoost
        final_model = xgb.XGBRegressor(
            learning_rate=study.best_params.get("learning_rate", 0.01),
            max_depth=study.best_params.get("max_depth", 6),
            n_estimators=study.best_params.get("n_estimators", 1000),
            subsample=study.best_params.get("subsample", 0.8),
            random_state=RANDOM_SEED,
            tree_method=tree_method,
            gpu_id=gpu_id
        )
    elif algo_name == "MLP":
        hidden_neurons = study.best_params.get("hidden_neurons", 100)
        final_model = MLPRegressor(
            hidden_layer_sizes=(hidden_neurons, hidden_neurons),
            learning_rate_init=study.best_params.get("learning_rate_init", 0.001),
            max_iter=study.best_params.get("max_iter", 500),
            random_state=RANDOM_SEED
        )
    elif algo_name == "SVM":
        final_model = SVR(
            C=study.best_params.get("C", 1.0),
            epsilon=study.best_params.get("epsilon", 0.1)
        )
    elif algo_name == "LSTM":
        # Parámetros LSTM
        units = study.best_params.get("units", 64)
        learning_rate = study.best_params.get("learning_rate", 0.001)
        dropout_rate = study.best_params.get("dropout_rate", 0.2)
        sequence_length = study.best_params.get("sequence_length", 10)
        
        # Crear secuencias para entrenamiento con todos los datos
        X_all_seq, y_all_seq = create_sequences(X_all_scaled, y_all, sequence_length)
        
        # Crear modelo LSTM
        n_features = X_all_scaled.shape[1]
        final_model = Sequential([
            LSTM(units=units, input_shape=(sequence_length, n_features), return_sequences=False),
            Dropout(dropout_rate),
            Dense(1)
        ])
        final_model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse'
        )
        
        # Entrenar modelo final
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Dividir para validación
        val_split = 0.1
        val_samples = int(len(X_all_seq) * val_split)
        if val_samples > 0:
            X_train_lstm = X_all_seq[:-val_samples]
            y_train_lstm = y_all_seq[:-val_samples]
            X_val_lstm = X_all_seq[-val_samples:]
            y_val_lstm = y_all_seq[-val_samples:]
            
            final_model.fit(
                X_train_lstm, y_train_lstm,
                validation_data=(X_val_lstm, y_val_lstm),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
        else:
            final_model.fit(
                X_all_seq, y_all_seq,
                epochs=50,
                batch_size=32,
                verbose=0
            )
        
        # Guardar modelo LSTM
        model_filename = f"{algo_name.lower()}_best.pkl"
        model_path = os.path.join(args.output_dir, model_filename)
        
        lstm_model_data = {
            'model': final_model,
            'scaler': scaler,
            'params': study.best_params,
            'random_search_params': best_params_random,
            'backtest_metrics': backtest_results['metrics'] if backtest_results else None,
            'holdout_metrics': holdout_results['metrics'] if holdout_results else None,
            'sequence_length': sequence_length
        }
        
        # Guardar modelo
        try:
            # Guardar modelo de keras separadamente
            model_keras_path = os.path.join(args.output_dir, f"{algo_name.lower()}_model.h5")
            final_model.save(model_keras_path)
            
            # Guardar datos adicionales
            lstm_model_data['model'] = 'saved_separately'  # Placeholder
            joblib.dump(lstm_model_data, model_path)
            logging.info(f"[{algo_name}] Modelo LSTM guardado en formato H5: {model_keras_path}")
            logging.info(f"[{algo_name}] Datos adicionales guardados en: {model_path}")
        except Exception as e:
            logging.error(f"[{algo_name}] Error al guardar modelo LSTM: {e}")
            # Intentar guardar todo junto como respaldo
            joblib.dump(lstm_model_data, model_path)
            logging.info(f"[{algo_name}] Se ha guardado todo en un solo archivo: {model_path}")
    else:
        logging.error(f"Algoritmo {algo_name} no soportado.")
        return None
    
    # Este bloque se ejecuta para todos los modelos (incluyendo LSTM)
    
    # Para modelos que no son LSTM, entrenar aquí
    if algo_name != "LSTM":
        # Entrenar modelo final
        final_model.fit(X_all_scaled, y_all)
        
        # Guardar modelo
        model_filename = f"{algo_name.lower()}_best.pkl"
        model_path = os.path.join(args.output_dir, model_filename)
        
        # Guardar modelo, scaler y parámetros
        model_data = {
            'model': final_model,
            'scaler': scaler,
            'params': study.best_params,
            'random_search_params': best_params_random,
            'backtest_metrics': backtest_results['metrics'] if backtest_results else None,
            'holdout_metrics': holdout_results['metrics'] if holdout_results else None
        }
        joblib.dump(model_data, model_path)
        logging.info(f"[{algo_name}] Modelo final guardado en: {model_path}")
    
    # Guardar historial de hiperparámetros (para todos los modelos)
    save_hyperparameters_to_csv(
        algo_name, 
        best_params_random, 
        study.best_params, 
        backtest_results['metrics'] if backtest_results else {}, 
        holdout_results['metrics'] if holdout_results else {}
    )
    
    # Generar forecast (para todos los modelos)
    if algo_name == "LSTM":
        # Para LSTM usar parámetros específicos
        sequence_length = study.best_params.get("sequence_length", 10)
        if len(X_all_scaled) >= sequence_length:
            future_preds = forecast_future(
                final_model, 
                X_all_scaled.iloc[-1], 
                forecast_horizon, 
                algo_name="LSTM",
                sequence_length=sequence_length,
                X_recent=X_all_scaled.values
            )
        else:
            future_preds = [np.nan] * forecast_horizon
    else:
        # Para otros modelos
        last_row = X_all_scaled.iloc[-1]
        future_preds = forecast_future(final_model, last_row, forecast_horizon)
    
    last_date = X_all.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')
    
    # Escalar cada zona individualmente para todos los modelos
    if scaling_required and scaler is not None:
        X_zone_A_scaled = pd.DataFrame(
            scaler.transform(X_zone_A),
            columns=X_zone_A.columns,
            index=X_zone_A.index
        )
        X_zone_B_scaled = pd.DataFrame(
            scaler.transform(X_zone_B),
            columns=X_zone_B.columns,
            index=X_zone_B.index
        )
        X_zone_C_scaled = pd.DataFrame(
            scaler.transform(X_zone_C),
            columns=X_zone_C.columns,
            index=X_zone_C.index
        )
    else:
        X_zone_A_scaled = X_zone_A.copy()
        X_zone_B_scaled = X_zone_B.copy()
        X_zone_C_scaled = X_zone_C.copy()
    
    # Crear DataFrames para cada período
    # 1. Sección Training (Zona A)
    if algo_name == "LSTM":
        # Para LSTM necesitamos manejar las predicciones de manera diferente
        sequence_length = study.best_params.get("sequence_length", 10)
        if len(X_zone_A) > sequence_length:
            X_zone_A_seq, y_zone_A_actual = create_sequences(X_zone_A_scaled, y_zone_A, sequence_length)
            if len(X_zone_A_seq) > 0:
                train_preds_seq = final_model.predict(X_zone_A_seq, verbose=0).flatten()
                train_preds = np.full(len(y_zone_A), np.nan)
                train_preds[sequence_length:sequence_length+len(train_preds_seq)] = train_preds_seq
                train_preds = np.nan_to_num(train_preds)
            else:
                train_preds = np.zeros(len(y_zone_A))
        else:
            train_preds = np.zeros(len(y_zone_A))
    else:
        # Para otros modelos
        train_preds = final_model.predict(X_zone_A_scaled) if len(X_zone_A) > 0 else []
    
    train_metrics = calcular_metricas_basicas(y_zone_A, train_preds) if len(X_zone_A) > 0 else {}
    
    df_train = pd.DataFrame({
        "date": X_zone_A.index,
        "Valor_Real": y_zone_A,
        "Valor_Predicho": train_preds,
        "Modelo": algo_name,
        "Version": f"Three-Zones",
        "RMSE": train_metrics.get('RMSE', np.nan),
        "MAE": train_metrics.get('MAE', np.nan),
        "R2": train_metrics.get('R2', np.nan),
        "SMAPE": train_metrics.get('SMAPE', np.nan),
        "Hyperparámetros": json.dumps(study.best_params),
        "Tipo_Mercado": args.tipo_mercado,
        "Periodo": "Training"
    })
    
    # 2. Sección Evaluación (Zona B - Backtest)
    eval_preds = backtest_results['predictions'] if backtest_results else []
    eval_metrics = backtest_results['metrics'] if backtest_results else {}
    
    df_eval = pd.DataFrame({
        "date": X_zone_B.index,
        "Valor_Real": y_zone_B,
        "Valor_Predicho": eval_preds,
        "Modelo": algo_name,
        "Version": f"Three-Zones",
        "RMSE": eval_metrics.get('RMSE', np.nan),
        "MAE": eval_metrics.get('MAE', np.nan),
        "R2": eval_metrics.get('R2', np.nan),
        "SMAPE": eval_metrics.get('SMAPE', np.nan),
        "Hyperparámetros": json.dumps(study.best_params),
        "Tipo_Mercado": args.tipo_mercado,
        "Periodo": "Evaluacion"
    })
    
    # 3. Sección Test (Zona C - Holdout)
    test_preds = holdout_results['predictions'] if holdout_results else []
    test_metrics = holdout_results['metrics'] if holdout_results else {}
    
    df_test = pd.DataFrame({
        "date": X_zone_C.index,
        "Valor_Real": y_zone_C,
        "Valor_Predicho": test_preds,
        "Modelo": algo_name,
        "Version": f"Three-Zones",
        "RMSE": test_metrics.get('RMSE', np.nan),
        "MAE": test_metrics.get('MAE', np.nan),
        "R2": test_metrics.get('R2', np.nan),
        "SMAPE": test_metrics.get('SMAPE', np.nan),
        "Hyperparámetros": json.dumps(study.best_params),
        "Tipo_Mercado": args.tipo_mercado,
        "Periodo": "Test"
    })
    
    # 4. Sección Forecast (predicción futura)
    df_forecast = pd.DataFrame({
        "date": future_dates,
        "Valor_Real": [np.nan] * forecast_horizon,
        "Valor_Predicho": future_preds,
        "Modelo": algo_name,
        "Version": f"Three-Zones",
        "RMSE": [np.nan] * forecast_horizon,
        "MAE": [np.nan] * forecast_horizon,
        "R2": [np.nan] * forecast_horizon,
        "SMAPE": [np.nan] * forecast_horizon,
        "Hyperparámetros": json.dumps(study.best_params),
        "Tipo_Mercado": args.tipo_mercado,
        "Periodo": "Forecast"
    })
    
    # Concatenar todos los DataFrames
    df_all = pd.concat([df_train, df_eval, df_test, df_forecast], ignore_index=True)
    
    # Guardar resultados específicos de este modelo
    model_csv_path = os.path.join(RESULTS_DIR, f"{args.tipo_mercado.lower()}_{algo_name.lower()}_three_zones.csv")
    df_all.to_csv(model_csv_path, index=False)
    logging.info(f"[{algo_name}] CSV guardado: {model_csv_path}")
    
    # ====================================================================
    # MODIFICACIÓN: Filtrar datos para gráficos (solo Zona B, C y Forecast)
    # ====================================================================
    # Filtrar para excluir la Zona A (Training) en los gráficos
    df_for_plots = df_all[df_all['Periodo'] != 'Training'].copy()
    logging.info(f"[{algo_name}] Generando gráficos solo con Zona B, Zona C y Forecast")
    logging.info(f"[{algo_name}] Datos para gráficos: {len(df_for_plots)} registros (excluyendo {len(df_train)} de Training)")
    
    # Visualizaciones (SOLO CON ZONA B, C Y FORECAST)
    # 1. Gráfico principal con Zona B, C y Forecast únicamente
    chart_path = os.path.join(IMG_CHARTS_DIR, f"{algo_name.lower()}_three_zones_eval_test_forecast.png")
    plot_real_vs_pred(
        df_for_plots,
        title=f"Back-test, Hold-out y Forecast - {algo_name} (Zonas B, C y Forecast)",
        model_name=algo_name,
        output_path=chart_path
    )
    logging.info(f"[{algo_name}] Gráfico principal guardado (sin Zona A): {chart_path}")
    
    # 2. Gráfico detallado del último mes + forecast
    # Obtener el último mes de datos reales (aproximadamente 20-25 días hábiles)
    days_before_forecast = 25  # Un mes de días hábiles aproximadamente
    
    # Combinar datos de Zona C (últimos datos reales) con forecast
    df_last_month_and_forecast = pd.concat([
        df_test.tail(min(days_before_forecast, len(df_test))),  # Últimos días de Zona C
        df_forecast  # Todo el forecast
    ], ignore_index=True)
    
    # Crear gráfico específico para horizonte de predicción
    forecast_detail_path = os.path.join(IMG_CHARTS_DIR, f"{algo_name.lower()}_forecast_horizon_detail.png")
    
    # Crear gráfico personalizado para mejor visualización del forecast
    plt.figure(figsize=(14, 8))
    
    # Filtrar datos reales y forecast
    real_data = df_last_month_and_forecast[df_last_month_and_forecast['Periodo'] != 'Forecast']
    forecast_data = df_last_month_and_forecast[df_last_month_and_forecast['Periodo'] == 'Forecast']
    
    # Graficar datos reales
    if len(real_data) > 0:
        plt.plot(real_data['date'], real_data['Valor_Real'], 
                'o-', color='blue', linewidth=2, markersize=6, 
                label='Valores Reales', alpha=0.8)
        plt.plot(real_data['date'], real_data['Valor_Predicho'], 
                's-', color='green', linewidth=2, markersize=5, 
                label='Predicciones (Hold-out)', alpha=0.7)
    
    # Graficar forecast
    if len(forecast_data) > 0:
        plt.plot(forecast_data['date'], forecast_data['Valor_Predicho'], 
                '^-', color='red', linewidth=2.5, markersize=7, 
                label='Forecast', alpha=0.9)
        
        # Añadir línea vertical para separar datos reales de forecast
        if len(real_data) > 0:
            last_real_date = real_data['date'].iloc[-1]
            plt.axvline(x=last_real_date, color='orange', linestyle='--', 
                       linewidth=2, alpha=0.8, label='Inicio Forecast')
    
    # Configurar formato de fechas en el eje X
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45, ha='right')
    
    # Configurar el gráfico
    plt.title(f'Horizonte de Predicción Detallado - {algo_name}\n(Último mes + Forecast de {forecast_horizon} días)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Añadir anotaciones con valores en puntos clave del forecast
    if len(forecast_data) > 0:
        # Anotar primer y último punto del forecast
        first_forecast = forecast_data.iloc[0]
        last_forecast = forecast_data.iloc[-1]
        
        plt.annotate(f'{first_forecast["Valor_Predicho"]:.2f}', 
                    xy=(first_forecast['date'], first_forecast['Valor_Predicho']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontsize=9)
        
        plt.annotate(f'{last_forecast["Valor_Predicho"]:.2f}', 
                    xy=(last_forecast['date'], last_forecast['Valor_Predicho']),
                    xytext=(10, -15), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig(forecast_detail_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"[{algo_name}] Gráfico detallado de horizonte guardado: {forecast_detail_path}")
    
    # 3. Gráfico desde 2024-01-01 hasta el forecast
    # Filtrar datos desde 2024-01-01 en adelante
    fecha_inicio_2024 = pd.Timestamp('2024-01-01')
    
    # Filtrar todos los datos desde 2024-01-01
    df_desde_2024 = df_all[pd.to_datetime(df_all['date']) >= fecha_inicio_2024].copy()
    desde_2024_path = None  # Inicializar la variable
    
    if len(df_desde_2024) > 0:
        desde_2024_path = os.path.join(IMG_CHARTS_DIR, f"{algo_name.lower()}_desde_2024_hasta_forecast.png")
        
        # Crear gráfico personalizado desde 2024
        plt.figure(figsize=(16, 8))
        
        # Separar por períodos
        training_2024 = df_desde_2024[df_desde_2024['Periodo'] == 'Training']
        evaluacion_2024 = df_desde_2024[df_desde_2024['Periodo'] == 'Evaluacion']
        test_2024 = df_desde_2024[df_desde_2024['Periodo'] == 'Test']
        forecast_2024 = df_desde_2024[df_desde_2024['Periodo'] == 'Forecast']
        
        # Graficar valores reales (línea continua)
        all_real_data = pd.concat([training_2024, evaluacion_2024, test_2024])
        if len(all_real_data) > 0:
            plt.plot(all_real_data['date'], all_real_data['Valor_Real'], 
                    'o-', color='blue', linewidth=2, markersize=4, 
                    label='Valores Reales', alpha=0.8)
        
        # Graficar predicciones por período con colores diferentes
        if len(training_2024) > 0:
            plt.plot(training_2024['date'], training_2024['Valor_Predicho'], 
                    's-', color='lightgreen', linewidth=1.5, markersize=3, 
                    label='Predicciones (Training)', alpha=0.6)
        
        if len(evaluacion_2024) > 0:
            plt.plot(evaluacion_2024['date'], evaluacion_2024['Valor_Predicho'], 
                    's-', color='green', linewidth=2, markersize=4, 
                    label='Predicciones (Back-test)', alpha=0.7)
        
        if len(test_2024) > 0:
            plt.plot(test_2024['date'], test_2024['Valor_Predicho'], 
                    's-', color='darkgreen', linewidth=2, markersize=4, 
                    label='Predicciones (Hold-out)', alpha=0.8)
        
        # Graficar forecast
        if len(forecast_2024) > 0:
            plt.plot(forecast_2024['date'], forecast_2024['Valor_Predicho'], 
                    '^-', color='red', linewidth=2.5, markersize=6, 
                    label='Forecast', alpha=0.9)
            
            # Añadir línea vertical para separar datos reales de forecast
            if len(all_real_data) > 0:
                last_real_date = all_real_data['date'].max()
                plt.axvline(x=last_real_date, color='orange', linestyle='--', 
                           linewidth=2, alpha=0.8, label='Inicio Forecast')
            
            # Anotar primer y último punto del forecast
            first_forecast = forecast_2024.iloc[0]
            last_forecast = forecast_2024.iloc[-1]
            
            plt.annotate(f'{first_forecast["Valor_Predicho"]:.2f}', 
                        xy=(first_forecast['date'], first_forecast['Valor_Predicho']),
                        xytext=(10, 15), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=9, ha='center')
            
            plt.annotate(f'{last_forecast["Valor_Predicho"]:.2f}', 
                        xy=(last_forecast['date'], last_forecast['Valor_Predicho']),
                        xytext=(10, -20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=9, ha='center')
        
        # Configurar formato de fechas en el eje X
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=1))
        plt.xticks(rotation=45, ha='right')
        
        # Configurar el gráfico
        total_dias = len(df_desde_2024)
        plt.title(f'Evolución desde 2024 hasta Forecast - {algo_name}\n(Desde 2024-01-01 + Forecast de {forecast_horizon} días - Total: {total_dias} registros)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Fecha', fontsize=12)
        plt.ylabel('Valor', fontsize=12)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(desde_2024_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"[{algo_name}] Gráfico desde 2024 guardado: {desde_2024_path}")
    else:
        logging.warning(f"[{algo_name}] No hay datos desde 2024-01-01 para graficar")
    
    # 4. Gráfico completo (zones_full) - ASEGURAR QUE SE GENERE PARA TODOS LOS MODELOS INCLUYENDO LSTM
    full_chart_path = os.path.join(IMG_CHARTS_DIR, f"{algo_name.lower()}_three_zones_full.png")
    plot_real_vs_pred(
        df_all,  # Usar df_all que incluye todas las zonas
        title=f"Histórico Completo y Forecast - {algo_name} (Todas las Zonas)",
        model_name=algo_name,
        output_path=full_chart_path
    )
    logging.info(f"[{algo_name}] Gráfico completo guardado (todas las zonas): {full_chart_path}")
    
    # 5. Feature importance si está disponible (solo para modelos tree-based)
    if hasattr(final_model, 'feature_importances_'):
        importance_chart_path = os.path.join(IMG_CHARTS_DIR, f"{algo_name.lower()}_three_zones_importance.png")
        
        try:
            # Obtener nombres de features y valores de importancia
            feature_names = X_all.columns
            importances = final_model.feature_importances_
            
            plot_feature_importance(
                importances,
                feature_names,
                title=f"Feature Importance - {algo_name}",
                top_n=min(20, len(feature_names)),
                model_name=algo_name,
                output_path=importance_chart_path
            )
            logging.info(f"[{algo_name}] Gráfico de importancia guardado: {importance_chart_path}")
        except Exception as e:
            logging.error(f"[{algo_name}] Error al generar gráfico de importancia: {e}")
    
    # 6. Resumen de gráficos generados
    logging.info(f"[{algo_name}] ✅ Gráficos generados:")
    logging.info(f"[{algo_name}]   - Zona B, C y Forecast: {chart_path}")
    logging.info(f"[{algo_name}]   - Horizonte detallado: {forecast_detail_path}")
    if desde_2024_path:
        logging.info(f"[{algo_name}]   - Desde 2024 hasta forecast: {desde_2024_path}")
    logging.info(f"[{algo_name}]   - Histórico completo: {full_chart_path}")
    if hasattr(final_model, 'feature_importances_'):
        logging.info(f"[{algo_name}]   - Feature importance: {importance_chart_path}")
    
    # Limpiar sesión de Keras para LSTM
    if algo_name == "LSTM":
        tf.keras.backend.clear_session()
    
    # Tiempo total de procesamiento
    total_time = time.perf_counter() - start_time
    logging.info(f"[{algo_name}] Procesamiento completo en {total_time:.2f}s")
    
    return df_all
# -----------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# -----------------------------------------------------------------

def run_training():
    """
    Ejecuta el proceso completo con el enfoque de tres zonas:
      - Zona A (70%): Ajuste de hiperparámetros (RandomSearch → Optuna)
      - Zona B (20%): Back-test externo
      - Zona C (10%): Hold-out final
    """
    # Tiempo total de ejecución
    total_start_time = time.perf_counter()
    
    # Información sobre GPU
    if has_gpu:
        # Obtener información detallada de GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        gpu_names = [gpu.name.split('/')[-1] for gpu in gpus]
        memory_limit = GPU_MEMORY_LIMIT
        mem_info = f" con límite de {memory_limit}MB" if memory_limit > 0 else ""
        
        logging.info(f"✅ GPU disponible para entrenamiento{mem_info}: {', '.join(gpu_names)}")
        logging.info("   Los modelos compatibles (LSTM, CatBoost, XGBoost) usarán aceleración GPU si está disponible")
    else:
        logging.info("ℹ️ No se detectó GPU o está deshabilitada. El entrenamiento se realizará en CPU.")
    
    # Asegurar que existen los directorios
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(IMG_CHARTS_DIR, exist_ok=True)
    
    # Determinar horizonte según forecast_period
    if args.forecast_period == "1MONTH":
        forecast_horizon = FORECAST_HORIZON_1MONTH
    elif args.forecast_period == "3MONTHS":
        forecast_horizon = FORECAST_HORIZON_3MONTHS
    else:
        forecast_horizon = FORECAST_HORIZON_1MONTH  # Default
    
    # Actualizar gap con el horizonte correcto
    args.gap = forecast_horizon
    logging.info(f"Usando horizonte de {forecast_horizon} días para forecast_period={args.forecast_period}")
    
    # Lectura y ordenamiento
    data_load_start = time.perf_counter()
    df = pd.read_excel(args.input_file)
    df.sort_values(by="date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logging.info("Datos leídos y ordenados por fecha.")
    
    # Limpieza de datos
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    logging.info("Se han imputado los valores NaN e inf (ffill y relleno con 0).")
    
    # Definición de target y features
    target_col = df.columns[-1]

    # MODIFICADO: Usar directamente el target original, sin aplicar lag adicional
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()
    
    logging.info(f"Target: '{target_col}'")
    logging.info(f"Dimensiones: X: {X.shape}, y: {y.shape}")
    
    # Manejo de la columna 'date'
    if "date" in X.columns:
        dates = X["date"]
        X.drop(columns=["date"], inplace=True)
        X.index = pd.to_datetime(dates)
    else:
        X.index = pd.to_datetime(X.index)
    
    # Eliminación de columnas de varianza cero
    zero_var_cols = [c for c in X.columns if X[c].std() == 0]
    if zero_var_cols:
        logging.warning(f"Eliminando columnas de varianza 0: {zero_var_cols}")
        X.drop(columns=zero_var_cols, inplace=True)
    
    # Escalado (aplicable para MLP y SVM)
    X_ = X.copy()
    y_ = y.copy()
    if any(model in SCALING_REQUIRED_MODELS and SCALING_REQUIRED_MODELS[model] for model in ["MLP", "SVM", "LSTM"]):
        scaler = StandardScaler()
        X_ = pd.DataFrame(scaler.fit_transform(X_), columns=X_.columns, index=X_.index)
    X_.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_.ffill(inplace=True)
    X_.fillna(0, inplace=True)
    y_.ffill(inplace=True)
    y_.fillna(0, inplace=True)
    if np.isinf(X_.values).any() or np.isnan(X_.values).any():
        logging.warning("Aún hay inf/NaN en X_ tras limpieza. Se detiene el proceso.")
        return
    if np.isinf(y_.values).any() or np.isnan(y_.values).any():
        logging.warning("Aún hay inf/NaN en y_ tras limpieza. Se detiene el proceso.")
        return
    
    data_load_time = time.perf_counter() - data_load_start
    logging.info(f"Tiempo de carga y preprocesamiento: {data_load_time:.2f}s")
    
    # División en tres zonas (70-20-10)
    split_start_time = time.perf_counter()
    logging.info("Realizando división en tres zonas (70-20-10)")

    # Calcular cortes para las tres zonas
    total_rows = len(X_)
    cut_70 = int(total_rows * 0.7)
    cut_90 = int(total_rows * 0.9)

    # Zona A (70%): Para ajuste de hiperparámetros
    X_zone_A = X_.iloc[:cut_70].copy()
    y_zone_A = y_.iloc[:cut_70].copy()

    # Zona B (20%): Para back-test externo
    X_zone_B = X_.iloc[cut_70:cut_90].copy()
    y_zone_B = y_.iloc[cut_70:cut_90].copy()

    # Zona C (10%): Para hold-out final
    X_zone_C = X_.iloc[cut_90:].copy()
    y_zone_C = y_.iloc[cut_90:].copy()

    logging.info(f"División en tres zonas: A (HPO)={len(X_zone_A)}, B (Back-test)={len(X_zone_B)}, C (Hold-out)={len(X_zone_C)}")

    # Para mantener compatibilidad con el resto del código
    # Puedes asignar variables como X_train, X_eval, X_test (si las necesitas)
    X_train = X_zone_A.copy()
    y_train = y_zone_A.copy()
    X_eval = X_zone_B.copy()  # Backtest
    y_eval = y_zone_B.copy()
    X_test = X_zone_C.copy()  # Holdout
    y_test = y_zone_C.copy()

    split_time = time.perf_counter() - split_start_time
    logging.info(f"Tiempo para split de datos: {split_time:.2f}s")
    
    # Definición de algoritmos a entrenar
    algorithms = [
        ("CatBoost", objective_catboost),
        ("LightGBM", objective_lgbm),
        ("XGBoost", objective_xgboost),
        ("MLP", objective_mlp),
        ("SVM", objective_svm),
        ("LSTM", objective_lstm)  # Añadido LSTM
    ]
    
    # Verificar si se han seleccionado algoritmos específicos mediante la variable SELECTED_ALGOS
    # Si SELECTED_ALGOS está definida y no está vacía, filtrar los algoritmos
    selected_algos = getattr(args, 'selected_algos', [])
    if selected_algos:
        algorithms = [(name, func) for name, func in algorithms if name in selected_algos]
        logging.info(f"Ejecutando SOLO los algoritmos seleccionados: {[a[0] for a in algorithms]}")
    
    # Registra el tiempo de inicio para cada algoritmo
    algorithm_times = {}
    final_results = []
    
    for algo_name, obj_func in algorithms:
        algo_start_time = time.perf_counter()
        logging.info(f"=== Optimizando y entrenando {algo_name}... ===")
        
        # Ejecutar el proceso completo para este algoritmo
        result_df = optimize_and_train_extended(
            algo_name, obj_func, 
            X_zone_A, y_zone_A,  # Zona A (70%) para HPO
            X_zone_B, y_zone_B,  # Zona B (20%) para back-test
            X_zone_C, y_zone_C,  # Zona C (10%) para hold-out final
            forecast_horizon=forecast_horizon, 
            top_n=3
        )
        
        final_results.append(result_df)
        algo_end_time = time.perf_counter()
        algorithm_times[algo_name] = algo_end_time - algo_start_time
        logging.info(f"=== Completado {algo_name} en {algorithm_times[algo_name]:.2f}s ===")
    
    # Guardar tiempos en un archivo JSON para referencia
    timings_file = os.path.join(RESULTS_DIR, "training_times.json")
    with open(timings_file, 'w') as f:
        json.dump(algorithm_times, f, indent=4)
    
    # Visualizar tiempos de entrenamiento
    plt.figure(figsize=(10, 6))
    plt.bar(algorithm_times.keys(), algorithm_times.values())
    plt.title("Tiempo de entrenamiento por algoritmo")
    plt.xlabel("Algoritmo")
    plt.ylabel("Tiempo (segundos)")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(IMG_CHARTS_DIR, "training_times.png"), dpi=300)
    plt.close()
    
    if final_results:
        predictions_df = pd.concat(final_results, ignore_index=True)
        # Guardar CSV con el histórico completo y el forecast
        columns_to_save = ["date", "Valor_Real", "Valor_Predicho", "Modelo", "Version", 
                           "Periodo", "RMSE", "MAE", "R2", "SMAPE", "Hyperparámetros", "Tipo_Mercado"]
        predictions_df.to_csv(args.output_predictions, index=False, float_format="%.6f")
        logging.info(f"Archivo final de predicciones guardado en {args.output_predictions}")
        
        # Gráfico comparativo de todos los modelos
        all_models_chart = os.path.join(IMG_CHARTS_DIR, "all_models_comparison.png")
        plot_real_vs_pred(
            predictions_df[predictions_df['Periodo'] != 'Forecast'],
            title="Comparación de todos los modelos",
            output_path=all_models_chart
        )
        logging.info(f"Gráfico comparativo de todos los modelos guardado: {all_models_chart}")
        
        # Gráfico comparativo de RMSE por modelo
        rmse_df = predictions_df[predictions_df['Periodo'] == 'Test'].groupby('Modelo')['RMSE'].mean().reset_index()
        rmse_df = rmse_df.sort_values('RMSE')
        
        plt.figure(figsize=(10, 6))
        plt.bar(rmse_df['Modelo'], rmse_df['RMSE'])
        plt.title("RMSE por modelo en conjunto de Test")
        plt.xlabel("Modelo")
        plt.ylabel("RMSE")
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_CHARTS_DIR, "rmse_comparison.png"), dpi=300)
        plt.close()
        
        # Generar informe resumen en formato CSV
        summary_metrics = []
        for modelo in predictions_df['Modelo'].unique():
            for periodo in ['Training', 'Evaluacion', 'Test']:
                df_filtered = predictions_df[(predictions_df['Modelo'] == modelo) & 
                                            (predictions_df['Periodo'] == periodo)]
                if not df_filtered.empty:
                    metrics = {
                        'Modelo': modelo,
                        'Periodo': periodo,
                        'RMSE_medio': df_filtered['RMSE'].mean(),
                        'MAE_medio': df_filtered['MAE'].mean(),
                        'R2_medio': df_filtered['R2'].mean(),
                        'SMAPE_medio': df_filtered['SMAPE'].mean(),
                        'Tiempo_Entrenamiento': algorithm_times.get(modelo, np.nan)
                    }
                    summary_metrics.append(metrics)
        
        summary_df = pd.DataFrame(summary_metrics)
        summary_file = os.path.join(RESULTS_DIR, "resumen_metricas.csv")
        summary_df.to_csv(summary_file, index=False)
        logging.info(f"Resumen de métricas guardado en: {summary_file}")
    else:
        logging.warning("No se han generado resultados finales debido a problemas en el proceso.")
    
    # Guardar todos los resultados de back-test y hold-out
    all_backtest_results = {}
    all_holdout_results = {}
    
    for algo_name, result_df in zip([algo[0] for algo in algorithms], final_results):
        # Extraer resultados de back-test y hold-out desde los DataFrames generados
        backtest_metrics = result_df[result_df['Periodo'] == 'Evaluacion'].iloc[0][['RMSE', 'MAE', 'R2', 'SMAPE']].to_dict()
        holdout_metrics = result_df[result_df['Periodo'] == 'Test'].iloc[0][['RMSE', 'MAE', 'R2', 'SMAPE']].to_dict()
        
        all_backtest_results[algo_name] = {'metrics': backtest_metrics}
        all_holdout_results[algo_name] = {'metrics': holdout_metrics}
    
    # Generar tabla comparativa
    metrics_comparison = pd.DataFrame(columns=['Algoritmo', 'Fase', 'RMSE', 'MAE', 'R2', 'SMAPE'])
    
    # Añadir métricas de back-test
    for algo_name, results in all_backtest_results.items():
        metrics = results['metrics']
        new_row = pd.DataFrame({
            'Algoritmo': [algo_name],
            'Fase': ['Back-test (Zona B)'],
            'RMSE': [metrics['RMSE']],
            'MAE': [metrics['MAE']],
            'R2': [metrics['R2']],
            'SMAPE': [metrics['SMAPE']]
        })
        metrics_comparison = pd.concat([metrics_comparison, new_row], ignore_index=True)
    
    # Añadir métricas de hold-out
    for algo_name, results in all_holdout_results.items():
        metrics = results['metrics']
        new_row = pd.DataFrame({
            'Algoritmo': [algo_name],
            'Fase': ['Hold-out (Zona C)'],
            'RMSE': [metrics['RMSE']],
            'MAE': [metrics['MAE']],
            'R2': [metrics['R2']],
            'SMAPE': [metrics['SMAPE']]
        })
        metrics_comparison = pd.concat([metrics_comparison, new_row], ignore_index=True)
    
    # Guardar tabla comparativa adicional
    metrics_file = os.path.join(RESULTS_DIR, "metrics_three_zones.csv")
    metrics_comparison.to_csv(metrics_file, index=False)
    logging.info(f"Tabla comparativa de zonas guardada en: {metrics_file}")
    
    # Gráfico comparativo de RMSE por algoritmo y fase
    plt.figure(figsize=(12, 6))
    
    # Filtrar para back-test y hold-out por separado
    backtest_df = metrics_comparison[metrics_comparison['Fase'] == 'Back-test (Zona B)']
    holdout_df = metrics_comparison[metrics_comparison['Fase'] == 'Hold-out (Zona C)']
    
    # Ordenar por RMSE
    backtest_df = backtest_df.sort_values('RMSE')
    holdout_df = holdout_df.sort_values('RMSE')
    
    # Graficar
    bar_width = 0.35
    index = np.arange(len(backtest_df))
    
    plt.bar(index, backtest_df['RMSE'], bar_width, label='Back-test (Zona B)')
    plt.bar(index + bar_width, holdout_df['RMSE'], bar_width, label='Hold-out (Zona C)')
    
    plt.xlabel('Algoritmo')
    plt.ylabel('RMSE')
    plt.title('Comparación de RMSE por Algoritmo y Fase')
    plt.xticks(index + bar_width/2, backtest_df['Algoritmo'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_CHARTS_DIR, "rmse_zones_comparison.png"), dpi=300)
    plt.close()
    
    # Tiempo total de ejecución
    total_time = time.perf_counter() - total_start_time
    logging.info(f"Proceso completo terminado en {total_time:.2f}s")
    
    # Imprimir resumen final con ✅ para mejor visualización
    gpu_info = "con GPU ⚡" if has_gpu else "en CPU 💻"
    print(f"\n✅ Entrenamiento completado con enfoque de tres zonas {gpu_info}")
    print(f"✅ Número de algoritmos procesados: {len(algorithms)}")
    print(f"✅ Visualizaciones generadas en: {IMG_CHARTS_DIR}")
    print(f"✅ Modelos guardados en: {args.output_dir}")
    print(f"✅ Predicciones consolidadas en: {args.output_predictions}")
    print(f"✅ Tabla comparativa de métricas: {metrics_file}")
    print(f"✅ Tiempo total de ejecución: {total_time:.2f}s")

    # Si se utilizó GPU, mostrar información adicional
    if has_gpu:
        gpu_models = ["LSTM"]
        if any(algo[0] in ["CatBoost", "XGBoost", "LightGBM"] for algo in algorithms):
            gpu_models.extend(["CatBoost", "XGBoost", "LightGBM"])
        
        print(f"🚀 Aceleración GPU utilizada para: {', '.join(gpu_models)}")
        
    return True

if __name__ == "__main__":
    run_training()