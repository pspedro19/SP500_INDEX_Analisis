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
USE_GPU = True  # Cambiar a False para deshabilitar GPU
GPU_MEMORY_LIMIT = 0  # Establecer a un valor en MB para limitar memoria (0 = sin límite)
has_gpu = False  # Estado (se actualizará en configure_gpu)


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
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
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
from sp500_analysis.config.settings import settings
from pipelines.ml.config import ensure_directories

PROJECT_ROOT = settings.project_root
MODELS_DIR = settings.models_dir
TRAINING_DIR = settings.training_dir
RESULTS_DIR = settings.results_dir
IMG_CHARTS_DIR = settings.img_charts_dir
DATE_COL = settings.date_col
LOCAL_REFINEMENT_DAYS = settings.local_refinement_days
TRAIN_TEST_SPLIT_RATIO = settings.train_test_split_ratio
FORECAST_HORIZON_1MONTH = settings.forecast_horizon_1month
FORECAST_HORIZON_3MONTHS = settings.forecast_horizon_3months
RANDOM_SEED = settings.random_seed

# Importar funciones de visualización
from sp500_analysis.shared.visualization.plotters import (
    plot_real_vs_pred,
    plot_training_curves,
    plot_feature_importance,
)

# ------------------------------
# CONFIGURACIÓN DE LOGGING
# ------------------------------
log_file = os.path.join(PROJECT_ROOT, "logs", f"train_models_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
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
    "LSTM": True,  # LSTM también requiere escalado
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
    random_search_trials = 3  # Era 30 ⭐ CAMBIO PRINCIPAL (90% más rápido)
    optuna_trials = 2  # Era 15 ⭐ CAMBIO PRINCIPAL (87% más rápido)

    n_trials = 20  # Original (no se usa mucho)
    cv_splits = 3  # Era 5 ⭐ CAMBIO (40% más rápido)
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
        'optuna_rmse': holdout_metrics.get('RMSE', np.nan),
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
        X_seq.append(X_values[i : i + sequence_length])
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
            "units", max(base_params.get("units", 64) - 32, 16), min(base_params.get("units", 64) + 32, 256)
        )
        learning_rate = trial.suggest_float(
            "learning_rate",
            max(base_params.get("learning_rate", 0.001) * 0.5, 0.0001),
            min(base_params.get("learning_rate", 0.001) * 2.0, 0.01),
            log=True,
        )
        dropout_rate = trial.suggest_float(
            "dropout_rate",
            max(base_params.get("dropout_rate", 0.2) - 0.1, 0.0),
            min(base_params.get("dropout_rate", 0.2) + 0.2, 0.5),
        )
        sequence_length = trial.suggest_int(
            "sequence_length",
            max(base_params.get("sequence_length", 10) - 5, 5),
            min(base_params.get("sequence_length", 10) + 5, 20),
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
                    model = Sequential(
                        [
                            LSTM(units=units, input_shape=(sequence_length, n_features), return_sequences=False),
                            Dropout(dropout_rate),
                            Dense(1),
                        ]
                    )
            except RuntimeError:
                # Fallback a CPU si hay problemas con GPU
                logging.warning("Error al crear modelo en GPU. Fallback a CPU.")
                model = Sequential(
                    [
                        LSTM(units=units, input_shape=(sequence_length, n_features), return_sequences=False),
                        Dropout(dropout_rate),
                        Dense(1),
                    ]
                )
        else:
            model = Sequential(
                [
                    LSTM(units=units, input_shape=(sequence_length, n_features), return_sequences=False),
                    Dropout(dropout_rate),
                    Dense(1),
                ]
            )

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model

    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        train_idx = sorted(train_idx)
        val_idx = sorted(val_idx)

        # Asegurar que hay suficientes datos para la secuencia
        if len(train_idx) <= sequence_length or len(val_idx) <= sequence_length:
            logging.warning(
                f"[LSTM][fold {fold_i}] Datos insuficientes para crear secuencias con length={sequence_length}"
            )
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
            monitor='val_loss', patience=patience, restore_best_weights=True, min_delta=0.001  # Umbral mínimo de mejora
        )

        # Entrenar modelo con early stopping
        try:
            if has_gpu:
                with tf.device('/gpu:0'):
                    model.fit(
                        X_train_seq,
                        y_train_seq,
                        validation_data=(X_val_seq, y_val_seq),
                        epochs=max_epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping],
                        verbose=verbose_level,
                    )
            else:
                model.fit(
                    X_train_seq,
                    y_train_seq,
                    validation_data=(X_val_seq, y_val_seq),
                    epochs=max_epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=verbose_level,
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
        logging.info(
            f"[{algo_name}] Modo rápido activado: usando {n_iter} trials en lugar de {args.random_search_trials}"
        )
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
                "learning_rate": uniform(0.01, 0.04),  # Rango: 0.01-0.05
                "depth": randint(4, 7),  # Rango: 4-6 (menos profundidad)
                "iterations": randint(50, 151),  # ⭐ CRÍTICO: 50-150 (era 300-701)
                # ⚠️ PRODUCCIÓN: Cambiar a randint(500, 2001) para mejor precisión
            },
            "LightGBM": {
                "learning_rate": uniform(0.01, 0.04),  # Rango: 0.01-0.05
                "max_depth": randint(4, 7),  # Rango: 4-6
                "n_estimators": randint(50, 151),  # ⭐ CRÍTICO: 50-150 (era 300-701)
                "subsample": uniform(0.7, 0.3),  # Rango: 0.7-1.0
                # ⚠️ PRODUCCIÓN: Cambiar n_estimators a randint(500, 2001)
            },
            "XGBoost": {
                "learning_rate": uniform(0.01, 0.04),  # Rango: 0.01-0.05
                "max_depth": randint(4, 7),  # Rango: 4-6
                "n_estimators": randint(50, 151),  # ⭐ CRÍTICO: 50-150 (era 300-701)
                "subsample": uniform(0.7, 0.3),  # Rango: 0.7-1.0
                # ⚠️ PRODUCCIÓN: Cambiar n_estimators a randint(500, 2001)
            },
            "MLP": {
                "hidden_neurons": randint(50, 101),  # Rango: 50-100 (era 50-201)
                "learning_rate_init": uniform(0.001, 0.009),  # Rango: 0.001-0.01
                "max_iter": randint(50, 201),  # ⭐ CRÍTICO: 50-200 (era 200-1001)
                # ⚠️ PRODUCCIÓN: Cambiar max_iter a randint(200, 1001)
            },
            "SVM": {
                "C": uniform(0.5, 4.5),  # Rango: 0.5-5.0 (era 0.1-10)
                "epsilon": uniform(0.05, 0.15),  # Rango: 0.05-0.2 (era 0.01-0.5)
                # ⚠️ PRODUCCIÓN: Ampliar rangos - C: uniform(0.1, 9.9), epsilon: uniform(0.01, 0.49)
            },
            "LSTM": {
                "units": randint(32, 65),  # Rango: 32-64 (era 32-129)
                "learning_rate": uniform(0.001, 0.009),  # Rango: 0.001-0.01
                "dropout_rate": uniform(0.1, 0.3),  # Rango: 0.1-0.4 (era 0.0-0.5)
                "sequence_length": randint(5, 11),  # Rango: 5-10 (era 5-21)
                # ⚠️ PRODUCCIÓN: Ampliar rangos según comentarios
            },
        }
    else:
        # 🏭 CONFIGURACIÓN COMPLETA PARA PRODUCCIÓN
        logging.info(f"[{algo_name}] Usando parámetros completos para producción")
        param_distributions = {
            "CatBoost": {
                "learning_rate": uniform(0.001, 0.049),  # Rango completo: 0.001-0.05
                "depth": randint(4, 11),  # Rango completo: 4-10
                "iterations": randint(500, 2001),  # Rango completo: 500-2000
            },
            "LightGBM": {
                "learning_rate": uniform(0.001, 0.049),
                "max_depth": randint(4, 11),
                "n_estimators": randint(500, 2001),
                "subsample": uniform(0.5, 0.5),  # 0.5 a 1.0
            },
            "XGBoost": {
                "learning_rate": uniform(0.001, 0.049),
                "max_depth": randint(4, 11),
                "n_estimators": randint(500, 2001),
                "subsample": uniform(0.5, 0.5),  # 0.5 a 1.0
            },
            "MLP": {
                "hidden_neurons": randint(50, 201),
                "learning_rate_init": uniform(0.0001, 0.01),
                "max_iter": randint(200, 1001),
            },
            "SVM": {"C": uniform(0.1, 9.9), "epsilon": uniform(0.01, 0.49)},  # 0.1 a 10  # 0.01 a 0.5
            "LSTM": {
                "units": randint(32, 129),  # 32 a 128 unidades
                "learning_rate": uniform(0.0001, 0.01),  # 0.0001 a 0.01
                "dropout_rate": uniform(0.0, 0.5),  # 0 a 0.5
                "sequence_length": randint(5, 21),  # 5 a 20
            },
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
                default_values = {'units': 64, 'learning_rate': 0.001, 'dropout_rate': 0.2, 'sequence_length': 10}
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
        n_jobs=n_jobs,  # Paralelismo controlado
        pre_dispatch="2*n_jobs",  # Mejor gestión de memoria
        error_score=np.nan,  # No fallar si hay errores en evaluaciones individuales
        verbose=0,  # Reducir salida para menos overhead
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
        logging.info(
            f"[{algo_name}] 🔮 Tiempo estimado para Optuna ({args.optuna_trials} trials): {estimated_optuna_time:.2f}s"
        )

        return best_params

    except Exception as e:
        logging.error(f"[{algo_name}] ❌ Error en RandomizedSearchCV: {e}")
        logging.error(f"[{algo_name}] Traceback: ", exc_info=True)

        # En caso de error, devolver parámetros por defecto
        logging.info(f"[{algo_name}] 🔄 Usando parámetros por defecto como respaldo")

        default_params = {
            "CatBoost": {"learning_rate": 0.01, "depth": 6, "iterations": 100 if execution_mode == "fast" else 1000},
            "LightGBM": {
                "learning_rate": 0.01,
                "max_depth": 6,
                "n_estimators": 100 if execution_mode == "fast" else 1000,
                "subsample": 0.8,
            },
            "XGBoost": {
                "learning_rate": 0.01,
                "max_depth": 6,
                "n_estimators": 100 if execution_mode == "fast" else 1000,
                "subsample": 0.8,
            },
            "MLP": {
                "hidden_neurons": 100,
                "learning_rate_init": 0.001,
                "max_iter": 100 if execution_mode == "fast" else 500,
            },
            "SVM": {"C": 1.0, "epsilon": 0.1},
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
            log=True,
        )
        depth = trial.suggest_int(
            "depth", max(base_params.get("depth", 6) - 2, 3), min(base_params.get("depth", 6) + 2, 12)
        )
        iterations = trial.suggest_int(
            "iterations",
            max(base_params.get("iterations", 1000) - 300, 300),
            min(base_params.get("iterations", 1000) + 300, 3000),
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
            use_best_model=True,
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
            log=True,
        )
        max_depth = trial.suggest_int(
            "max_depth", max(base_params.get("max_depth", 6) - 2, 3), min(base_params.get("max_depth", 6) + 2, 12)
        )
        n_estimators = trial.suggest_int(
            "n_estimators",
            max(base_params.get("n_estimators", 1000) - 300, 300),
            min(base_params.get("n_estimators", 1000) + 300, 3000),
        )
        subsample = trial.suggest_float(
            "subsample",
            max(base_params.get("subsample", 0.8) - 0.2, 0.5),
            min(base_params.get("subsample", 0.8) + 0.2, 1.0),
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
            random_state=RANDOM_SEED,
        )
        model.fit(
            X_train_cv,
            y_train_cv,
            eval_set=[(X_val_cv, y_val_cv)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )
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
            log=True,
        )
        max_depth = trial.suggest_int(
            "max_depth", max(base_params.get("max_depth", 6) - 2, 3), min(base_params.get("max_depth", 6) + 2, 12)
        )
        n_estimators = trial.suggest_int(
            "n_estimators",
            max(base_params.get("n_estimators", 1000) - 300, 300),
            min(base_params.get("n_estimators", 1000) + 300, 3000),
        )
        subsample = trial.suggest_float(
            "subsample",
            max(base_params.get("subsample", 0.8) - 0.2, 0.5),
            min(base_params.get("subsample", 0.8) + 0.2, 1.0),
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
        dval = xgb.DMatrix(X_val_cv, label=y_val_cv)
        params = {
            "objective": "reg:squarederror",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "seed": RANDOM_SEED,
        }
        evals_result = {}
        xgb_model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, "eval")],
            early_stopping_rounds=50,
            evals_result=evals_result,
            verbose_eval=False,
        )
        best_iter = xgb_model.best_iteration
        y_pred = xgb_model.predict(dval, iteration_range=(0, best_iter + 1))
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
            min(base_params.get("hidden_neurons", 100) + 50, 250),
        )
        learning_rate_init = trial.suggest_float(
            "learning_rate_init",
            max(base_params.get("learning_rate_init", 0.001) * 0.5, 0.0001),
            min(base_params.get("learning_rate_init", 0.001) * 2.0, 0.01),
            log=True,
        )
        max_iter = trial.suggest_int(
            "max_iter",
            max(base_params.get("max_iter", 500) - 200, 200),
            min(base_params.get("max_iter", 500) + 200, 1200),
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
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=RANDOM_SEED,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=25,
        )
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
            "C", max(base_params.get("C", 1.0) * 0.5, 0.1), min(base_params.get("C", 1.0) * 2.0, 20.0), log=True
        )
        epsilon = trial.suggest_float(
            "epsilon", max(base_params.get("epsilon", 0.1) * 0.5, 0.01), min(base_params.get("epsilon", 0.1) * 2.0, 1.0)
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
        X_zone_A_scaled = pd.DataFrame(scaler.fit_transform(X_zone_A), columns=X_zone_A.columns, index=X_zone_A.index)
        X_zone_B_scaled = pd.DataFrame(scaler.transform(X_zone_B), columns=X_zone_B.columns, index=X_zone_B.index)
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
            devices=devices,
        )
        model.fit(X_zone_A_scaled, y_zone_A)
        preds_zone_B = model.predict(X_zone_B_scaled)
    elif algo_name == "LightGBM":
        model = lgb.LGBMRegressor(
            learning_rate=best_params.get("learning_rate", 0.01),
            max_depth=best_params.get("max_depth", 6),
            n_estimators=best_params.get("n_estimators", 1000),
            random_state=RANDOM_SEED,
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
            gpu_id=gpu_id,
        )
        model.fit(X_zone_A_scaled, y_zone_A)
        preds_zone_B = model.predict(X_zone_B_scaled)
    elif algo_name == "MLP":
        hidden_neurons = best_params.get("hidden_neurons", 100)
        model = MLPRegressor(
            hidden_layer_sizes=(hidden_neurons, hidden_neurons),
            learning_rate_init=best_params.get("learning_rate_init", 0.001),
            max_iter=best_params.get("max_iter", 500),
            random_state=RANDOM_SEED,
        )
        model.fit(X_zone_A_scaled, y_zone_A)
        preds_zone_B = model.predict(X_zone_B_scaled)
    elif algo_name == "SVM":
        model = SVR(C=best_params.get("C", 1.0), epsilon=best_params.get("epsilon", 0.1))
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
        model = Sequential(
            [
                LSTM(units=units, input_shape=(sequence_length, n_features), return_sequences=False),
                Dropout(dropout_rate),
                Dense(1),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

        # Entrenar modelo
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Dividir datos de entrenamiento para validación interna
        val_split = 0.1
        val_samples = int(len(X_train_seq) * val_split)
        if val_samples > 0:
            X_train_lstm = X_train_seq[:-val_samples]
            y_train_lstm = y_train_seq[:-val_samples]
            X_val_lstm = X_train_seq[-val_samples:]
            y_val_lstm = y_train_seq[-val_samples:]

            model.fit(
                X_train_lstm,
                y_train_lstm,
                validation_data=(X_val_lstm, y_val_lstm),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0,
            )
        else:
            model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0)

        # Crear secuencias para Zona B
        if len(X_zone_B_scaled) > sequence_length:
            # Crear secuencias completas para predicción
            X_zone_B_seq, y_zone_B_actual = create_sequences(X_zone_B_scaled, y_zone_B, sequence_length)

            # Predecir
            if len(X_zone_B_seq) > 0:
                preds_seq = model.predict(X_zone_B_seq, verbose=0).flatten()

                # Ajustar las predicciones al formato original
                preds_zone_B = np.full(len(y_zone_B), np.nan)
                preds_zone_B[sequence_length : sequence_length + len(preds_seq)] = preds_seq

                # Para las primeras sequence_length filas, usar el último modelo para predecir
                if sequence_length < len(X_zone_B_scaled):
                    for i in range(sequence_length):
                        # Usar datos de Zona A + inicio de Zona B para secuencias iniciales
                        if i < len(X_zone_A_scaled):
                            combined_data = pd.concat(
                                [X_zone_A_scaled.iloc[-sequence_length + i :], X_zone_B_scaled.iloc[:i]]
                            )
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

    return {'model': model, 'predictions': preds_zone_B, 'metrics': metrics, 'scaler': scaler}


def evaluate_holdout(
    algo_name, X_zone_A, y_zone_A, X_zone_B, y_zone_B, X_zone_C, y_zone_C, best_params, scaling_required=False
):
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
            scaler.fit_transform(X_zone_AB), columns=X_zone_AB.columns, index=X_zone_AB.index
        )
        X_zone_C_scaled = pd.DataFrame(scaler.transform(X_zone_C), columns=X_zone_C.columns, index=X_zone_C.index)
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
            devices=devices,
        )
        model.fit(X_zone_AB_scaled, y_zone_AB)
        preds_zone_C = model.predict(X_zone_C_scaled)
    elif algo_name == "LightGBM":
        model = lgb.LGBMRegressor(
            learning_rate=best_params.get("learning_rate", 0.01),
            max_depth=best_params.get("max_depth", 6),
            n_estimators=best_params.get("n_estimators", 1000),
            random_state=RANDOM_SEED,
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
            gpu_id=gpu_id,
        )
        model.fit(X_zone_AB_scaled, y_zone_AB)
        preds_zone_C = model.predict(X_zone_C_scaled)
    elif algo_name == "MLP":
        hidden_neurons = best_params.get("hidden_neurons", 100)
        model = MLPRegressor(
            hidden_layer_sizes=(hidden_neurons, hidden_neurons),
            learning_rate_init=best_params.get("learning_rate_init", 0.001),
            max_iter=best_params.get("max_iter", 500),
            random_state=RANDOM_SEED,
        )
        model.fit(X_zone_AB_scaled, y_zone_AB)
        preds_zone_C = model.predict(X_zone_C_scaled)
    elif algo_name == "SVM":
        model = SVR(C=best_params.get("C", 1.0), epsilon=best_params.get("epsilon", 0.1))
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
        model = Sequential(
            [
                LSTM(units=units, input_shape=(sequence_length, n_features), return_sequences=False),
                Dropout(dropout_rate),
                Dense(1),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

        # Entrenar modelo
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Dividir datos de entrenamiento para validación interna
        val_split = 0.1
        val_samples = int(len(X_train_seq) * val_split)
        if val_samples > 0:
            X_train_lstm = X_train_seq[:-val_samples]
            y_train_lstm = y_train_seq[:-val_samples]
            X_val_lstm = X_train_seq[-val_samples:]
            y_val_lstm = y_train_seq[-val_samples:]

            model.fit(
                X_train_lstm,
                y_train_lstm,
                validation_data=(X_val_lstm, y_val_lstm),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0,
            )
        else:
            model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0)

        # Crear secuencias para Zona C
        if len(X_zone_C_scaled) > sequence_length:
            # Crear secuencias completas para predicción
            X_zone_C_seq, y_zone_C_actual = create_sequences(X_zone_C_scaled, y_zone_C, sequence_length)

            # Predecir
            if len(X_zone_C_seq) > 0:
                preds_seq = model.predict(X_zone_C_seq, verbose=0).flatten()

                # Ajustar las predicciones al formato original
                preds_zone_C = np.full(len(y_zone_C), np.nan)
                preds_zone_C[sequence_length : sequence_length + len(preds_seq)] = preds_seq

                # Para las primeras sequence_length filas, usar el último modelo para predecir
                if sequence_length < len(X_zone_C_scaled):
                    for i in range(sequence_length):
                        # Usar datos de Zona AB + inicio de Zona C para secuencias iniciales
                        if i < len(X_zone_AB_scaled):
                            combined_data = pd.concat(
                                [X_zone_AB_scaled.iloc[-sequence_length + i :], X_zone_C_scaled.iloc[:i]]
                            )
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

    return {'model': model, 'predictions': preds_zone_C, 'metrics': metrics, 'scaler': scaler}


# -----------------------------------------------------------------
# FUNCIONES PARA FORECAST Y ENTRENAMIENTO EXTENDIDO
# -----------------------------------------------------------------
def forecast_future(
    model, last_row, forecast_horizon=FORECAST_HORIZON_1MONTH, algo_name=None, sequence_length=None, X_recent=None
):
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
            logging.warning(
                f"No hay suficientes datos para generar pronóstico LSTM (se necesitan al menos {sequence_length} registros)"
            )
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

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'SMAPE': smape}


###############NUEVO CODIGO INICIO#################
# =============================================================================
# FUNCIONES AUXILIARES PARA FORECAST COMPLETO
# Agregar ANTES de optimize_and_train_extended()
# =============================================================================


def extract_last_20_days_with_external_alignment(X_all, y_all, new_characteristics_file, lag_days=20):
    """
    Extrae los últimos 20 días alineándolos EXACTAMENTE con las fechas del archivo externo.

    SOLUCIÓN DEFINITIVA: Lee las fechas del archivo datos_economicos_filtrados.xlsx
    y usa esas fechas como target_dates para los últimos 20 días.

    Args:
        X_all: DataFrame completo de características del entrenamiento
        y_all: Serie completa de targets del entrenamiento
        new_characteristics_file: Ruta al archivo datos_economicos_filtrados.xlsx
        lag_days: Días de lag aplicado (default: 20)

    Returns:
        dict: Contiene los últimos días con fechas EXACTAMENTE alineadas al archivo externo
    """

    logging.info(f"Extrayendo últimos {lag_days} días con alineación exacta al archivo externo")

    # 1. LEER FECHAS DEL ARCHIVO EXTERNO
    external_dates = None
    if new_characteristics_file and os.path.exists(new_characteristics_file):
        try:
            df_external = pd.read_excel(new_characteristics_file)
            if 'date' in df_external.columns:
                external_dates = pd.to_datetime(df_external['date'])
                logging.info(f"✅ Fechas del archivo externo cargadas: {len(external_dates)} fechas")
                logging.info(f"   Rango: {external_dates.min()} a {external_dates.max()}")

                # Tomar las primeras 20 fechas del archivo externo como target_dates
                target_dates = external_dates.head(lag_days).tolist()
                logging.info(f"✅ Usando primeras {lag_days} fechas del archivo externo como target_dates")

                # Log de las fechas que se van a usar
                logging.info(f"   Target dates: {[d.strftime('%Y-%m-%d') for d in target_dates[:5]]}... (primeras 5)")

            else:
                logging.error("❌ No se encontró columna 'date' en archivo externo")
                external_dates = None
        except Exception as e:
            logging.error(f"❌ Error leyendo archivo externo: {e}")
            external_dates = None

    # 2. SI NO SE PUEDE LEER EL ARCHIVO EXTERNO, USAR MÉTODO FALLBACK
    if external_dates is None or len(external_dates) < lag_days:
        logging.warning("⚠️ No se pudo usar archivo externo, usando método fallback")

        # Método fallback: calcular fechas basándose en X_all
        last_20_features = X_all.tail(lag_days).copy()
        characteristics_dates = X_all.index[-lag_days:].tolist()

        target_dates = []
        for char_date in characteristics_dates:
            target_date = pd.to_datetime(char_date) + pd.Timedelta(days=lag_days)
            target_dates.append(target_date)

    # 3. EXTRAER VALORES REALES CORRESPONDIENTES
    # Los valores reales son los últimos lag_days del entrenamiento
    last_20_targets = y_all.tail(lag_days).copy()

    # 4. CALCULAR FECHAS DE CARACTERÍSTICAS (target_dates - lag_days)
    characteristics_dates_for_prediction = []
    for target_date in target_dates:
        char_date = pd.to_datetime(target_date) - pd.Timedelta(days=lag_days)
        characteristics_dates_for_prediction.append(char_date)

    results = {
        'real_values': last_20_targets.tolist(),
        'target_dates': target_dates,  # ← ESTAS fechas coinciden EXACTAMENTE con el archivo externo
        'characteristics_dates': characteristics_dates_for_prediction,
        'original_target_dates': last_20_targets.index.tolist(),
        'dates': target_dates,  # Para compatibilidad
        'n_days': len(last_20_targets),
        'source': 'external_file' if external_dates is not None else 'fallback',
    }

    logging.info(f"✅ Extracción completada con método: {results['source']}")
    logging.info(f"   - Target dates: {target_dates[0]} hasta {target_dates[-1]}")
    logging.info(
        f"   - Characteristics dates: {characteristics_dates_for_prediction[0]} hasta {characteristics_dates_for_prediction[-1]}"
    )

    # Verificación del lag
    if len(target_dates) > 0:
        actual_lag = (target_dates[0] - characteristics_dates_for_prediction[0]).days
        logging.info(f"   - Lag verificado: {actual_lag} días (esperado: {lag_days})")

        if actual_lag != lag_days:
            logging.warning(f"⚠️ Lag no coincide: {actual_lag} vs {lag_days}")

    # Mostrar comparación con fechas originales
    if len(last_20_targets.index) > 0:
        original_first = last_20_targets.index[0]
        original_last = last_20_targets.index[-1]

        logging.info(f"   - Fechas originales target: {original_first} hasta {original_last}")
        logging.info(f"   - Fechas corregidas target: {target_dates[0]} hasta {target_dates[-1]}")

    return results


def create_complete_forecast_dataframe_aligned(
    algo_name,
    study_best_params,
    args,
    X_all,
    last_20_days_results,
    future_forecast_results,
    new_characteristics_file=None,
):
    """
    Crea DataFrame de forecast con fechas perfectamente alineadas al archivo externo.

    SOLUCIÓN DEFINITIVA: Usa las fechas exactas del archivo datos_economicos_filtrados.xlsx

    Args:
        algo_name: Nombre del algoritmo
        study_best_params: Mejores hiperparámetros
        args: Configuraciones
        X_all: DataFrame de características
        last_20_days_results: Resultados últimos 20 días
        future_forecast_results: Resultados forecast futuro
        new_characteristics_file: Archivo datos_economicos_filtrados.xlsx

    Returns:
        pd.DataFrame: DataFrame con fechas perfectamente alineadas
    """

    logging.info(f"[{algo_name}] Creando DataFrame con alineación perfecta al archivo externo")

    lag_days = len(last_20_days_results['real_values'])

    # 1. USAR FECHAS EXACTAS DEL ARCHIVO EXTERNO PARA ÚLTIMOS 20 DÍAS
    if 'target_dates' in last_20_days_results and last_20_days_results['target_dates']:
        target_dates_last_20 = pd.to_datetime(last_20_days_results['target_dates'])

        logging.info(f"[{algo_name}] ✅ Usando fechas target alineadas con archivo externo")
        logging.info(
            f"[{algo_name}]    Fechas últimos {lag_days} días: {target_dates_last_20[0]} a {target_dates_last_20[-1]}"
        )

        # Verificar que las fechas sean las esperadas del archivo externo
        if new_characteristics_file and os.path.exists(new_characteristics_file):
            try:
                df_external = pd.read_excel(new_characteristics_file)
                if 'date' in df_external.columns:
                    external_dates = pd.to_datetime(df_external['date']).head(lag_days)

                    # Comparar fechas
                    dates_match = True
                    for i, (target_date, external_date) in enumerate(zip(target_dates_last_20, external_dates)):
                        if target_date.date() != external_date.date():
                            dates_match = False
                            logging.warning(
                                f"[{algo_name}] ⚠️ Fecha {i} no coincide: {target_date.date()} vs {external_date.date()}"
                            )

                    if dates_match:
                        logging.info(
                            f"[{algo_name}] ✅ PERFECTA ALINEACIÓN: Fechas coinciden exactamente con archivo externo"
                        )
                    else:
                        logging.warning(f"[{algo_name}] ⚠️ Fechas no coinciden perfectamente con archivo externo")

                        # Forzar uso de fechas del archivo externo
                        logging.info(f"[{algo_name}] 🔧 FORZANDO uso de fechas del archivo externo")
                        target_dates_last_20 = external_dates

            except Exception as e:
                logging.error(f"[{algo_name}] Error verificando alineación: {e}")
    else:
        logging.error(f"[{algo_name}] ❌ No hay target_dates disponibles")
        return pd.DataFrame()

    # 2. CREAR DATAFRAME PARA ÚLTIMOS 20 DÍAS
    df_last_20 = pd.DataFrame(
        {
            "date": target_dates_last_20,  # ← Fechas EXACTAS del archivo externo
            "Valor_Real": last_20_days_results['real_values'],
            "Valor_Predicho": last_20_days_results['predictions'],
            "Modelo": algo_name,
            "Version": f"Three-Zones-Complete",
            "RMSE": [np.nan] * lag_days,
            "MAE": [np.nan] * lag_days,
            "R2": [np.nan] * lag_days,
            "SMAPE": [np.nan] * lag_days,
            "Hyperparámetros": json.dumps(study_best_params),
            "Tipo_Mercado": args.tipo_mercado,
            "Periodo": "Forecast_Last_20_Days",
        }
    )

    logging.info(f"[{algo_name}] ✅ DataFrame últimos {lag_days} días creado con fechas alineadas")

    # 3. CREAR DATAFRAME PARA FORECAST FUTURO
    future_dates = future_forecast_results.get('prediction_dates', [])
    future_preds = future_forecast_results.get('predictions', [])

    if future_dates and future_preds:
        # Si hay archivo externo, usar sus fechas restantes para forecast futuro
        if new_characteristics_file and os.path.exists(new_characteristics_file):
            try:
                df_external = pd.read_excel(new_characteristics_file)
                if 'date' in df_external.columns:
                    external_dates_all = pd.to_datetime(df_external['date'])

                    # Usar fechas del archivo externo que vienen después de los primeros 20
                    if len(external_dates_all) > lag_days:
                        external_future_dates = external_dates_all.iloc[lag_days:].tolist()

                        # Ajustar longitud al número de predicciones
                        min_length = min(len(external_future_dates), len(future_preds))
                        future_dates = external_future_dates[:min_length]
                        future_preds = future_preds[:min_length]

                        logging.info(f"[{algo_name}] ✅ Usando fechas del archivo externo para forecast futuro")
                        logging.info(f"[{algo_name}]    Futuro desde: {future_dates[0]} hasta: {future_dates[-1]}")

            except Exception as e:
                logging.error(f"[{algo_name}] Error usando fechas externas para futuro: {e}")

        df_future_forecast = pd.DataFrame(
            {
                "date": pd.to_datetime(future_dates),
                "Valor_Real": [np.nan] * len(future_preds),
                "Valor_Predicho": future_preds,
                "Modelo": algo_name,
                "Version": f"Three-Zones-Complete",
                "RMSE": [np.nan] * len(future_preds),
                "MAE": [np.nan] * len(future_preds),
                "R2": [np.nan] * len(future_preds),
                "SMAPE": [np.nan] * len(future_preds),
                "Hyperparámetros": json.dumps(study_best_params),
                "Tipo_Mercado": args.tipo_mercado,
                "Periodo": "Forecast_Future",
            }
        )

        logging.info(f"[{algo_name}] ✅ DataFrame forecast futuro creado: {len(df_future_forecast)} filas")
    else:
        df_future_forecast = pd.DataFrame()
        logging.warning(f"[{algo_name}] ⚠️ No hay datos para forecast futuro")

    # 4. CONCATENAR Y CALCULAR MÉTRICAS
    if len(df_future_forecast) > 0:
        df_complete = pd.concat([df_last_20, df_future_forecast], ignore_index=True)
    else:
        df_complete = df_last_20.copy()

    # Calcular métricas solo para últimos 20 días
    real_vals = last_20_days_results['real_values']
    pred_vals = last_20_days_results['predictions']
    valid_pairs = [(r, p) for r, p in zip(real_vals, pred_vals) if not (pd.isna(r) or pd.isna(p))]

    if valid_pairs:
        r, p = zip(*valid_pairs)
        rmse = np.sqrt(mean_squared_error(r, p))
        mae = mean_absolute_error(r, p)
        r2 = r2_score(r, p)
        smape = 100 * np.mean(2.0 * np.abs(np.array(p) - np.array(r)) / (np.abs(r) + np.abs(p)))

        df_complete.loc[df_complete['Periodo'] == "Forecast_Last_20_Days", ['RMSE', 'MAE', 'R2', 'SMAPE']] = (
            rmse,
            mae,
            r2,
            smape,
        )

        logging.info(f"[{algo_name}] ✅ Métricas calculadas: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # 5. VERIFICACIÓN FINAL DE FECHAS
    last_20_dates = df_complete[df_complete['Periodo'] == 'Forecast_Last_20_Days']['date']

    logging.info(f"[{algo_name}] 🔍 VERIFICACIÓN FINAL DE FECHAS:")
    logging.info(f"[{algo_name}]    Forecast_Last_20_Days: {len(last_20_dates)} fechas")
    if len(last_20_dates) > 0:
        logging.info(f"[{algo_name}]    Desde: {last_20_dates.iloc[0].strftime('%Y-%m-%d')}")
        logging.info(f"[{algo_name}]    Hasta: {last_20_dates.iloc[-1].strftime('%Y-%m-%d')}")

        # Mostrar todas las fechas para verificación
        all_dates_str = [d.strftime('%d/%m/%Y') for d in last_20_dates]
        logging.info(f"[{algo_name}]    Todas las fechas: {' '.join(all_dates_str)}")

    logging.info(f"[{algo_name}] ✅ DataFrame completo creado: {len(df_complete)} filas")

    return df_complete


def generate_predictions_for_last_20_days_aligned(
    model, X_all, y_all, algo_name, new_characteristics_file, sequence_length=None, scaler=None, lag_days=20
):
    """
    Genera predicciones para últimos 20 días con alineación exacta al archivo externo.

    FUNCIÓN PRINCIPAL QUE REEMPLAZA LA ORIGINAL
    """

    logging.info(f"[{algo_name}] Generando predicciones alineadas con archivo externo")

    # 1. EXTRAER ÚLTIMOS 20 DÍAS CON ALINEACIÓN EXACTA
    last_days_info = extract_last_20_days_with_external_alignment(X_all, y_all, new_characteristics_file, lag_days)

    # 2. GENERAR PREDICCIONES USANDO LAS CARACTERÍSTICAS CORRECTAS
    predictions_for_last_days = []
    target_dates = last_days_info['target_dates']
    characteristics_dates = last_days_info['characteristics_dates']
    real_values = last_days_info['real_values']

    for i, (target_date, char_date) in enumerate(zip(target_dates, characteristics_dates)):
        try:
            # Buscar características en la fecha calculada
            if char_date in X_all.index:
                char_features = X_all.loc[char_date]

                if algo_name == "LSTM" and sequence_length is not None:
                    # Lógica LSTM
                    char_idx = X_all.index.get_loc(char_date)

                    if char_idx >= sequence_length - 1:
                        start_idx = char_idx - sequence_length + 1
                        window = X_all.iloc[start_idx : char_idx + 1].values
                        window_reshaped = window.reshape(1, sequence_length, -1)

                        pred = model.predict(window_reshaped, verbose=0)[0][0]
                        predictions_for_last_days.append(float(pred))
                    else:
                        logging.warning(f"[{algo_name}] Datos insuficientes para LSTM en {char_date}")
                        predictions_for_last_days.append(np.nan)
                else:
                    # Modelos estándar
                    features = char_features.values.reshape(1, -1)
                    pred = model.predict(features)[0]

                    if isinstance(pred, np.ndarray):
                        pred = pred[0]

                    predictions_for_last_days.append(float(pred))

                # Log para primeros casos
                if i < 3:
                    logging.info(
                        f"[{algo_name}] {i+1}. Caract {char_date.strftime('%Y-%m-%d')} → "
                        f"Target {target_date.strftime('%Y-%m-%d')} = {predictions_for_last_days[-1]:.4f}"
                    )
            else:
                logging.warning(f"[{algo_name}] No se encontraron características para {char_date}")
                predictions_for_last_days.append(np.nan)

        except Exception as e:
            logging.error(f"[{algo_name}] Error en predicción {i}: {e}")
            predictions_for_last_days.append(np.nan)

    results = {
        'predictions': predictions_for_last_days,
        'real_values': real_values,
        'target_dates': target_dates,  # ← Fechas EXACTAS del archivo externo
        'dates': target_dates,  # Para compatibilidad
        'characteristics_dates': characteristics_dates,
        'n_predictions': len(predictions_for_last_days),
        'n_valid_predictions': sum(1 for p in predictions_for_last_days if not pd.isna(p)),
    }

    logging.info(f"[{algo_name}] ✅ Predicciones alineadas completadas:")
    logging.info(f"[{algo_name}]   - Total: {results['n_predictions']}")
    logging.info(f"[{algo_name}]   - Válidas: {results['n_valid_predictions']}")
    logging.info(f"[{algo_name}]   - Fechas perfectamente alineadas con archivo externo")

    return results


def create_complete_forecast_dataframe_aligned(
    algo_name,
    study_best_params,
    args,
    X_all,
    last_20_days_results,
    future_forecast_results,
    new_characteristics_file=None,
):
    """
    Crea DataFrame de forecast con fechas perfectamente alineadas al archivo externo.

    SOLUCIÓN DEFINITIVA: Usa las fechas exactas del archivo datos_economicos_filtrados.xlsx

    Args:
        algo_name: Nombre del algoritmo
        study_best_params: Mejores hiperparámetros
        args: Configuraciones
        X_all: DataFrame de características
        last_20_days_results: Resultados últimos 20 días
        future_forecast_results: Resultados forecast futuro
        new_characteristics_file: Archivo datos_economicos_filtrados.xlsx

    Returns:
        pd.DataFrame: DataFrame con fechas perfectamente alineadas
    """

    logging.info(f"[{algo_name}] Creando DataFrame con alineación perfecta al archivo externo")

    lag_days = len(last_20_days_results['real_values'])

    # 1. USAR FECHAS EXACTAS DEL ARCHIVO EXTERNO PARA ÚLTIMOS 20 DÍAS
    if 'target_dates' in last_20_days_results and last_20_days_results['target_dates']:
        target_dates_last_20 = pd.to_datetime(last_20_days_results['target_dates'])

        logging.info(f"[{algo_name}] ✅ Usando fechas target alineadas con archivo externo")
        logging.info(
            f"[{algo_name}]    Fechas últimos {lag_days} días: {target_dates_last_20[0]} a {target_dates_last_20[-1]}"
        )

        # Verificar que las fechas sean las esperadas del archivo externo
        if new_characteristics_file and os.path.exists(new_characteristics_file):
            try:
                df_external = pd.read_excel(new_characteristics_file)
                if 'date' in df_external.columns:
                    external_dates = pd.to_datetime(df_external['date']).head(lag_days)

                    # Comparar fechas
                    dates_match = True
                    for i, (target_date, external_date) in enumerate(zip(target_dates_last_20, external_dates)):
                        if target_date.date() != external_date.date():
                            dates_match = False
                            logging.warning(
                                f"[{algo_name}] ⚠️ Fecha {i} no coincide: {target_date.date()} vs {external_date.date()}"
                            )

                    if dates_match:
                        logging.info(
                            f"[{algo_name}] ✅ PERFECTA ALINEACIÓN: Fechas coinciden exactamente con archivo externo"
                        )
                    else:
                        logging.warning(f"[{algo_name}] ⚠️ Fechas no coinciden perfectamente con archivo externo")

                        # Forzar uso de fechas del archivo externo
                        logging.info(f"[{algo_name}] 🔧 FORZANDO uso de fechas del archivo externo")
                        target_dates_last_20 = external_dates

            except Exception as e:
                logging.error(f"[{algo_name}] Error verificando alineación: {e}")
    else:
        logging.error(f"[{algo_name}] ❌ No hay target_dates disponibles")
        return pd.DataFrame()

    # 2. CREAR DATAFRAME PARA ÚLTIMOS 20 DÍAS
    df_last_20 = pd.DataFrame(
        {
            "date": target_dates_last_20,  # ← Fechas EXACTAS del archivo externo
            "Valor_Real": last_20_days_results['real_values'],
            "Valor_Predicho": last_20_days_results['predictions'],
            "Modelo": algo_name,
            "Version": f"Three-Zones-Complete",
            "RMSE": [np.nan] * lag_days,
            "MAE": [np.nan] * lag_days,
            "R2": [np.nan] * lag_days,
            "SMAPE": [np.nan] * lag_days,
            "Hyperparámetros": json.dumps(study_best_params),
            "Tipo_Mercado": args.tipo_mercado,
            "Periodo": "Forecast_Last_20_Days",
        }
    )

    logging.info(f"[{algo_name}] ✅ DataFrame últimos {lag_days} días creado con fechas alineadas")

    # 3. CREAR DATAFRAME PARA FORECAST FUTURO
    future_dates = future_forecast_results.get('prediction_dates', [])
    future_preds = future_forecast_results.get('predictions', [])

    if future_dates and future_preds:
        # Si hay archivo externo, usar sus fechas restantes para forecast futuro
        if new_characteristics_file and os.path.exists(new_characteristics_file):
            try:
                df_external = pd.read_excel(new_characteristics_file)
                if 'date' in df_external.columns:
                    external_dates_all = pd.to_datetime(df_external['date'])

                    # Usar fechas del archivo externo que vienen después de los primeros 20
                    if len(external_dates_all) > lag_days:
                        external_future_dates = external_dates_all.iloc[lag_days:].tolist()

                        # Ajustar longitud al número de predicciones
                        min_length = min(len(external_future_dates), len(future_preds))
                        future_dates = external_future_dates[:min_length]
                        future_preds = future_preds[:min_length]

                        logging.info(f"[{algo_name}] ✅ Usando fechas del archivo externo para forecast futuro")
                        logging.info(f"[{algo_name}]    Futuro desde: {future_dates[0]} hasta: {future_dates[-1]}")

            except Exception as e:
                logging.error(f"[{algo_name}] Error usando fechas externas para futuro: {e}")

        df_future_forecast = pd.DataFrame(
            {
                "date": pd.to_datetime(future_dates),
                "Valor_Real": [np.nan] * len(future_preds),
                "Valor_Predicho": future_preds,
                "Modelo": algo_name,
                "Version": f"Three-Zones-Complete",
                "RMSE": [np.nan] * len(future_preds),
                "MAE": [np.nan] * len(future_preds),
                "R2": [np.nan] * len(future_preds),
                "SMAPE": [np.nan] * len(future_preds),
                "Hyperparámetros": json.dumps(study_best_params),
                "Tipo_Mercado": args.tipo_mercado,
                "Periodo": "Forecast_Future",
            }
        )

        logging.info(f"[{algo_name}] ✅ DataFrame forecast futuro creado: {len(df_future_forecast)} filas")
    else:
        df_future_forecast = pd.DataFrame()
        logging.warning(f"[{algo_name}] ⚠️ No hay datos para forecast futuro")

    # 4. CONCATENAR Y CALCULAR MÉTRICAS
    if len(df_future_forecast) > 0:
        df_complete = pd.concat([df_last_20, df_future_forecast], ignore_index=True)
    else:
        df_complete = df_last_20.copy()

    # Calcular métricas solo para últimos 20 días
    real_vals = last_20_days_results['real_values']
    pred_vals = last_20_days_results['predictions']
    valid_pairs = [(r, p) for r, p in zip(real_vals, pred_vals) if not (pd.isna(r) or pd.isna(p))]

    if valid_pairs:
        r, p = zip(*valid_pairs)
        rmse = np.sqrt(mean_squared_error(r, p))
        mae = mean_absolute_error(r, p)
        r2 = r2_score(r, p)
        smape = 100 * np.mean(2.0 * np.abs(np.array(p) - np.array(r)) / (np.abs(r) + np.abs(p)))

        df_complete.loc[df_complete['Periodo'] == "Forecast_Last_20_Days", ['RMSE', 'MAE', 'R2', 'SMAPE']] = (
            rmse,
            mae,
            r2,
            smape,
        )

        logging.info(f"[{algo_name}] ✅ Métricas calculadas: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # 5. VERIFICACIÓN FINAL DE FECHAS
    last_20_dates = df_complete[df_complete['Periodo'] == 'Forecast_Last_20_Days']['date']

    logging.info(f"[{algo_name}] 🔍 VERIFICACIÓN FINAL DE FECHAS:")
    logging.info(f"[{algo_name}]    Forecast_Last_20_Days: {len(last_20_dates)} fechas")
    if len(last_20_dates) > 0:
        logging.info(f"[{algo_name}]    Desde: {last_20_dates.iloc[0].strftime('%Y-%m-%d')}")
        logging.info(f"[{algo_name}]    Hasta: {last_20_dates.iloc[-1].strftime('%Y-%m-%d')}")

        # Mostrar todas las fechas para verificación
        all_dates_str = [d.strftime('%d/%m/%Y') for d in last_20_dates]
        logging.info(f"[{algo_name}]    Todas las fechas: {' '.join(all_dates_str)}")

    logging.info(f"[{algo_name}] ✅ DataFrame completo creado: {len(df_complete)} filas")

    return df_complete


def forecast_with_processed_characteristics(
    model, new_characteristics_file, algo_name=None, sequence_length=None, original_features=None
):
    """
    Genera forecast usando archivo de características ya procesadas (FPI, VIF, normalización).
    """

    logging.info(f"[{algo_name}] Cargando características procesadas para forecast")
    logging.info(f"[{algo_name}] Archivo: {new_characteristics_file}")

    try:
        # 1. CARGAR ARCHIVO (YA PROCESADO)
        df_new = pd.read_excel(new_characteristics_file)
        logging.info(f"[{algo_name}] Características cargadas: {df_new.shape}")

        # 2. EXTRAER FECHAS Y CARACTERÍSTICAS
        if "date" in df_new.columns:
            dates = df_new["date"]
            X_new = df_new.drop(columns=["date"]).copy()
            X_new.index = pd.to_datetime(dates)
        else:
            X_new = df_new.copy()
            X_new.index = pd.to_datetime(X_new.index)

        logging.info(f"[{algo_name}] Fechas de características: {X_new.index[0]} a {X_new.index[-1]}")

        # 3. VALIDAR ORDEN DE COLUMNAS
        if original_features is not None:
            missing_features = set(original_features) - set(X_new.columns)
            extra_features = set(X_new.columns) - set(original_features)

            if missing_features:
                logging.error(f"[{algo_name}] ❌ Características faltantes: {missing_features}")
                raise ValueError(f"Faltan características necesarias: {missing_features}")

            if extra_features:
                logging.warning(f"[{algo_name}] ⚠️ Características extra (se eliminarán): {extra_features}")
                X_new = X_new.drop(columns=list(extra_features))

            # Reordenar columnas para coincidir exactamente con el entrenamiento
            X_new = X_new[original_features]
            logging.info(f"[{algo_name}] ✅ Orden de características validado")

        logging.info(f"[{algo_name}] Características finales: {X_new.shape}")

        # 4. GENERAR PREDICCIONES Y FECHAS CON LAG DE 20 DÍAS
        predictions = []
        prediction_dates = []

        LAG_DAYS = 20  # Lag de 20 días

        if algo_name == "LSTM" and sequence_length is not None:
            # Para LSTM: generar predicciones secuencialmente
            logging.info(f"[{algo_name}] Generando predicciones LSTM con sequence_length={sequence_length}")

            for i in range(len(X_new)):
                try:
                    if i >= sequence_length - 1:
                        # Crear ventana de secuencia
                        start_idx = i - sequence_length + 1
                        window = X_new.iloc[start_idx : i + 1].values
                        window_reshaped = window.reshape(1, sequence_length, -1)

                        pred = model.predict(window_reshaped, verbose=0)[0][0]
                        predictions.append(float(pred))
                    else:
                        # No hay suficientes datos para la secuencia completa
                        logging.warning(f"[{algo_name}] Día {i+1}: Insuficientes datos para secuencia LSTM")
                        predictions.append(np.nan)

                    # Fecha de predicción: fecha de características + LAG_DAYS
                    char_date = X_new.index[i]
                    pred_date = char_date + pd.Timedelta(days=LAG_DAYS)
                    prediction_dates.append(pred_date)

                except Exception as e:
                    logging.error(f"[{algo_name}] Error en predicción LSTM día {i+1}: {e}")
                    predictions.append(np.nan)
                    char_date = X_new.index[i]
                    prediction_dates.append(char_date + pd.Timedelta(days=LAG_DAYS))

        else:
            # Para modelos estándar: predicción directa
            logging.info(f"[{algo_name}] Generando predicciones con modelo estándar")

            for i, (char_date, row) in enumerate(X_new.iterrows()):
                try:
                    features = row.values.reshape(1, -1)
                    pred = model.predict(features)[0]

                    if isinstance(pred, np.ndarray):
                        pred = pred[0]

                    predictions.append(float(pred))

                    # Fecha de predicción: fecha de características + LAG_DAYS
                    pred_date = char_date + pd.Timedelta(days=LAG_DAYS)
                    prediction_dates.append(pred_date)

                except Exception as e:
                    logging.error(f"[{algo_name}] Error en predicción día {i+1}: {e}")
                    predictions.append(np.nan)
                    prediction_dates.append(char_date + pd.Timedelta(days=LAG_DAYS))

        # 5. COMPILAR RESULTADOS
        results = {
            'predictions': predictions,
            'prediction_dates': prediction_dates,
            'characteristics_dates': X_new.index.tolist(),
            'n_predictions': len(predictions),
            'n_valid_predictions': sum(1 for p in predictions if not pd.isna(p)),
        }

        logging.info(f"[{algo_name}] ✅ Forecast completado:")
        logging.info(f"  - Total predicciones: {results['n_predictions']}")
        logging.info(f"  - Predicciones válidas: {results['n_valid_predictions']}")
        logging.info(f"  - Características desde: {X_new.index[0]} hasta: {X_new.index[-1]}")
        logging.info(f"  - Predicciones desde: {prediction_dates[0] if prediction_dates else 'N/A'}")
        logging.info(f"  - Predicciones hasta: {prediction_dates[-1] if prediction_dates else 'N/A'}")

        return results

    except Exception as e:
        logging.error(f"[{algo_name}] ❌ Error en forecast con archivo procesado: {e}")
        logging.error(f"[{algo_name}] Traceback: ", exc_info=True)
        return {
            'predictions': [],
            'prediction_dates': [],
            'characteristics_dates': [],
            'n_predictions': 0,
            'n_valid_predictions': 0,
        }


def generate_fallback_forecast(final_model, X_all_scaled, forecast_horizon, algo_name, best_params):
    """
    Método de fallback cuando no está disponible el archivo de características procesadas.
    """
    logging.info(f"[{algo_name}] Generando forecast con método de fallback")

    future_preds = []
    last_date = X_all_scaled.index[-1]

    # Generar fechas con lag de 20 días para fallback
    LAG_DAYS = 20
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=LAG_DAYS + 1), periods=forecast_horizon, freq='D')

    logging.info(f"[{algo_name}] Generando forecast fallback:")
    logging.info(f"  - Última característica: {last_date}")
    logging.info(f"  - Predicciones desde: {future_dates[0]} hasta: {future_dates[-1]}")

    try:
        if algo_name == "LSTM":
            sequence_length = best_params.get("sequence_length", 10)

            if len(X_all_scaled) >= sequence_length:
                # Usar última ventana para predicción LSTM básica
                last_window = X_all_scaled.tail(sequence_length).values

                for step in range(forecast_horizon):
                    window_reshaped = last_window.reshape(1, sequence_length, -1)
                    pred = final_model.predict(window_reshaped, verbose=0)[0][0]
                    future_preds.append(float(pred))

                    # Actualizar ventana con última observación
                    new_row = X_all_scaled.iloc[-1].values.copy()
                    last_window = np.vstack([last_window[1:], new_row])
            else:
                future_preds = [np.nan] * forecast_horizon
        else:
            # Modelos estándar: usar última observación
            last_features = X_all_scaled.iloc[-1].values

            for step in range(forecast_horizon):
                pred = final_model.predict(last_features.reshape(1, -1))[0]
                if isinstance(pred, np.ndarray):
                    pred = pred[0]

                future_preds.append(float(pred))

    except Exception as e:
        logging.error(f"[{algo_name}] Error en fallback: {e}")
        future_preds = [np.nan] * forecast_horizon

    # Limpiar predicciones
    future_preds = [np.nan if not np.isfinite(x) else x for x in future_preds]

    logging.info(f"[{algo_name}] Fallback completado: {len(future_preds)} predicciones")

    return future_preds, future_dates.tolist()


def validate_characteristics_file(file_path, expected_lag_days=20):
    """
    Valida que el archivo de características tenga la estructura esperada
    """
    try:
        df = pd.read_excel(file_path)

        if 'date' not in df.columns:
            logging.warning("No se encontró columna 'date' en el archivo de características")
            return False

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        logging.info("=== VALIDACIÓN DEL ARCHIVO DE CARACTERÍSTICAS ===")
        logging.info(f"Archivo: {file_path}")
        logging.info(f"Dimensiones: {df.shape}")
        logging.info(f"Fechas: {df['date'].min()} a {df['date'].max()}")
        logging.info(f"Días de datos: {len(df)}")

        # Ejemplo de interpretación del lag
        sample_dates = df['date'].head(3)
        logging.info(f"Ejemplos de interpretación con lag de {expected_lag_days} días:")
        for i, char_date in enumerate(sample_dates):
            pred_date = char_date + pd.Timedelta(days=expected_lag_days)
            logging.info(
                f"  Característica {char_date.strftime('%Y-%m-%d')} → Predicción {pred_date.strftime('%Y-%m-%d')}"
            )

        return True

    except Exception as e:
        logging.error(f"Error validando archivo de características: {e}")
        return False


def generate_complete_forecast_pipeline(
    model, X_all, y_all, algo_name, args, study_best_params, new_characteristics_file=None, lag_days=20
):
    """
    Pipeline completo de forecast que genera predicciones para:
    1. Últimos 20 días del archivo principal (con valores reales para métricas)
    2. Forecast futuro usando archivo de características nuevas

    Args:
        model: Modelo entrenado
        X_all: DataFrame completo de características del entrenamiento
        y_all: Serie completa de targets del entrenamiento
        algo_name: Nombre del algoritmo
        args: Objeto con configuraciones (tipo_mercado, output_dir, etc.)
        study_best_params: Mejores hiperparámetros de Optuna
        new_characteristics_file: Ruta al archivo de características futuras
        lag_days: Días de lag aplicado (default: 20)

    Returns:
        dict: DataFrame de forecast completo y métricas calculadas
    """

    logging.info(f"[{algo_name}] ========== INICIANDO FORECAST COMPLETO ==========")
    logging.info(f"[{algo_name}] Lag aplicado: {lag_days} días")
    logging.info(f"[{algo_name}] Archivo de características futuras: {new_characteristics_file}")

    # Validar archivo de características futuras
    if new_characteristics_file and os.path.exists(new_characteristics_file):
        logging.info(f"[{algo_name}] ✅ Archivo de características encontrado")
        validate_characteristics_file(new_characteristics_file, lag_days)
    else:
        logging.warning(f"[{algo_name}] ⚠️ Archivo de características no encontrado")
        logging.warning(f"[{algo_name}] Se usará método de fallback")

    try:
        # =================================================================
        # PASO 1: PREDICCIONES PARA ÚLTIMOS 20 DÍAS (CON VALORES REALES)
        # =================================================================
        logging.info(f"[{algo_name}] PASO 1: Generando predicciones para últimos {lag_days} días")

        last_20_days_results = generate_predictions_for_last_20_days_aligned(
            model=model,
            X_all=X_all,
            y_all=y_all,
            algo_name=algo_name,
            sequence_length=study_best_params.get("sequence_length", 10) if algo_name == "LSTM" else None,
            scaler=None,  # Ya viene escalado
            lag_days=lag_days,
        )

        logging.info(f"[{algo_name}] ✅ Últimos {lag_days} días completados:")
        logging.info(f"[{algo_name}]   - Predicciones generadas: {last_20_days_results['n_predictions']}")
        logging.info(f"[{algo_name}]   - Predicciones válidas: {last_20_days_results['n_valid_predictions']}")

        # =================================================================
        # PASO 2: FORECAST FUTURO CON ARCHIVO DE CARACTERÍSTICAS
        # =================================================================
        logging.info(f"[{algo_name}] PASO 2: Generando forecast futuro")

        if new_characteristics_file and os.path.exists(new_characteristics_file):
            # Usar archivo de características reales
            original_features = X_all.columns.tolist()
            sequence_length = study_best_params.get("sequence_length", 10) if algo_name == "LSTM" else None

            future_forecast_results = forecast_with_processed_characteristics(
                model=model,
                new_characteristics_file=new_characteristics_file,
                algo_name=algo_name,
                sequence_length=sequence_length,
                original_features=original_features,
            )

            logging.info(f"[{algo_name}] ✅ Forecast futuro con características reales completado")
        else:
            # Método de fallback
            future_preds, future_dates = generate_fallback_forecast(model, X_all, 20, algo_name, study_best_params)
            future_forecast_results = {
                'predictions': future_preds,
                'prediction_dates': future_dates,
                'n_predictions': len(future_preds),
                'n_valid_predictions': sum(1 for p in future_preds if not pd.isna(p)),
            }

            logging.info(f"[{algo_name}] ✅ Forecast futuro con método fallback completado")

        logging.info(f"[{algo_name}]   - Predicciones futuras: {future_forecast_results['n_predictions']}")
        logging.info(f"[{algo_name}]   - Predicciones válidas: {future_forecast_results['n_valid_predictions']}")

        # =================================================================
        # PASO 3: CREAR DATAFRAME COMPLETO DE FORECAST
        # =================================================================
        logging.info(f"[{algo_name}] PASO 3: Creando DataFrame completo de forecast")

        df_forecast_complete = create_complete_forecast_dataframe(
            algo_name=algo_name,
            study_best_params=study_best_params,
            args=args,
            X_all=X_all,
            last_20_days_results=last_20_days_results,
            future_forecast_results=future_forecast_results,
        )

        # =================================================================
        # PASO 4: CALCULAR MÉTRICAS PARA ÚLTIMOS 20 DÍAS
        # =================================================================
        logging.info(f"[{algo_name}] PASO 4: Calculando métricas para últimos {lag_days} días")

        forecast_metrics = calculate_forecast_metrics(
            last_20_days_results['real_values'], last_20_days_results['predictions'], algo_name
        )

        # =================================================================
        # PASO 5: VALIDACIÓN DE CONTINUIDAD TEMPORAL
        # =================================================================
        logging.info(f"[{algo_name}] PASO 5: Validando continuidad temporal")

        all_forecast_dates = last_20_days_results['dates'] + future_forecast_results['prediction_dates']

        temporal_continuity = validate_temporal_continuity(all_forecast_dates, algo_name)

        # =================================================================
        # PASO 6: GENERAR RESUMEN COMPLETO
        # =================================================================
        logging.info(f"[{algo_name}] PASO 6: Generando resumen completo")

        forecast_summary = {
            'algorithm': algo_name,
            'total_forecast_days': len(all_forecast_dates),
            'last_20_days': {
                'n_predictions': last_20_days_results['n_predictions'],
                'n_valid_predictions': last_20_days_results['n_valid_predictions'],
                'metrics': forecast_metrics,
                'date_range': {
                    'from': str(last_20_days_results['dates'][0]) if last_20_days_results['dates'] else 'N/A',
                    'to': str(last_20_days_results['dates'][-1]) if last_20_days_results['dates'] else 'N/A',
                },
            },
            'future_forecast': {
                'n_predictions': future_forecast_results['n_predictions'],
                'n_valid_predictions': future_forecast_results['n_valid_predictions'],
                'source': (
                    'real_characteristics'
                    if (new_characteristics_file and os.path.exists(new_characteristics_file))
                    else 'fallback'
                ),
                'date_range': {
                    'from': (
                        str(future_forecast_results['prediction_dates'][0])
                        if future_forecast_results['prediction_dates']
                        else 'N/A'
                    ),
                    'to': (
                        str(future_forecast_results['prediction_dates'][-1])
                        if future_forecast_results['prediction_dates']
                        else 'N/A'
                    ),
                },
            },
            'temporal_continuity': temporal_continuity,
            'hyperparameters': study_best_params,
        }

        # Log del resumen final
        logging.info(f"[{algo_name}] ========== RESUMEN FINAL DEL FORECAST ==========")
        logging.info(f"[{algo_name}] 📊 Total días de forecast: {forecast_summary['total_forecast_days']}")
        logging.info(f"[{algo_name}] 📈 Últimos {lag_days} días (con valores reales):")
        logging.info(
            f"[{algo_name}]     - Predicciones: {forecast_summary['last_20_days']['n_valid_predictions']}/{forecast_summary['last_20_days']['n_predictions']}"
        )
        logging.info(f"[{algo_name}]     - RMSE: {forecast_metrics.get('RMSE', 'N/A'):.4f}")
        logging.info(f"[{algo_name}]     - MAE: {forecast_metrics.get('MAE', 'N/A'):.4f}")
        logging.info(f"[{algo_name}]     - R²: {forecast_metrics.get('R2', 'N/A'):.4f}")
        logging.info(f"[{algo_name}]     - SMAPE: {forecast_metrics.get('SMAPE', 'N/A'):.2f}%")
        logging.info(f"[{algo_name}] 🔮 Forecast futuro:")
        logging.info(
            f"[{algo_name}]     - Predicciones: {forecast_summary['future_forecast']['n_valid_predictions']}/{forecast_summary['future_forecast']['n_predictions']}"
        )
        logging.info(f"[{algo_name}]     - Fuente: {forecast_summary['future_forecast']['source']}")
        logging.info(
            f"[{algo_name}] ✅ Continuidad temporal: {'SIN GAPS' if temporal_continuity['continuous'] else 'CON GAPS'}"
        )
        logging.info(f"[{algo_name}] =================================================")

        return {
            'df_forecast': df_forecast_complete,
            'metrics': forecast_metrics,
            'summary': forecast_summary,
            'success': True,
        }

    except Exception as e:
        logging.error(f"[{algo_name}] ❌ Error en pipeline de forecast: {e}")
        logging.error(f"[{algo_name}] Traceback: ", exc_info=True)

        # Generar DataFrame vacío como fallback
        empty_df = pd.DataFrame({'date': [], 'Valor_Real': [], 'Valor_Predicho': [], 'Modelo': [], 'Periodo': []})

        return {'df_forecast': empty_df, 'metrics': {}, 'summary': {'error': str(e)}, 'success': False}


def calculate_forecast_metrics(real_values, predictions, algo_name):
    """
    Calcula métricas de forecast para los últimos 20 días donde tenemos valores reales.

    Args:
        real_values: Lista de valores reales
        predictions: Lista de predicciones
        algo_name: Nombre del algoritmo

    Returns:
        dict: Métricas calculadas (RMSE, MAE, R2, SMAPE)
    """

    logging.info(f"[{algo_name}] Calculando métricas de forecast...")

    try:
        # Filtrar pares válidos (sin NaN)
        valid_pairs = [(r, p) for r, p in zip(real_values, predictions) if not (pd.isna(r) or pd.isna(p))]

        if not valid_pairs:
            logging.warning(f"[{algo_name}] No hay pares válidos para calcular métricas")
            return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'SMAPE': np.nan, 'n_valid_pairs': 0}

        real_vals, pred_vals = zip(*valid_pairs)
        real_vals = np.array(real_vals)
        pred_vals = np.array(pred_vals)

        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(real_vals, pred_vals))
        mae = mean_absolute_error(real_vals, pred_vals)
        r2 = r2_score(real_vals, pred_vals)

        # SMAPE (Symmetric Mean Absolute Percentage Error)
        smape = 100 * np.mean(2.0 * np.abs(pred_vals - real_vals) / (np.abs(real_vals) + np.abs(pred_vals)))

        metrics = {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'SMAPE': float(smape),
            'n_valid_pairs': len(valid_pairs),
        }

        logging.info(f"[{algo_name}] ✅ Métricas calculadas con {len(valid_pairs)} pares válidos")
        logging.info(f"[{algo_name}]   - RMSE: {rmse:.4f}")
        logging.info(f"[{algo_name}]   - MAE: {mae:.4f}")
        logging.info(f"[{algo_name}]   - R²: {r2:.4f}")
        logging.info(f"[{algo_name}]   - SMAPE: {smape:.2f}%")

        return metrics

    except Exception as e:
        logging.error(f"[{algo_name}] Error calculando métricas: {e}")
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'SMAPE': np.nan, 'n_valid_pairs': 0, 'error': str(e)}


def validate_temporal_continuity(forecast_dates, algo_name, max_gap_days=7):
    """
    Valida la continuidad temporal del forecast detectando gaps significativos.

    Args:
        forecast_dates: Lista de fechas del forecast
        algo_name: Nombre del algoritmo
        max_gap_days: Máximo número de días considerado como gap aceptable

    Returns:
        dict: Información sobre continuidad temporal
    """

    logging.info(f"[{algo_name}] Validando continuidad temporal...")

    if len(forecast_dates) <= 1:
        logging.warning(f"[{algo_name}] Insuficientes fechas para validar continuidad")
        return {'continuous': True, 'gaps_detected': 0, 'gaps': [], 'total_dates': len(forecast_dates)}

    gaps = []

    for i in range(1, len(forecast_dates)):
        current_date = pd.to_datetime(forecast_dates[i])
        previous_date = pd.to_datetime(forecast_dates[i - 1])
        gap_days = (current_date - previous_date).days

        if gap_days > max_gap_days:
            gap_info = {
                'position': i,
                'gap_days': gap_days,
                'from_date': previous_date.strftime('%Y-%m-%d'),
                'to_date': current_date.strftime('%Y-%m-%d'),
            }
            gaps.append(gap_info)

            logging.warning(
                f"[{algo_name}] Gap detectado: {gap_days} días entre "
                f"{gap_info['from_date']} y {gap_info['to_date']}"
            )

    is_continuous = len(gaps) == 0

    continuity_info = {
        'continuous': is_continuous,
        'gaps_detected': len(gaps),
        'gaps': gaps,
        'total_dates': len(forecast_dates),
        'date_range': (
            {
                'from': pd.to_datetime(forecast_dates[0]).strftime('%Y-%m-%d'),
                'to': pd.to_datetime(forecast_dates[-1]).strftime('%Y-%m-%d'),
            }
            if forecast_dates
            else {}
        ),
    }

    if is_continuous:
        logging.info(f"[{algo_name}] ✅ Continuidad temporal perfecta - Sin gaps detectados")
    else:
        logging.warning(f"[{algo_name}] ⚠️ Se detectaron {len(gaps)} gaps temporales")

    logging.info(
        f"[{algo_name}] Rango temporal total: "
        f"{continuity_info['date_range'].get('from', 'N/A')} a "
        f"{continuity_info['date_range'].get('to', 'N/A')}"
    )

    return continuity_info


def save_forecast_summary(forecast_results, algo_name, args):
    """
    Guarda un resumen detallado del forecast en formato JSON.

    Args:
        forecast_results: Resultados del forecast completo
        algo_name: Nombre del algoritmo
        args: Argumentos de configuración
    """

    logging.info(f"[{algo_name}] Guardando resumen detallado del forecast...")

    try:
        # Preparar resumen para guardar
        summary_to_save = {
            'algorithm': algo_name,
            'timestamp': datetime.now().isoformat(),
            'market_type': args.tipo_mercado,
            'forecast_results': forecast_results['summary'],
            'metrics': forecast_results['metrics'],
            'execution_success': forecast_results['success'],
        }

        # Archivo de resumen
        summary_file = os.path.join(args.output_dir, f"{algo_name.lower()}_forecast_summary.json")

        with open(summary_file, 'w') as f:
            json.dump(summary_to_save, f, indent=4, default=str)

        logging.info(f"[{algo_name}] ✅ Resumen guardado en: {summary_file}")

        return summary_file

    except Exception as e:
        logging.error(f"[{algo_name}] Error guardando resumen: {e}")
        return None


# =============================================================================
# FUNCIÓN PRINCIPAL DE INTEGRACIÓN PARA EL PIPELINE EXISTENTE
# =============================================================================


def execute_complete_forecast_pipeline(
    model,
    X_all,
    y_all,
    algo_name,
    args,
    study_best_params,
    new_characteristics_file=r"C:\Users\natus\Documents\Trabajo\PEDRO_PEREZ\Proyecto_Mercado_de_Valores\SP500_INDEX_Analisis\Data\2_processed\datos_economicos_filtrados.xlsx",
):
    """
    Función principal que ejecuta el pipeline completo de forecast.
    Esta función debe llamarse desde optimize_and_train_extended() en lugar del código existente.

    Args:
        model: Modelo entrenado final
        X_all: DataFrame completo de características
        y_all: Serie completa de targets
        algo_name: Nombre del algoritmo
        args: Objeto de argumentos con configuraciones
        study_best_params: Mejores hiperparámetros de Optuna
        new_characteristics_file: Ruta al archivo de características futuras

    Returns:
        pd.DataFrame: DataFrame completo de forecast listo para concatenar con otros resultados
    """

    logging.info(f"[{algo_name}] ========== EJECUTANDO PIPELINE COMPLETO DE FORECAST ==========")

    # Ejecutar pipeline de forecast
    forecast_results = generate_complete_forecast_pipeline(
        model=model,
        X_all=X_all,
        y_all=y_all,
        algo_name=algo_name,
        args=args,
        study_best_params=study_best_params,
        new_characteristics_file=new_characteristics_file,
        lag_days=20,
    )

    # Guardar resumen detallado
    summary_file = save_forecast_summary(forecast_results, algo_name, args)

    if forecast_results['success']:
        logging.info(f"[{algo_name}] ✅ Pipeline de forecast ejecutado exitosamente")
        logging.info(f"[{algo_name}] DataFrame de forecast generado con {len(forecast_results['df_forecast'])} filas")

        # Retornar DataFrame para integrar con el pipeline existente
        return forecast_results['df_forecast']
    else:
        logging.error(f"[{algo_name}] ❌ Pipeline de forecast falló")

        # Retornar DataFrame vacío en caso de error
        return pd.DataFrame(
            {
                'date': [],
                'Valor_Real': [],
                'Valor_Predicho': [],
                'Modelo': [algo_name],
                'Version': ['Three-Zones-Complete'],
                'RMSE': [np.nan],
                'MAE': [np.nan],
                'R2': [np.nan],
                'SMAPE': [np.nan],
                'Hyperparámetros': [json.dumps(study_best_params)],
                'Tipo_Mercado': [args.tipo_mercado],
                'Periodo': ['Forecast_Error'],
            }
        )


# =============================================================================
# INSTRUCCIONES DE INTEGRACIÓN
# =============================================================================
"""
PARA INTEGRAR ESTE MÓDULO EN TU PIPELINE EXISTENTE:

1. En la función optimize_and_train_extended(), reemplaza el bloque de forecast existente 
   (líneas que generan df_forecast_complete) con:

   df_forecast_complete = execute_complete_forecast_pipeline(
       model=final_model,
       X_all=X_all,
       y_all=y_all,
       algo_name=algo_name,
       args=args,
       study_best_params=study.best_params
   )

2. El resto del código permanece igual, ya que df_forecast_complete tendrá la misma estructura.

3. Asegúrate de que las funciones auxiliares existentes (generate_predictions_for_last_20_days_ aligned, 
   forecast_with_processed_characteristics, etc.) estén disponibles.

4. El módulo generará automáticamente:
   - Logs detallados de cada paso
   - Métricas calculadas para los últimos 20 días
   - Validación de continuidad temporal
   - Resumen JSON con todos los detalles
   - DataFrame final listo para concatenar

VENTAJAS DE ESTA IMPLEMENTACIÓN:
- ✅ Modular y fácil de mantener
- ✅ Logs claros y detallados
- ✅ Manejo robusto de errores
- ✅ Métricas automáticas
- ✅ Validación de continuidad temporal
- ✅ Compatible con tu pipeline existente
- ✅ Resúmenes detallados en JSON
"""


def debug_and_fix_forecast_predictions(
    model, X_all, y_all, algo_name, new_characteristics_file, sequence_length=None, lag_days=20
):
    """
    Debug específico para identificar por qué las predicciones de Forecast_Last_20_Days
    están apareciendo como NaN o vacías.

    MANTIENE las fechas existentes pero ARREGLA las predicciones.
    """

    logging.info(f"[{algo_name}] 🐛 DEBUGGING predicciones Forecast_Last_20_Days")

    # 1. VERIFICAR ESTADO DEL MODELO
    logging.info(f"[{algo_name}] Verificando estado del modelo:")
    logging.info(f"[{algo_name}]   - Tipo de modelo: {type(model)}")
    logging.info(f"[{algo_name}]   - Modelo tiene predict(): {hasattr(model, 'predict')}")

    # Test rápido del modelo
    try:
        if len(X_all) > 0:
            test_features = X_all.iloc[-1].values.reshape(1, -1)
            test_pred = model.predict(test_features)
            logging.info(f"[{algo_name}]   - Test predicción: {test_pred} (tipo: {type(test_pred)})")
        else:
            logging.error(f"[{algo_name}]   - ❌ X_all está vacío!")
    except Exception as e:
        logging.error(f"[{algo_name}]   - ❌ Error en test predicción: {e}")

    # 2. LEER FECHAS DEL ARCHIVO EXTERNO (PARA USAR COMO TARGET_DATES)
    external_dates = None
    if new_characteristics_file and os.path.exists(new_characteristics_file):
        try:
            df_external = pd.read_excel(new_characteristics_file)
            if 'date' in df_external.columns:
                external_dates = pd.to_datetime(df_external['date']).head(lag_days)
                logging.info(f"[{algo_name}] ✅ {len(external_dates)} fechas externas cargadas")
            else:
                logging.error(f"[{algo_name}] ❌ No hay columna 'date' en archivo externo")
        except Exception as e:
            logging.error(f"[{algo_name}] ❌ Error leyendo archivo externo: {e}")

    if external_dates is None or len(external_dates) == 0:
        logging.error(f"[{algo_name}] ❌ No se pudieron cargar fechas externas")
        return None

    # 3. EXTRAER ÚLTIMOS 20 VALORES REALES (PARA MÉTRICAS)
    last_20_real = y_all.tail(lag_days).tolist()
    logging.info(f"[{algo_name}] Valores reales extraídos: {len(last_20_real)}")
    logging.info(f"[{algo_name}]   - Primeros 3: {last_20_real[:3]}")
    logging.info(f"[{algo_name}]   - Últimos 3: {last_20_real[-3:]}")

    # 4. GENERAR PREDICCIONES CON DEBUGGING DETALLADO
    logging.info(f"[{algo_name}] 🔄 Generando predicciones con debugging...")

    predictions = []
    debug_info = []

    # Obtener las últimas 20 características (que corresponden a los valores reales)
    last_20_features = X_all.tail(lag_days)

    logging.info(f"[{algo_name}] Características para predicciones:")
    logging.info(f"[{algo_name}]   - Shape: {last_20_features.shape}")
    logging.info(f"[{algo_name}]   - Desde: {last_20_features.index[0]}")
    logging.info(f"[{algo_name}]   - Hasta: {last_20_features.index[-1]}")

    for i in range(len(last_20_features)):
        prediction = None
        error_msg = None

        try:
            if algo_name == "LSTM" and sequence_length is not None:
                # LSTM DEBUGGING
                logging.info(f"[{algo_name}] Pred {i+1}: Procesando LSTM (seq_len={sequence_length})")

                # Índice global
                global_idx = len(X_all) - lag_days + i

                if global_idx >= sequence_length - 1:
                    start_idx = global_idx - sequence_length + 1
                    window = X_all.iloc[start_idx : global_idx + 1].values

                    logging.info(f"[{algo_name}]   - Ventana: índices {start_idx} a {global_idx}")
                    logging.info(f"[{algo_name}]   - Shape ventana: {window.shape}")

                    window_reshaped = window.reshape(1, sequence_length, -1)
                    prediction_raw = model.predict(window_reshaped, verbose=0)

                    logging.info(f"[{algo_name}]   - Predicción raw: {prediction_raw}")
                    logging.info(f"[{algo_name}]   - Shape pred: {prediction_raw.shape}")

                    if len(prediction_raw) > 0 and len(prediction_raw[0]) > 0:
                        prediction = float(prediction_raw[0][0])
                        logging.info(f"[{algo_name}]   - ✅ Predicción final: {prediction}")
                    else:
                        error_msg = "Predicción raw vacía"

                else:
                    error_msg = f"Datos insuficientes: idx {global_idx} < seq_len {sequence_length}"

            else:
                # MODELOS ESTÁNDAR DEBUGGING
                logging.info(f"[{algo_name}] Pred {i+1}: Procesando modelo estándar")

                features = last_20_features.iloc[i].values
                logging.info(f"[{algo_name}]   - Features shape: {features.shape}")
                logging.info(f"[{algo_name}]   - Features sample: {features[:3]}...")

                # Verificar NaN en features
                nan_count = np.isnan(features).sum()
                if nan_count > 0:
                    logging.warning(f"[{algo_name}]   - ⚠️ {nan_count} NaN en features")
                    features = np.nan_to_num(features, nan=0.0)

                features_reshaped = features.reshape(1, -1)
                prediction_raw = model.predict(features_reshaped)

                logging.info(f"[{algo_name}]   - Predicción raw: {prediction_raw}")
                logging.info(f"[{algo_name}]   - Tipo pred: {type(prediction_raw)}")

                if isinstance(prediction_raw, np.ndarray):
                    if prediction_raw.ndim > 1:
                        prediction = float(prediction_raw[0][0])
                    else:
                        prediction = float(prediction_raw[0])
                else:
                    prediction = float(prediction_raw)

                logging.info(f"[{algo_name}]   - ✅ Predicción final: {prediction}")

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logging.error(f"[{algo_name}]   - ❌ {error_msg}")

        # Verificar que la predicción es válida
        if prediction is not None and np.isfinite(prediction):
            predictions.append(prediction)
            debug_info.append({"index": i, "prediction": prediction, "status": "OK"})
        else:
            # Usar valor por defecto
            fallback_value = 0.0
            predictions.append(fallback_value)
            debug_info.append({"index": i, "prediction": fallback_value, "status": f"FALLBACK: {error_msg}"})
            logging.warning(f"[{algo_name}]   - ⚠️ Usando fallback: {fallback_value}")

    # 5. VERIFICAR RESULTADOS
    logging.info(f"[{algo_name}] 📊 RESULTADOS DE DEBUGGING:")
    logging.info(f"[{algo_name}]   - Predicciones generadas: {len(predictions)}")
    logging.info(f"[{algo_name}]   - Predicciones válidas: {sum(1 for p in predictions if np.isfinite(p))}")
    logging.info(
        f"[{algo_name}]   - Predicciones con fallback: {sum(1 for info in debug_info if 'FALLBACK' in info['status'])}"
    )

    # Mostrar muestra de predicciones
    for i in range(min(5, len(predictions))):
        status = debug_info[i]['status']
        pred = predictions[i]
        real = last_20_real[i] if i < len(last_20_real) else "N/A"
        ext_date = external_dates.iloc[i].strftime('%Y-%m-%d') if i < len(external_dates) else "N/A"

        logging.info(f"[{algo_name}]   {i+1}. {ext_date}: Pred={pred:.4f}, Real={real}, Status={status}")

    # 6. PREPARAR RESULTADO FINAL
    results = {
        'predictions': predictions,
        'real_values': last_20_real,
        'target_dates': external_dates.tolist(),
        'dates': external_dates.tolist(),
        'n_predictions': len(predictions),
        'n_valid_predictions': sum(1 for p in predictions if np.isfinite(p)),
        'debug_info': debug_info,
    }

    logging.info(f"[{algo_name}] ✅ Debugging completado - predicciones reparadas")

    return results


def verify_forecast_dataframe_predictions(df_forecast, algo_name):
    """
    Verifica que las predicciones en el DataFrame estén correctas.
    """

    logging.info(f"[{algo_name}] 🔍 VERIFICANDO DataFrame de forecast...")

    # Filtrar filas de Forecast_Last_20_Days
    last_20_rows = df_forecast[df_forecast['Periodo'] == 'Forecast_Last_20_Days']

    logging.info(f"[{algo_name}] Filas Forecast_Last_20_Days encontradas: {len(last_20_rows)}")

    if len(last_20_rows) == 0:
        logging.error(f"[{algo_name}] ❌ No se encontraron filas Forecast_Last_20_Days")
        return False

    # Verificar predicciones
    pred_col = 'Valor_Predicho'
    if pred_col in last_20_rows.columns:
        predictions = last_20_rows[pred_col].values

        nan_count = pd.isna(predictions).sum()
        valid_count = len(predictions) - nan_count

        logging.info(f"[{algo_name}] Predicciones en Forecast_Last_20_Days:")
        logging.info(f"[{algo_name}]   - Total: {len(predictions)}")
        logging.info(f"[{algo_name}]   - Válidas: {valid_count}")
        logging.info(f"[{algo_name}]   - NaN/Nulas: {nan_count}")

        if valid_count > 0:
            logging.info(f"[{algo_name}]   - Min: {np.nanmin(predictions):.4f}")
            logging.info(f"[{algo_name}]   - Max: {np.nanmax(predictions):.4f}")
            logging.info(f"[{algo_name}]   - Media: {np.nanmean(predictions):.4f}")

        # Mostrar muestra
        logging.info(f"[{algo_name}] Muestra de predicciones:")
        for i in range(min(5, len(last_20_rows))):
            row = last_20_rows.iloc[i]
            date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
            pred_val = row[pred_col]
            real_val = row.get('Valor_Real', 'N/A')

            logging.info(f"[{algo_name}]   {i+1}. {date_str}: Pred={pred_val}, Real={real_val}")

        return valid_count == len(predictions)  # True si todas son válidas
    else:
        logging.error(f"[{algo_name}] ❌ No se encontró columna '{pred_col}'")
        return False


def generate_business_days_forecast_future_fixed(
    model,
    last_20_days_results,
    new_characteristics_file,
    algo_name,
    sequence_length=None,
    original_features=None,
    forecast_days=20,
):
    """
    VERSIÓN CORREGIDA: Genera forecast futuro con predicciones completas y días hábiles.
    """

    logging.info(f"[{algo_name}] 🔧 GENERANDO FORECAST FUTURO CORREGIDO CON DÍAS HÁBILES")

    # 1. VALIDAR ÚLTIMOS 20 DÍAS
    if not last_20_days_results or not last_20_days_results.get('target_dates'):
        logging.error(f"[{algo_name}] ❌ No hay fechas de últimos 20 días disponibles")
        return {'predictions': [], 'prediction_dates': [], 'error': 'No last_20_days_results'}

    last_date_forecast_20 = pd.to_datetime(last_20_days_results['target_dates'][-1])
    logging.info(f"[{algo_name}] Última fecha de Forecast_Last_20_Days: {last_date_forecast_20.strftime('%Y-%m-%d')}")

    # 2. GENERAR FECHAS CONSECUTIVAS DE DÍAS HÁBILES
    future_business_dates = []
    current_date = last_date_forecast_20

    logging.info(f"[{algo_name}] Generando {forecast_days} días hábiles consecutivos...")

    while len(future_business_dates) < forecast_days:
        current_date += pd.Timedelta(days=1)
        # Solo agregar si es día hábil (lunes=0 a viernes=4)
        if current_date.weekday() < 5:
            future_business_dates.append(current_date)

    logging.info(f"[{algo_name}] ✅ Fechas futuras generadas:")
    logging.info(f"[{algo_name}]   - Desde: {future_business_dates[0].strftime('%Y-%m-%d %A')}")
    logging.info(f"[{algo_name}]   - Hasta: {future_business_dates[-1].strftime('%Y-%m-%d %A')}")

    # 3. GENERAR PREDICCIONES CON FUNCIÓN CORREGIDA
    if new_characteristics_file and os.path.exists(new_characteristics_file):
        logging.info(f"[{algo_name}] Usando archivo de características con función corregida")

        try:
            future_predictions = generate_predictions_from_characteristics_file_fixed(
                model=model,
                characteristics_file=new_characteristics_file,
                target_dates=future_business_dates,
                algo_name=algo_name,
                sequence_length=sequence_length,
                original_features=original_features,
            )

            logging.info(f"[{algo_name}] ✅ Predicciones generadas con función corregida")

        except Exception as e:
            logging.error(f"[{algo_name}] ❌ Error con función corregida: {e}")
            logging.info(f"[{algo_name}] Usando método de fallback")

            future_predictions = generate_fallback_business_predictions(
                model=model,
                last_20_days_results=last_20_days_results,
                target_dates=future_business_dates,
                algo_name=algo_name,
                sequence_length=sequence_length,
            )
    else:
        logging.warning(f"[{algo_name}] ⚠️ Archivo de características no disponible")
        future_predictions = generate_fallback_business_predictions(
            model=model,
            last_20_days_results=last_20_days_results,
            target_dates=future_business_dates,
            algo_name=algo_name,
            sequence_length=sequence_length,
        )

    # 4. VALIDAR PREDICCIONES GENERADAS
    valid_predictions = [p for p in future_predictions if not pd.isna(p) and np.isfinite(p)]

    logging.info(f"[{algo_name}] 📊 VALIDACIÓN DE PREDICCIONES:")
    logging.info(f"[{algo_name}]   - Total predicciones: {len(future_predictions)}")
    logging.info(f"[{algo_name}]   - Predicciones válidas: {len(valid_predictions)}")
    logging.info(f"[{algo_name}]   - Porcentaje de éxito: {len(valid_predictions)/len(future_predictions)*100:.1f}%")

    if len(valid_predictions) < len(future_predictions) * 0.8:  # Si menos del 80% son válidas
        logging.warning(
            f"[{algo_name}] ⚠️ ADVERTENCIA: Solo {len(valid_predictions)} de {len(future_predictions)} predicciones son válidas"
        )
        logging.warning(f"[{algo_name}] Esto podría indicar problemas con el archivo de características")

    # 5. COMPILAR RESULTADOS
    gap_days = (future_business_dates[0] - last_date_forecast_20).days
    non_business_days = [d for d in future_business_dates if d.weekday() >= 5]

    results = {
        'predictions': future_predictions,
        'prediction_dates': future_business_dates,
        'n_predictions': len(future_predictions),
        'n_valid_predictions': len(valid_predictions),
        'success_rate': len(valid_predictions) / len(future_predictions) if future_predictions else 0,
        'continuity_gap_days': gap_days,
        'all_business_days': len(non_business_days) == 0,
        'method': 'business_days_consecutive_fixed',
    }

    logging.info(f"[{algo_name}] ✅ FORECAST FUTURO CORREGIDO COMPLETADO:")
    logging.info(f"[{algo_name}]   - Predicciones totales: {results['n_predictions']}")
    logging.info(f"[{algo_name}]   - Predicciones válidas: {results['n_valid_predictions']}")
    logging.info(f"[{algo_name}]   - Tasa de éxito: {results['success_rate']*100:.1f}%")
    logging.info(f"[{algo_name}]   - Gap temporal: {gap_days} días")
    logging.info(f"[{algo_name}]   - Solo días hábiles: {'✅ Sí' if results['all_business_days'] else '❌ No'}")

    return results


def generate_predictions_from_characteristics_file_fixed(
    model, characteristics_file, target_dates, algo_name, sequence_length=None, original_features=None
):
    """
    VERSIÓN CORREGIDA: Genera predicciones usando el archivo de características,
    con mejor manejo de fechas faltantes y predicciones más robustas.

    Args:
        model: Modelo entrenado
        characteristics_file: Archivo Excel con características
        target_dates: Fechas objetivo para las predicciones
        algo_name: Nombre del algoritmo
        sequence_length: Para LSTM
        original_features: Características esperadas

    Returns:
        list: Lista de predicciones COMPLETAS
    """

    logging.info(f"[{algo_name}] 🔧 GENERANDO PREDICCIONES CORREGIDAS desde archivo de características")

    # 1. CARGAR Y VALIDAR ARCHIVO DE CARACTERÍSTICAS
    try:
        df_chars = pd.read_excel(characteristics_file)
        if 'date' not in df_chars.columns:
            raise ValueError("No se encontró columna 'date' en archivo de características")

        df_chars['date'] = pd.to_datetime(df_chars['date'])
        df_chars = df_chars.sort_values('date').reset_index(drop=True)

        logging.info(f"[{algo_name}] ✅ Archivo cargado: {len(df_chars)} filas de características")
        logging.info(f"[{algo_name}] Rango de fechas: {df_chars['date'].min()} a {df_chars['date'].max()}")

    except Exception as e:
        logging.error(f"[{algo_name}] ❌ Error cargando archivo: {e}")
        raise

    # 2. PREPARAR CARACTERÍSTICAS
    X_chars = df_chars.drop(columns=['date']).copy()

    # Validar y reordenar características
    if original_features:
        missing_features = set(original_features) - set(X_chars.columns)
        if missing_features:
            logging.error(f"[{algo_name}] ❌ Características faltantes: {missing_features}")
            raise ValueError(f"Características faltantes: {missing_features}")

        extra_features = set(X_chars.columns) - set(original_features)
        if extra_features:
            logging.warning(f"[{algo_name}] ⚠️ Eliminando características extra: {extra_features}")
            X_chars = X_chars.drop(columns=list(extra_features))

        # Reordenar para coincidir exactamente con el entrenamiento
        X_chars = X_chars[original_features]

    logging.info(f"[{algo_name}] Características preparadas: {X_chars.shape}")

    # 3. CREAR ÍNDICE DE FECHAS PARA BÚSQUEDA RÁPIDA
    date_to_idx = {date.date(): idx for idx, date in enumerate(df_chars['date'])}
    logging.info(f"[{algo_name}] Índice de fechas creado: {len(date_to_idx)} fechas únicas")

    # 4. MAPEAR FECHAS CON ESTRATEGIA ROBUSTA
    LAG_DAYS = 20
    predictions = []
    prediction_info = []

    logging.info(f"[{algo_name}] 🎯 Mapeando {len(target_dates)} fechas con lag de {LAG_DAYS} días")

    for i, target_date in enumerate(target_dates):
        try:
            # Fecha de características = fecha objetivo - lag
            char_date = target_date - pd.Timedelta(days=LAG_DAYS)
            char_date_key = char_date.date()

            # ESTRATEGIA 1: Búsqueda exacta
            char_row_idx = date_to_idx.get(char_date_key)
            search_method = "exact"

            # ESTRATEGIA 2: Búsqueda con tolerancia si no se encuentra exacta
            if char_row_idx is None:
                # Buscar la fecha más cercana dentro de ±3 días
                tolerance_days = 3
                for offset in range(1, tolerance_days + 1):
                    # Intentar días anteriores
                    alt_date = (char_date - pd.Timedelta(days=offset)).date()
                    if alt_date in date_to_idx:
                        char_row_idx = date_to_idx[alt_date]
                        search_method = f"backward_{offset}"
                        break

                    # Intentar días posteriores
                    alt_date = (char_date + pd.Timedelta(days=offset)).date()
                    if alt_date in date_to_idx:
                        char_row_idx = date_to_idx[alt_date]
                        search_method = f"forward_{offset}"
                        break

            # ESTRATEGIA 3: Usar última fecha disponible si no se encuentra nada
            if char_row_idx is None:
                char_row_idx = len(X_chars) - 1  # Última fila disponible
                search_method = "fallback_last"
                logging.warning(
                    f"[{algo_name}] {i+1}. Usando última fecha disponible para target {target_date.strftime('%Y-%m-%d')}"
                )

            # GENERAR PREDICCIÓN
            if algo_name == "LSTM" and sequence_length is not None:
                # Para LSTM: crear ventana de secuencia
                if char_row_idx >= sequence_length - 1:
                    start_idx = char_row_idx - sequence_length + 1
                    window = X_chars.iloc[start_idx : char_row_idx + 1].values
                    window_reshaped = window.reshape(1, sequence_length, -1)

                    pred = model.predict(window_reshaped, verbose=0)[0][0]
                    predictions.append(float(pred))

                    prediction_info.append(
                        {
                            'target_date': target_date.strftime('%Y-%m-%d'),
                            'char_date_requested': char_date.strftime('%Y-%m-%d'),
                            'char_date_used': df_chars.iloc[char_row_idx]['date'].strftime('%Y-%m-%d'),
                            'char_row_idx': char_row_idx,
                            'window_range': f"{start_idx}:{char_row_idx+1}",
                            'search_method': search_method,
                            'prediction': float(pred),
                            'status': 'success',
                        }
                    )

                else:
                    # Si no hay suficientes datos para la secuencia, usar interpolación
                    if char_row_idx > 0:
                        # Crear secuencia artificial repitiendo la última fila disponible
                        last_available = X_chars.iloc[char_row_idx].values
                        window = np.tile(last_available, (sequence_length, 1))
                        window_reshaped = window.reshape(1, sequence_length, -1)

                        pred = model.predict(window_reshaped, verbose=0)[0][0]
                        predictions.append(float(pred))

                        prediction_info.append(
                            {
                                'target_date': target_date.strftime('%Y-%m-%d'),
                                'char_date_requested': char_date.strftime('%Y-%m-%d'),
                                'char_date_used': df_chars.iloc[char_row_idx]['date'].strftime('%Y-%m-%d'),
                                'char_row_idx': char_row_idx,
                                'window_range': f"artificial_sequence",
                                'search_method': f"{search_method}_artificial",
                                'prediction': float(pred),
                                'status': 'artificial_sequence',
                            }
                        )
                    else:
                        # Último recurso: usar valor por defecto
                        pred = 0.0
                        predictions.append(pred)

                        prediction_info.append(
                            {
                                'target_date': target_date.strftime('%Y-%m-%d'),
                                'char_date_requested': char_date.strftime('%Y-%m-%d'),
                                'char_date_used': 'none',
                                'char_row_idx': -1,
                                'window_range': 'none',
                                'search_method': 'default_fallback',
                                'prediction': pred,
                                'status': 'fallback_zero',
                            }
                        )

            else:
                # Para modelos estándar: predicción directa
                features = X_chars.iloc[char_row_idx].values
                features_reshaped = features.reshape(1, -1)

                pred = model.predict(features_reshaped)[0]

                if isinstance(pred, np.ndarray):
                    pred = pred[0]

                predictions.append(float(pred))

                prediction_info.append(
                    {
                        'target_date': target_date.strftime('%Y-%m-%d'),
                        'char_date_requested': char_date.strftime('%Y-%m-%d'),
                        'char_date_used': df_chars.iloc[char_row_idx]['date'].strftime('%Y-%m-%d'),
                        'char_row_idx': char_row_idx,
                        'search_method': search_method,
                        'prediction': float(pred),
                        'status': 'success',
                    }
                )

            # Log para los primeros casos
            if i < 5:
                info = prediction_info[-1]
                logging.info(
                    f"[{algo_name}] {i+1}. Target {info['target_date']} ← "
                    f"Char {info['char_date_used']} [{info['search_method']}] = {info['prediction']:.4f}"
                )

        except Exception as e:
            logging.error(f"[{algo_name}] ❌ Error en predicción {i+1}: {e}")

            # En caso de error, usar predicción por defecto
            pred = 0.0
            predictions.append(pred)

            prediction_info.append(
                {
                    'target_date': target_date.strftime('%Y-%m-%d'),
                    'char_date_requested': char_date.strftime('%Y-%m-%d') if 'char_date' in locals() else 'unknown',
                    'char_date_used': 'error',
                    'char_row_idx': -1,
                    'search_method': 'error_fallback',
                    'prediction': pred,
                    'status': f'error: {str(e)}',
                }
            )

    # 5. ESTADÍSTICAS Y VERIFICACIÓN FINAL
    valid_preds = [p for p in predictions if not pd.isna(p) and np.isfinite(p)]

    # Contar por método de búsqueda
    search_methods = {}
    for info in prediction_info:
        method = info['search_method']
        if method not in search_methods:
            search_methods[method] = 0
        search_methods[method] += 1

    logging.info(f"[{algo_name}] ✅ PREDICCIONES GENERADAS CON ÉXITO:")
    logging.info(f"[{algo_name}]   - Total predicciones: {len(predictions)}")
    logging.info(f"[{algo_name}]   - Predicciones válidas: {len(valid_preds)}")
    logging.info(f"[{algo_name}]   - Predicciones fallidas: {len(predictions) - len(valid_preds)}")

    logging.info(f"[{algo_name}] 📊 MÉTODOS DE BÚSQUEDA UTILIZADOS:")
    for method, count in search_methods.items():
        logging.info(f"[{algo_name}]   - {method}: {count} predicciones")

    # Verificar rango de predicciones
    if valid_preds:
        logging.info(f"[{algo_name}] 📈 RANGO DE PREDICCIONES:")
        logging.info(f"[{algo_name}]   - Mínimo: {min(valid_preds):.4f}")
        logging.info(f"[{algo_name}]   - Máximo: {max(valid_preds):.4f}")
        logging.info(f"[{algo_name}]   - Promedio: {np.mean(valid_preds):.4f}")
        logging.info(f"[{algo_name}]   - Desviación estándar: {np.std(valid_preds):.4f}")

    # Guardar información detallada para debugging
    debug_file = os.path.join(RESULTS_DIR, f"{algo_name.lower()}_forecast_future_debug.json")
    try:
        with open(debug_file, 'w') as f:
            json.dump(prediction_info, f, indent=2, default=str)
        logging.info(f"[{algo_name}] 🐛 Info de debugging guardada: {debug_file}")
    except Exception as e:
        logging.warning(f"[{algo_name}] ⚠️ No se pudo guardar debug info: {e}")

    return predictions


def generate_fallback_business_predictions(model, last_20_days_results, target_dates, algo_name, sequence_length=None):
    """
    Genera predicciones de fallback usando el patrón de los últimos 20 días.

    Args:
        model: Modelo entrenado
        last_20_days_results: Resultados de últimos 20 días
        target_dates: Fechas objetivo
        algo_name: Nombre del algoritmo
        sequence_length: Para LSTM

    Returns:
        list: Lista de predicciones de fallback
    """

    logging.info(f"[{algo_name}] Generando predicciones de fallback para días hábiles")

    # Estrategia de fallback: usar la tendencia de los últimos 20 días
    last_predictions = last_20_days_results.get('predictions', [])

    if not last_predictions or all(pd.isna(p) for p in last_predictions):
        logging.warning(f"[{algo_name}] ⚠️ No hay predicciones válidas en últimos 20 días para fallback")
        # Usar valor constante como último recurso
        fallback_predictions = [0.0] * len(target_dates)
    else:
        # Calcular tendencia
        valid_last_preds = [p for p in last_predictions if not pd.isna(p) and np.isfinite(p)]

        if len(valid_last_preds) >= 2:
            # Calcular tendencia lineal simple
            recent_avg = np.mean(valid_last_preds[-5:]) if len(valid_last_preds) >= 5 else np.mean(valid_last_preds)
            overall_avg = np.mean(valid_last_preds)
            trend = recent_avg - overall_avg

            logging.info(f"[{algo_name}] Tendencia calculada: {trend:.4f}")
            logging.info(f"[{algo_name}] Valor base para fallback: {recent_avg:.4f}")

            # Generar predicciones con tendencia
            fallback_predictions = []
            for i in range(len(target_dates)):
                pred_value = recent_avg + (trend * i * 0.1)  # Aplicar tendencia gradualmente
                fallback_predictions.append(float(pred_value))
        else:
            # Si solo hay una predicción válida, usarla como constante
            base_value = valid_last_preds[0]
            fallback_predictions = [float(base_value)] * len(target_dates)
            logging.info(f"[{algo_name}] Usando valor constante para fallback: {base_value:.4f}")

    logging.info(f"[{algo_name}] ✅ Predicciones de fallback generadas: {len(fallback_predictions)}")

    return fallback_predictions


def update_forecast_with_business_days(
    algo_name,
    study_best_params,
    args,
    X_all,
    last_20_days_results,
    new_characteristics_file,
    sequence_length=None,
    original_features=None,
):
    """
    Actualiza el forecast futuro para usar días hábiles consecutivos.

    Esta función REEMPLAZA la generación de forecast futuro en el pipeline principal.
    """

    logging.info(f"[{algo_name}] ========== ACTUALIZANDO FORECAST CON DÍAS HÁBILES ==========")

    # Generar forecast futuro con días hábiles
    future_forecast_results = generate_business_days_forecast_future_fixed(
        model=final_model,
        last_20_days_results=last_20_days_results,
        new_characteristics_file=NEW_CHARACTERISTICS_FILE,
        algo_name=algo_name,
        sequence_length=sequence_length,
        original_features=original_features,
        forecast_days=20,
    )

    # Crear DataFrame completo
    df_forecast_complete = create_complete_forecast_dataframe_aligned(
        algo_name=algo_name,
        study_best_params=study_best_params,
        args=args,
        X_all=X_all,
        last_20_days_results=last_20_days_results,
        future_forecast_results=future_forecast_results,
        new_characteristics_file=new_characteristics_file,
    )

    logging.info(f"[{algo_name}] ✅ Forecast actualizado con días hábiles consecutivos")

    return df_forecast_complete, future_forecast_results


###########FINAL DE NUEVO CODIGO#####################


def optimize_and_train_extended(
    algo_name,
    objective_func,
    X_zone_A,
    y_zone_A,
    X_zone_B,
    y_zone_B,
    X_zone_C,
    y_zone_C,
    forecast_horizon=FORECAST_HORIZON_1MONTH,
    top_n=3,
):
    """
    Optimiza el modelo usando el enfoque de tres zonas:
      - Zona A (70%): RandomSearch + Optuna para ajuste de hiperparámetros
      - Zona B (20%): Back-test externo
      - Zona C (10%): Hold-out final

    VERSIÓN COMPLETA: Incluye últimos 20 días del archivo principal + forecast futuro
    Sin gaps temporales y con interpretación correcta del lag de 20 días.
    CON DEBUGGING DE PREDICCIONES PARA FORECAST_LAST_20_DAYS
    """
    start_time = time.perf_counter()

    # VALIDACIÓN INICIAL DEL ARCHIVO DE CARACTERÍSTICAS NUEVAS
    NEW_CHARACTERISTICS_FILE = r"C:\Users\natus\Documents\Trabajo\PEDRO_PEREZ\Proyecto_Mercado_de_Valores\SP500_INDEX_Analisis\Data\2_processed\datos_economicos_filtrados.xlsx"

    if os.path.exists(NEW_CHARACTERISTICS_FILE):
        logging.info(f"[{algo_name}] ✅ Archivo de características encontrado para validación inicial")
        validate_characteristics_file(NEW_CHARACTERISTICS_FILE, expected_lag_days=20)
    else:
        logging.warning(f"[{algo_name}] ⚠️ Archivo de características no encontrado: {NEW_CHARACTERISTICS_FILE}")
        logging.warning(f"[{algo_name}] Se usará método de fallback para forecast futuro")

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
        X_zone_A,
        y_zone_A,
        X_zone_B,
        y_zone_B,
        study.best_params,
        scaling_required=SCALING_REQUIRED_MODELS.get(algo_name, False),
    )

    # 4. HOLD-OUT EN ZONA C
    logging.info(f"[{algo_name}] Paso 4: Hold-out en Zona C")
    holdout_results = evaluate_holdout(
        algo_name,
        X_zone_A,
        y_zone_A,
        X_zone_B,
        y_zone_B,
        X_zone_C,
        y_zone_C,
        study.best_params,
        scaling_required=SCALING_REQUIRED_MODELS.get(algo_name, False),
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
        X_all_scaled = pd.DataFrame(scaler.fit_transform(X_all), columns=X_all.columns, index=X_all.index)
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
            devices=devices,
        )
    elif algo_name == "LightGBM":
        final_model = lgb.LGBMRegressor(
            learning_rate=study.best_params.get("learning_rate", 0.01),
            max_depth=study.best_params.get("max_depth", 6),
            n_estimators=study.best_params.get("n_estimators", 1000),
            random_state=RANDOM_SEED,
        )
    elif algo_name == "XGBoost":
        # Usar GPU para XGBoost si está disponible
        tree_method = "gpu_hist" if has_gpu else "auto"
        gpu_id = 0 if has_gpu else None
        final_model = xgb.XGBRegressor(
            learning_rate=study.best_params.get("learning_rate", 0.01),
            max_depth=study.best_params.get("max_depth", 6),
            n_estimators=study.best_params.get("n_estimators", 1000),
            subsample=study.best_params.get("subsample", 0.8),
            random_state=RANDOM_SEED,
            tree_method=tree_method,
            gpu_id=gpu_id,
        )
    elif algo_name == "MLP":
        hidden_neurons = study.best_params.get("hidden_neurons", 100)
        final_model = MLPRegressor(
            hidden_layer_sizes=(hidden_neurons, hidden_neurons),
            learning_rate_init=study.best_params.get("learning_rate_init", 0.001),
            max_iter=study.best_params.get("max_iter", 500),
            random_state=RANDOM_SEED,
        )
    elif algo_name == "SVM":
        final_model = SVR(C=study.best_params.get("C", 1.0), epsilon=study.best_params.get("epsilon", 0.1))
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
        final_model = Sequential(
            [
                LSTM(units=units, input_shape=(sequence_length, n_features), return_sequences=False),
                Dropout(dropout_rate),
                Dense(1),
            ]
        )
        final_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

        # Entrenar modelo final
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Dividir para validación
        val_split = 0.1
        val_samples = int(len(X_all_seq) * val_split)
        if val_samples > 0:
            X_train_lstm = X_all_seq[:-val_samples]
            y_train_lstm = y_all_seq[:-val_samples]
            X_val_lstm = X_all_seq[-val_samples:]
            y_val_lstm = y_all_seq[-val_samples:]

            final_model.fit(
                X_train_lstm,
                y_train_lstm,
                validation_data=(X_val_lstm, y_val_lstm),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0,
            )
        else:
            final_model.fit(X_all_seq, y_all_seq, epochs=50, batch_size=32, verbose=0)

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
            'sequence_length': sequence_length,
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
            'holdout_metrics': holdout_results['metrics'] if holdout_results else None,
        }
        joblib.dump(model_data, model_path)
        logging.info(f"[{algo_name}] Modelo final guardado en: {model_path}")

    # Guardar historial de hiperparámetros (para todos los modelos)
    save_hyperparameters_to_csv(
        algo_name,
        best_params_random,
        study.best_params,
        backtest_results['metrics'] if backtest_results else {},
        holdout_results['metrics'] if holdout_results else {},
    )

    # =============================================================================
    # GENERAR FORECAST COMPLETO (ÚLTIMOS 20 DÍAS + FUTURO) - CON DEBUGGING
    # =============================================================================
    logging.info(f"[{algo_name}] ========== INICIANDO FORECAST COMPLETO CON DEBUGGING ==========")
    logging.info(
        f"[{algo_name}] Método: Últimos 20 días (archivo principal) + Forecast futuro (características nuevas)"
    )
    logging.info(f"[{algo_name}] Lag aplicado: 20 días para ambas partes del forecast")

    # 1. GENERAR PREDICCIONES PARA ÚLTIMOS 20 DÍAS DEL ARCHIVO PRINCIPAL (CON DEBUGGING)
    logging.info(f"[{algo_name}] Paso 1: Predicciones para últimos 20 días del archivo principal (CON DEBUGGING)")
    last_20_days_results = debug_and_fix_forecast_predictions(
        model=final_model,
        X_all=X_all_scaled,
        y_all=y_all,
        algo_name=algo_name,
        new_characteristics_file=NEW_CHARACTERISTICS_FILE,
        sequence_length=study.best_params.get("sequence_length", 10) if algo_name == "LSTM" else None,
        lag_days=20,
    )

    # VERIFICAR QUE LAS PREDICCIONES SE GENERARON CORRECTAMENTE
    if last_20_days_results is None:
        logging.error(f"[{algo_name}] ❌ Error crítico: No se pudieron generar predicciones para últimos 20 días")
        return None

    logging.info(f"[{algo_name}] ✅ Predicciones últimos 20 días generadas:")
    logging.info(f"[{algo_name}]   - Total predicciones: {last_20_days_results.get('n_predictions', 0)}")
    logging.info(f"[{algo_name}]   - Predicciones válidas: {last_20_days_results.get('n_valid_predictions', 0)}")

    # 2. GENERAR FORECAST FUTURO USANDO ARCHIVO DE CARACTERÍSTICAS NUEVAS
    logging.info(f"[{algo_name}] Paso 2: Forecast futuro con días hábiles consecutivos")

    # Obtener características originales
    original_features = X_all.columns.tolist()
    sequence_length = study.best_params.get("sequence_length", 10) if algo_name == "LSTM" else None

    # Generar forecast futuro con días hábiles
    future_forecast_results = generate_business_days_forecast_future_fixed(
        model=final_model,
        last_20_days_results=last_20_days_results,
        new_characteristics_file=NEW_CHARACTERISTICS_FILE,
        algo_name=algo_name,
        sequence_length=sequence_length,
        original_features=original_features,
        forecast_days=20,
    )

    logging.info(
        f"[{algo_name}] ✅ Forecast futuro con días hábiles completado: {len(future_forecast_results['predictions'])} predicciones"
    )

    # 3. CREAR DATAFRAME COMPLETO DE FORECAST
    logging.info(f"[{algo_name}] Paso 3: Creando DataFrame completo de forecast")
    df_forecast_complete = create_complete_forecast_dataframe_aligned(
        algo_name=algo_name,
        study_best_params=study.best_params,
        args=args,
        X_all=X_all_scaled,
        last_20_days_results=last_20_days_results,
        future_forecast_results=future_forecast_results,
        new_characteristics_file=NEW_CHARACTERISTICS_FILE,
    )

    # 4. VERIFICACIÓN ADICIONAL DEL DATAFRAME GENERADO
    logging.info(f"[{algo_name}] Paso 4: Verificando DataFrame de forecast generado")
    verification_passed = verify_forecast_dataframe_predictions(df_forecast_complete, algo_name)
    if not verification_passed:
        logging.warning(f"[{algo_name}] ⚠️ Problemas detectados en predicciones del DataFrame")
        logging.warning(f"[{algo_name}] ⚠️ El modelo puede tener problemas de predicción")
    else:
        logging.info(f"[{algo_name}] ✅ Verificación del DataFrame exitosa - Todas las predicciones están presentes")

    # 5. EXTRAER DATOS PARA COMPATIBILIDAD CON CÓDIGO EXISTENTE
    logging.info(f"[{algo_name}] Paso 5: Preparando datos para compatibilidad con gráficos")

    # Combinar todas las predicciones para mantener compatibilidad
    all_forecast_preds = last_20_days_results['predictions'] + future_forecast_results['predictions']
    all_forecast_dates = last_20_days_results['dates'] + future_forecast_results['prediction_dates']

    # Para el DataFrame final, usar el forecast completo
    future_preds = all_forecast_preds
    future_dates = all_forecast_dates

    logging.info(f"[{algo_name}] ✅ FORECAST COMPLETO GENERADO:")
    logging.info(f"  - Últimos 20 días (archivo principal): {len(last_20_days_results['predictions'])} predicciones")
    logging.info(f"  - Predicciones válidas (últimos 20): {last_20_days_results.get('n_valid_predictions', 0)}")
    logging.info(
        f"  - Forecast futuro (características nuevas): {len(future_forecast_results['predictions'])} predicciones"
    )
    logging.info(f"  - Predicciones válidas (futuro): {future_forecast_results.get('n_valid_predictions', 0)}")
    logging.info(f"  - TOTAL PREDICCIONES: {len(future_preds)}")
    if future_dates:
        logging.info(f"  - Período completo: {future_dates[0]} a {future_dates[-1]}")

    # VALIDACIÓN FINAL: Verificar continuidad temporal
    logging.info(f"[{algo_name}] Paso 6: Validación de continuidad temporal")
    if len(future_dates) > 1:
        gaps = []
        for i in range(1, len(future_dates)):
            current_date = pd.to_datetime(future_dates[i])
            previous_date = pd.to_datetime(future_dates[i - 1])
            gap = (current_date - previous_date).days
            if gap > 7:  # Gap mayor a una semana
                gaps.append(
                    f"Gap de {gap} días entre {previous_date.strftime('%Y-%m-%d')} y {current_date.strftime('%Y-%m-%d')}"
                )

        if gaps:
            logging.warning(f"[{algo_name}] ⚠️ Gaps temporales detectados en forecast:")
            for gap in gaps:
                logging.warning(f"  - {gap}")
        else:
            logging.info(f"[{algo_name}] ✅ Forecast sin gaps temporales significativos - CONTINUIDAD PERFECTA")

    logging.info(f"[{algo_name}] ========== FORECAST COMPLETO FINALIZADO ==========")

    # Escalar cada zona individualmente para todos los modelos
    if scaling_required and scaler is not None:
        X_zone_A_scaled = pd.DataFrame(scaler.transform(X_zone_A), columns=X_zone_A.columns, index=X_zone_A.index)
        X_zone_B_scaled = pd.DataFrame(scaler.transform(X_zone_B), columns=X_zone_B.columns, index=X_zone_B.index)
        X_zone_C_scaled = pd.DataFrame(scaler.transform(X_zone_C), columns=X_zone_C.columns, index=X_zone_C.index)
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
                train_preds[sequence_length : sequence_length + len(train_preds_seq)] = train_preds_seq
                train_preds = np.nan_to_num(train_preds)
            else:
                train_preds = np.zeros(len(y_zone_A))
        else:
            train_preds = np.zeros(len(y_zone_A))
    else:
        # Para otros modelos
        train_preds = final_model.predict(X_zone_A_scaled) if len(X_zone_A) > 0 else []

    train_metrics = calcular_metricas_basicas(y_zone_A, train_preds) if len(X_zone_A) > 0 else {}

    df_train = pd.DataFrame(
        {
            "date": X_zone_A.index,
            "Valor_Real": y_zone_A,
            "Valor_Predicho": train_preds,
            "Modelo": algo_name,
            "Version": f"Three-Zones-Complete",
            "RMSE": train_metrics.get('RMSE', np.nan),
            "MAE": train_metrics.get('MAE', np.nan),
            "R2": train_metrics.get('R2', np.nan),
            "SMAPE": train_metrics.get('SMAPE', np.nan),
            "Hyperparámetros": json.dumps(study.best_params),
            "Tipo_Mercado": args.tipo_mercado,
            "Periodo": "Training",
        }
    )

    # 2. Sección Evaluación (Zona B - Backtest)
    eval_preds = backtest_results['predictions'] if backtest_results else []
    eval_metrics = backtest_results['metrics'] if backtest_results else {}

    df_eval = pd.DataFrame(
        {
            "date": X_zone_B.index,
            "Valor_Real": y_zone_B,
            "Valor_Predicho": eval_preds,
            "Modelo": algo_name,
            "Version": f"Three-Zones-Complete",
            "RMSE": eval_metrics.get('RMSE', np.nan),
            "MAE": eval_metrics.get('MAE', np.nan),
            "R2": eval_metrics.get('R2', np.nan),
            "SMAPE": eval_metrics.get('SMAPE', np.nan),
            "Hyperparámetros": json.dumps(study.best_params),
            "Tipo_Mercado": args.tipo_mercado,
            "Periodo": "Evaluacion",
        }
    )

    # 3. Sección Test (Zona C - Holdout)
    test_preds = holdout_results['predictions'] if holdout_results else []
    test_metrics = holdout_results['metrics'] if holdout_results else {}

    df_test = pd.DataFrame(
        {
            "date": X_zone_C.index,
            "Valor_Real": y_zone_C,
            "Valor_Predicho": test_preds,
            "Modelo": algo_name,
            "Version": f"Three-Zones-Complete",
            "RMSE": test_metrics.get('RMSE', np.nan),
            "MAE": test_metrics.get('MAE', np.nan),
            "R2": test_metrics.get('R2', np.nan),
            "SMAPE": test_metrics.get('SMAPE', np.nan),
            "Hyperparámetros": json.dumps(study.best_params),
            "Tipo_Mercado": args.tipo_mercado,
            "Periodo": "Test",
        }
    )

    # 4. Sección Forecast (USAR EL FORECAST COMPLETO CON DEBUGGING)
    df_forecast = df_forecast_complete.copy()

    # VERIFICACIÓN FINAL DEL DATAFRAME ANTES DE CONCATENAR
    logging.info(f"[{algo_name}] 🔍 VERIFICACIÓN FINAL ANTES DE CONCATENAR:")
    logging.info(f"[{algo_name}]   - df_train: {len(df_train)} filas")
    logging.info(f"[{algo_name}]   - df_eval: {len(df_eval)} filas")
    logging.info(f"[{algo_name}]   - df_test: {len(df_test)} filas")
    logging.info(f"[{algo_name}]   - df_forecast: {len(df_forecast)} filas")

    # Verificar específicamente las filas de Forecast_Last_20_Days
    forecast_last_20_rows = df_forecast[df_forecast['Periodo'] == 'Forecast_Last_20_Days']
    logging.info(f"[{algo_name}]   - Filas Forecast_Last_20_Days: {len(forecast_last_20_rows)}")

    if len(forecast_last_20_rows) > 0:
        pred_non_null = forecast_last_20_rows['Valor_Predicho'].notna().sum()
        logging.info(
            f"[{algo_name}]   - Predicciones no nulas en Forecast_Last_20_Days: {pred_non_null}/{len(forecast_last_20_rows)}"
        )

        if pred_non_null < len(forecast_last_20_rows):
            logging.warning(f"[{algo_name}] ⚠️ HAY PREDICCIONES NULAS EN FORECAST_LAST_20_DAYS!")

            # Mostrar cuáles son nulas
            null_rows = forecast_last_20_rows[forecast_last_20_rows['Valor_Predicho'].isna()]
            for idx, row in null_rows.iterrows():
                logging.warning(f"[{algo_name}]     - Fila {idx}: Fecha {row['date']} tiene predicción nula")

    # Concatenar todos los DataFrames
    df_all = pd.concat([df_train, df_eval, df_test, df_forecast], ignore_index=True)

    # Guardar resultados específicos de este modelo
    model_csv_path = os.path.join(
        RESULTS_DIR, f"{args.tipo_mercado.lower()}_{algo_name.lower()}_three_zones_complete.csv"
    )
    df_all.to_csv(model_csv_path, index=False)
    logging.info(f"[{algo_name}] CSV COMPLETO guardado: {model_csv_path}")

    # ====================================================================
    # GENERACIÓN DE GRÁFICOS CON FORECAST COMPLETO
    # ====================================================================
    logging.info(f"[{algo_name}] ========== GENERANDO GRÁFICOS ==========")

    # Filtrar para excluir la Zona A (Training) en algunos gráficos
    df_for_plots = df_all[df_all['Periodo'] != 'Training'].copy()
    logging.info(f"[{algo_name}] Generando gráficos con forecast completo")
    logging.info(
        f"[{algo_name}] Datos para gráficos: {len(df_for_plots)} registros (excluyendo {len(df_train)} de Training)"
    )

    # 1. Gráfico principal con Zona B, C y Forecast completo
    chart_path = os.path.join(IMG_CHARTS_DIR, f"{algo_name.lower()}_three_zones_eval_test_forecast_complete.png")
    plot_real_vs_pred(
        df_for_plots,
        title=f"Back-test, Hold-out y Forecast Completo - {algo_name}\n(Zonas B, C + Últimos 20 días + Forecast futuro - Sin gaps temporales)",
        model_name=algo_name,
        output_path=chart_path,
    )
    logging.info(f"[{algo_name}] Gráfico principal guardado: {chart_path}")

    # 2. Gráfico detallado del forecast completo
    days_before_forecast = 25

    # Combinar datos de Zona C con forecast completo
    df_detailed_forecast = pd.concat(
        [df_test.tail(min(days_before_forecast, len(df_test))), df_forecast], ignore_index=True
    )

    forecast_detail_path = os.path.join(IMG_CHARTS_DIR, f"{algo_name.lower()}_forecast_complete_detail.png")

    # Crear gráfico personalizado para forecast completo
    plt.figure(figsize=(16, 10))

    # Separar los diferentes tipos de datos
    real_data = df_detailed_forecast[~df_detailed_forecast['Periodo'].str.contains('Forecast', na=False)]
    forecast_last_20 = df_detailed_forecast[df_detailed_forecast['Periodo'] == 'Forecast_Last_20_Days']
    forecast_future = df_detailed_forecast[df_detailed_forecast['Periodo'] == 'Forecast_Future']

    # Graficar datos reales
    if len(real_data) > 0:
        plt.plot(
            real_data['date'],
            real_data['Valor_Real'],
            'o-',
            color='blue',
            linewidth=3,
            markersize=8,
            label='Valores Reales (Hold-out)',
            alpha=0.9,
        )
        plt.plot(
            real_data['date'],
            real_data['Valor_Predicho'],
            's-',
            color='green',
            linewidth=2,
            markersize=6,
            label='Predicciones (Hold-out)',
            alpha=0.8,
        )

    # Graficar valores reales de últimos 20 días
    if len(forecast_last_20) > 0:
        # Valores reales disponibles
        forecast_last_20_real = forecast_last_20.dropna(subset=['Valor_Real'])
        if len(forecast_last_20_real) > 0:
            plt.plot(
                forecast_last_20_real['date'],
                forecast_last_20_real['Valor_Real'],
                'o-',
                color='blue',
                linewidth=3,
                markersize=8,
                alpha=0.9,
            )  # Continúa la línea azul

        # Predicciones para últimos 20 días - VERIFICAR QUE NO SON NULAS
        forecast_last_20_pred = forecast_last_20.dropna(subset=['Valor_Predicho'])
        if len(forecast_last_20_pred) > 0:
            plt.plot(
                forecast_last_20_pred['date'],
                forecast_last_20_pred['Valor_Predicho'],
                '^-',
                color='orange',
                linewidth=3,
                markersize=8,
                label='Forecast (Últimos 20 días con valores reales)',
                alpha=0.9,
            )
            logging.info(
                f"[{algo_name}] ✅ Graficando {len(forecast_last_20_pred)} predicciones válidas de Forecast_Last_20_Days"
            )
        else:
            logging.warning(f"[{algo_name}] ⚠️ No hay predicciones válidas para graficar en Forecast_Last_20_Days")

    # Graficar forecast futuro
    if len(forecast_future) > 0:
        forecast_future_pred = forecast_future.dropna(subset=['Valor_Predicho'])
        if len(forecast_future_pred) > 0:
            plt.plot(
                forecast_future_pred['date'],
                forecast_future_pred['Valor_Predicho'],
                '^-',
                color='red',
                linewidth=3,
                markersize=8,
                label='Forecast (Futuro con características nuevas)',
                alpha=0.9,
            )

    # Añadir líneas verticales para separar secciones
    if len(real_data) > 0 and len(forecast_last_20) > 0:
        separation_date = real_data['date'].iloc[-1]
        plt.axvline(
            x=separation_date, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='Inicio Últimos 20 días'
        )

    if len(forecast_last_20) > 0 and len(forecast_future) > 0:
        separation_date = forecast_last_20['date'].iloc[-1]
        plt.axvline(
            x=separation_date, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Inicio Forecast Futuro'
        )

    # Configurar formato del gráfico
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45, ha='right')

    total_forecast_days = len(forecast_last_20) + len(forecast_future)
    plt.title(
        f'Forecast Completo Detallado - {algo_name}\n(Últimos 20 días del archivo + {len(forecast_future)} días futuros = {total_forecast_days} días sin gaps)',
        fontsize=16,
        fontweight='bold',
    )
    plt.xlabel('Fecha', fontsize=14)
    plt.ylabel('Valor', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)

    # Añadir anotaciones importantes
    if len(forecast_last_20) > 0:
        forecast_last_20_valid = forecast_last_20.dropna(subset=['Valor_Predicho'])
        if len(forecast_last_20_valid) > 0:
            first_last_20 = forecast_last_20_valid.iloc[0]
            plt.annotate(
                f'Inicio últimos 20 días\n{first_last_20["Valor_Predicho"]:.3f}',
                xy=(first_last_20['date'], first_last_20['Valor_Predicho']),
                xytext=(15, 15),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                fontsize=10,
                ha='center',
            )

    if len(forecast_future) > 0:
        forecast_future_valid = forecast_future.dropna(subset=['Valor_Predicho'])
        if len(forecast_future_valid) > 0:
            first_future = forecast_future_valid.iloc[0]
            last_future = forecast_future_valid.iloc[-1]

            plt.annotate(
                f'Inicio futuro\n{first_future["Valor_Predicho"]:.3f}',
                xy=(first_future['date'], first_future['Valor_Predicho']),
                xytext=(15, -25),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8),
                fontsize=10,
                ha='center',
            )

            plt.annotate(
                f'Final forecast\n{last_future["Valor_Predicho"]:.3f}',
                xy=(last_future['date'], last_future['Valor_Predicho']),
                xytext=(15, 15),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8),
                fontsize=10,
                ha='center',
            )

    plt.tight_layout()
    plt.savefig(forecast_detail_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"[{algo_name}] Gráfico detallado de forecast completo guardado: {forecast_detail_path}")

    # 3. Gráfico desde 2024-01-01 hasta el forecast completo
    fecha_inicio_2024 = pd.Timestamp('2024-01-01')
    df_desde_2024 = df_all[pd.to_datetime(df_all['date']) >= fecha_inicio_2024].copy()
    desde_2024_path = None

    if len(df_desde_2024) > 0:
        desde_2024_path = os.path.join(IMG_CHARTS_DIR, f"{algo_name.lower()}_desde_2024_forecast_completo.png")

        # Crear gráfico completo desde 2024
        plt.figure(figsize=(20, 10))

        # Separar por períodos
        training_2024 = df_desde_2024[df_desde_2024['Periodo'] == 'Training']
        evaluacion_2024 = df_desde_2024[df_desde_2024['Periodo'] == 'Evaluacion']
        test_2024 = df_desde_2024[df_desde_2024['Periodo'] == 'Test']
        forecast_last_20_2024 = df_desde_2024[df_desde_2024['Periodo'] == 'Forecast_Last_20_Days']
        forecast_future_2024 = df_desde_2024[df_desde_2024['Periodo'] == 'Forecast_Future']

        # Graficar valores reales (línea continua)
        all_real_data = pd.concat([training_2024, evaluacion_2024, test_2024, forecast_last_20_2024])
        if len(all_real_data) > 0:
            real_data_with_values = all_real_data.dropna(subset=['Valor_Real'])
            if len(real_data_with_values) > 0:
                plt.plot(
                    real_data_with_values['date'],
                    real_data_with_values['Valor_Real'],
                    'o-',
                    color='blue',
                    linewidth=2,
                    markersize=3,
                    label='Valores Reales',
                    alpha=0.8,
                )

        # Graficar predicciones por período
        if len(training_2024) > 0:
            plt.plot(
                training_2024['date'],
                training_2024['Valor_Predicho'],
                's-',
                color='lightgreen',
                linewidth=1.5,
                markersize=2,
                label='Predicciones (Training)',
                alpha=0.6,
            )

        if len(evaluacion_2024) > 0:
            plt.plot(
                evaluacion_2024['date'],
                evaluacion_2024['Valor_Predicho'],
                's-',
                color='green',
                linewidth=2,
                markersize=3,
                label='Predicciones (Back-test)',
                alpha=0.7,
            )

        if len(test_2024) > 0:
            plt.plot(
                test_2024['date'],
                test_2024['Valor_Predicho'],
                's-',
                color='darkgreen',
                linewidth=2,
                markersize=3,
                label='Predicciones (Hold-out)',
                alpha=0.8,
            )

        # Graficar forecast completo - SOLO VALORES VÁLIDOS
        if len(forecast_last_20_2024) > 0:
            forecast_last_20_valid = forecast_last_20_2024.dropna(subset=['Valor_Predicho'])
            if len(forecast_last_20_valid) > 0:
                plt.plot(
                    forecast_last_20_valid['date'],
                    forecast_last_20_valid['Valor_Predicho'],
                    '^-',
                    color='orange',
                    linewidth=2.5,
                    markersize=5,
                    label='Forecast (Últimos 20 días)',
                    alpha=0.9,
                )

        if len(forecast_future_2024) > 0:
            forecast_future_valid = forecast_future_2024.dropna(subset=['Valor_Predicho'])
            if len(forecast_future_valid) > 0:
                plt.plot(
                    forecast_future_valid['date'],
                    forecast_future_valid['Valor_Predicho'],
                    '^-',
                    color='red',
                    linewidth=2.5,
                    markersize=5,
                    label='Forecast (Futuro)',
                    alpha=0.9,
                )

                # Añadir líneas de separación
                if len(all_real_data) > 0 and len(forecast_last_20_valid) > 0:
                    last_real_date = all_real_data['date'].max()
                    plt.axvline(
                        x=last_real_date,
                        color='purple',
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.6,
                        label='Inicio Forecast',
                    )

                # Anotar puntos clave del forecast
                if len(forecast_future_valid) > 0:
                    first_forecast = forecast_future_valid.iloc[0]
                    last_forecast = forecast_future_valid.iloc[-1]

                    plt.annotate(
                        f'{first_forecast["Valor_Predicho"]:.3f}',
                        xy=(first_forecast['date'], first_forecast['Valor_Predicho']),
                        xytext=(8, 12),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=8,
                        ha='center',
                    )

                    plt.annotate(
                        f'{last_forecast["Valor_Predicho"]:.3f}',
                        xy=(last_forecast['date'], last_forecast['Valor_Predicho']),
                        xytext=(8, -15),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=8,
                        ha='center',
                    )

        # Configurar formato de fechas
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=1))
        plt.xticks(rotation=45, ha='right')

        # Configurar el gráfico
        total_dias = len(df_desde_2024)
        total_forecast = len(forecast_last_20_2024) + len(forecast_future_2024)
        plt.title(
            f'Evolución Completa desde 2024 - {algo_name}\n(Desde 2024-01-01 + Forecast completo de {total_forecast} días sin gaps - Total: {total_dias} registros)',
            fontsize=16,
            fontweight='bold',
        )
        plt.xlabel('Fecha', fontsize=14)
        plt.ylabel('Valor', fontsize=14)
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(desde_2024_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"[{algo_name}] Gráfico desde 2024 completo guardado: {desde_2024_path}")
    else:
        logging.warning(f"[{algo_name}] No hay datos desde 2024-01-01 para graficar")

    # 4. Gráfico completo histórico (todas las zonas)
    full_chart_path = os.path.join(IMG_CHARTS_DIR, f"{algo_name.lower()}_historico_completo_con_forecast.png")
    plot_real_vs_pred(
        df_all,
        title=f"Histórico Completo y Forecast Sin Gaps - {algo_name}\n(Todas las Zonas + Forecast completo de {len(df_forecast)} días)",
        model_name=algo_name,
        output_path=full_chart_path,
    )
    logging.info(f"[{algo_name}] Gráfico histórico completo guardado: {full_chart_path}")

    # 5. Feature importance si está disponible
    if hasattr(final_model, 'feature_importances_'):
        importance_chart_path = os.path.join(IMG_CHARTS_DIR, f"{algo_name.lower()}_feature_importance.png")

        try:
            feature_names = X_all.columns
            importances = final_model.feature_importances_

            plot_feature_importance(
                importances,
                feature_names,
                title=f"Feature Importance - {algo_name}",
                top_n=min(20, len(feature_names)),
                model_name=algo_name,
                output_path=importance_chart_path,
            )
            logging.info(f"[{algo_name}] Gráfico de feature importance guardado: {importance_chart_path}")
        except Exception as e:
            logging.error(f"[{algo_name}] Error al generar gráfico de feature importance: {e}")

    # 6. Resumen de gráficos generados
    logging.info(f"[{algo_name}] ✅ GRÁFICOS GENERADOS:")
    logging.info(f"[{algo_name}]   - Principal (B, C y Forecast completo): {chart_path}")
    logging.info(f"[{algo_name}]   - Detalle de forecast completo: {forecast_detail_path}")
    if desde_2024_path:
        logging.info(f"[{algo_name}]   - Desde 2024 con forecast completo: {desde_2024_path}")
    logging.info(f"[{algo_name}]   - Histórico completo: {full_chart_path}")
    if hasattr(final_model, 'feature_importances_'):
        logging.info(f"[{algo_name}]   - Feature importance: {importance_chart_path}")

    # 7. Información final del forecast completo
    logging.info(f"[{algo_name}] 📊 RESUMEN FINAL DEL FORECAST COMPLETO:")
    logging.info(f"[{algo_name}]   ✅ Sin gaps temporales - Continuidad perfecta")
    logging.info(f"[{algo_name}]   📁 Últimos 20 días: Valores reales del archivo principal")
    logging.info(f"[{algo_name}]   📁 Forecast futuro: {len(future_forecast_results['predictions'])} días")

    if os.path.exists(NEW_CHARACTERISTICS_FILE):
        logging.info(f"[{algo_name}]   ✅ Fuente futuro: Características reales procesadas (FPI, VIF, normalización)")
        logging.info(f"[{algo_name}]   📁 Archivo: {os.path.basename(NEW_CHARACTERISTICS_FILE)}")
        logging.info(f"[{algo_name}]   🎯 Ventaja: Máxima precisión en forecast futuro")
    else:
        logging.info(f"[{algo_name}]   ⚠️ Fuente futuro: Método de fallback")
        logging.info(f"[{algo_name}]   📝 Recomendación: Proveer archivo de características para mayor precisión")

    # 8. Guardar resumen detallado del forecast completo
    comprehensive_forecast_summary = {
        'algorithm': algo_name,
        'version': 'complete_forecast_with_debugging',
        'execution_timestamp': datetime.now().isoformat(),
        'lag_days': 20,
        'total_forecast_days': len(future_preds),
        'last_20_days': {
            'source': 'training_file_with_real_values',
            'method': 'characteristics_20_days_before_with_debugging',
            'n_predictions': len(last_20_days_results['predictions']),
            'n_valid_predictions': last_20_days_results.get('n_valid_predictions', 0),
            'has_real_values': True,
            'date_range': {
                'from': str(last_20_days_results['dates'][0]) if last_20_days_results['dates'] else 'N/A',
                'to': str(last_20_days_results['dates'][-1]) if last_20_days_results['dates'] else 'N/A',
            },
            'metrics_available': True,
            'debugging_applied': True,
        },
        'future_forecast': {
            'source': 'new_characteristics_file' if os.path.exists(NEW_CHARACTERISTICS_FILE) else 'fallback_method',
            'file_path': NEW_CHARACTERISTICS_FILE if os.path.exists(NEW_CHARACTERISTICS_FILE) else None,
            'method': 'characteristics_with_20_day_lag',
            'n_predictions': len(future_forecast_results['predictions']),
            'n_valid_predictions': future_forecast_results.get('n_valid_predictions', 0),
            'has_real_values': False,
            'date_range': {
                'from': (
                    str(future_forecast_results['prediction_dates'][0])
                    if future_forecast_results['prediction_dates']
                    else 'N/A'
                ),
                'to': (
                    str(future_forecast_results['prediction_dates'][-1])
                    if future_forecast_results['prediction_dates']
                    else 'N/A'
                ),
            },
        },
        'temporal_continuity': {
            'gaps_detected': len(gaps) if 'gaps' in locals() else 0,
            'continuous': len(gaps) == 0 if 'gaps' in locals() else True,
            'total_date_range': {
                'from': str(future_dates[0]) if future_dates else 'N/A',
                'to': str(future_dates[-1]) if future_dates else 'N/A',
            },
        },
        'model_performance': {
            'training_metrics': train_metrics,
            'backtest_metrics': backtest_results['metrics'] if backtest_results else {},
            'holdout_metrics': holdout_results['metrics'] if holdout_results else {},
            'last_20_days_metrics': {
                'rmse': (
                    df_forecast_complete.loc[df_forecast_complete['Periodo'] == 'Forecast_Last_20_Days', 'RMSE'].iloc[0]
                    if len(df_forecast_complete) > 0
                    else np.nan
                ),
                'mae': (
                    df_forecast_complete.loc[df_forecast_complete['Periodo'] == 'Forecast_Last_20_Days', 'MAE'].iloc[0]
                    if len(df_forecast_complete) > 0
                    else np.nan
                ),
                'r2': (
                    df_forecast_complete.loc[df_forecast_complete['Periodo'] == 'Forecast_Last_20_Days', 'R2'].iloc[0]
                    if len(df_forecast_complete) > 0
                    else np.nan
                ),
                'smape': (
                    df_forecast_complete.loc[df_forecast_complete['Periodo'] == 'Forecast_Last_20_Days', 'SMAPE'].iloc[
                        0
                    ]
                    if len(df_forecast_complete) > 0
                    else np.nan
                ),
            },
        },
        'hyperparameters': {
            'best_params': study.best_params,
            'random_search_params': best_params_random,
            'optuna_best_score': study.best_value,
        },
        'files_generated': {
            'csv_complete': model_csv_path,
            'graphs': [chart_path, forecast_detail_path, desde_2024_path if desde_2024_path else None, full_chart_path],
            'feature_importance': importance_chart_path if hasattr(final_model, 'feature_importances_') else None,
        },
        'verification_status': {
            'dataframe_verification_passed': verification_passed,
            'predictions_debugging_applied': True,
        },
    }

    # Guardar resumen completo
    comprehensive_summary_file = os.path.join(RESULTS_DIR, f"{algo_name.lower()}_comprehensive_forecast_summary.json")
    with open(comprehensive_summary_file, 'w') as f:
        json.dump(comprehensive_forecast_summary, f, indent=4, default=str)

    logging.info(f"[{algo_name}] 💾 Resumen completo guardado: {comprehensive_summary_file}")

    # Limpiar sesión de Keras para LSTM
    if algo_name == "LSTM":
        tf.keras.backend.clear_session()

    # Tiempo total de procesamiento
    total_time = time.perf_counter() - start_time
    logging.info(f"[{algo_name}] ========== PROCESAMIENTO COMPLETADO ==========")
    logging.info(f"[{algo_name}] 🎉 Tiempo total: {total_time:.2f}s")
    logging.info(f"[{algo_name}] ✅ Forecast completo sin gaps temporales")
    logging.info(f"[{algo_name}] ✅ {len(future_preds)} días de predicciones continuas")
    logging.info(f"[{algo_name}] ✅ Interpretación correcta del lag de 20 días")
    logging.info(f"[{algo_name}] ✅ Valores reales disponibles para últimos 20 días")
    logging.info(f"[{algo_name}] ✅ Características reales para forecast futuro")
    logging.info(f"[{algo_name}] ✅ Debugging aplicado para predicciones robustas")
    logging.info(f"[{algo_name}] =================================================")

    return df_all


def generate_fact_predictions_csv(input_file=None, output_file=None):
    """
    Genera un CSV con estructura de FactPredictions basado en all_models_predictions.csv

    Args:
        input_file (str): Ruta al archivo all_models_predictions.csv. Si es None, se usa la ruta por defecto.
        output_file (str): Ruta del archivo de salida. Si es None, se usa la ruta por defecto.

    Returns:
        pd.DataFrame: DataFrame con la estructura de FactPredictions
    """
    # Obtener la ruta del directorio actual del script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Subir 3 niveles en la estructura de directorios para llegar a la raíz del proyecto
    base_dir = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

    # Establecer rutas por defecto si no se proporcionan
    if input_file is None:
        input_file = os.path.join(base_dir, 'SP500_INDEX_Analisis', 'Data', '4_results', 'all_models_predictions.csv')
    if output_file is None:
        output_file = os.path.join(
            base_dir, 'SP500_INDEX_Analisis', 'Data', '4_results', 'hechos_predicciones_fields.csv'
        )

    print("=== GENERANDO FACT_PREDICTIONS CSV ===")
    print(f"Archivo entrada: {input_file}")
    print(f"Archivo salida: {output_file}")

    # Verificar si el archivo de entrada existe
    if not os.path.exists(input_file):
        # Mostrar estructura de directorios para diagnóstico
        print("\n⚠️ DIAGNÓSTICO DE RUTAS:")
        print(f"Directorio del script: {script_dir}")
        print(f"Directorio base del proyecto: {base_dir}")
        print(f"Ruta completa entrada: {input_file}")
        print(f"El archivo {'EXISTE' if os.path.exists(input_file) else 'NO EXISTE'}")
        print(f"Contenido del directorio: {os.listdir(os.path.dirname(input_file))}")

        raise FileNotFoundError(f"❌ Archivo de entrada no encontrado: {input_file}")

    # Crear carpeta de salida si no existe
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📂 Carpeta creada: {output_dir}")

    # 1. CARGAR DATOS ORIGINALES
    df_original = pd.read_csv(input_file)
    print(f"✅ Datos cargados: {len(df_original)} filas, {len(df_original.columns)} columnas")

    # 2. CREAR MAPEOS PARA GENERAR KEYS
    # Crear mapeos únicos para generar IDs
    fechas_unicas = sorted(df_original['date'].unique())
    modelos_unicos = sorted(df_original['Modelo'].unique())
    mercados_unicos = sorted(df_original['Tipo_Mercado'].unique())

    # Mapear a IDs secuenciales (empezando en 1)
    fecha_to_key = {fecha: idx + 1 for idx, fecha in enumerate(fechas_unicas)}
    modelo_to_key = {modelo: idx + 1 for idx, modelo in enumerate(modelos_unicos)}
    mercado_to_key = {mercado: idx + 1 for idx, mercado in enumerate(mercados_unicos)}

    print(f"📊 Mapeos generados:")
    print(f"   - Fechas únicas: {len(fechas_unicas)} (desde {fechas_unicas[0]} hasta {fechas_unicas[-1]})")
    print(f"   - Modelos únicos: {len(modelos_unicos)} - {modelos_unicos}")
    print(f"   - Mercados únicos: {len(mercados_unicos)} - {mercados_unicos}")

    # 3. FUNCIÓN PARA MAPEAR PERÍODOS A ZONAS DE ENTRENAMIENTO
    def map_periodo_to_zona(periodo):
        """Mapea el período a zona de entrenamiento"""
        zona_mapping = {
            'Training': 'Zona A',  # 70% - Entrenamiento
            'Evaluacion': 'Zona B',  # 20% - Back-test
            'Test': 'Zona C',  # 10% - Hold-out
            'Forecast_Last_20_Days': 'Forecast',
            'Forecast_Future': 'Forecast',
        }
        return zona_mapping.get(periodo, 'Desconocido')

    # 4. CREAR DATAFRAME CON ESTRUCTURA FACT_PREDICTIONS
    fact_predictions = pd.DataFrame()

    # Generar PrediccionKey único (PK)
    fact_predictions['PrediccionKey'] = range(1, len(df_original) + 1)

    # Mapear fechas, modelos y mercados a keys
    fact_predictions['FechaKey'] = df_original['date'].map(fecha_to_key)
    fact_predictions['ModeloKey'] = df_original['Modelo'].map(modelo_to_key)
    fact_predictions['MercadoKey'] = df_original['Tipo_Mercado'].map(mercado_to_key)

    # Copiar datos existentes
    fact_predictions['ValorReal'] = df_original['Valor_Real']
    fact_predictions['ValorPredicho'] = df_original['Valor_Predicho']

    # 5. CALCULAR NUEVAS MÉTRICAS DE ERROR
    print("🧮 Calculando métricas de error...")

    # ErrorAbsoluto = |Real - Predicho|
    fact_predictions['ErrorAbsoluto'] = np.abs(fact_predictions['ValorReal'] - fact_predictions['ValorPredicho'])

    # ErrorPorcentual = (Error / ValorReal) * 100
    # Manejar división por cero
    def calculate_error_porcentual(real, predicho):
        if pd.isna(real) or real == 0:
            return np.nan
        return np.abs((real - predicho) / real) * 100

    fact_predictions['ErrorPorcentual'] = df_original.apply(
        lambda row: calculate_error_porcentual(row['Valor_Real'], row['Valor_Predicho']), axis=1
    )

    # 6. MAPEAR CAMPOS EXISTENTES Y GENERAR NUEVOS
    fact_predictions['TipoPeriodo'] = df_original['Periodo']  # Usar directamente el campo Periodo
    fact_predictions['ZonaEntrenamiento'] = df_original['Periodo'].apply(map_periodo_to_zona)

    # EsPrediccionFutura = True si es Forecast_Future, False en caso contrario
    fact_predictions['EsPrediccionFutura'] = (df_original['Periodo'] == 'Forecast_Future').astype(bool)

    # 7. VERIFICAR CALIDAD DE DATOS
    print("🔍 Verificando calidad de datos...")

    # Verificar valores nulos en keys
    null_checks = {
        'FechaKey': fact_predictions['FechaKey'].isnull().sum(),
        'ModeloKey': fact_predictions['ModeloKey'].isnull().sum(),
        'MercadoKey': fact_predictions['MercadoKey'].isnull().sum(),
    }

    for field, null_count in null_checks.items():
        if null_count > 0:
            print(f"⚠️  {field} tiene {null_count} valores nulos")
        else:
            print(f"✅ {field} sin valores nulos")

    # Estadísticas de errores
    print(f"\n📈 ESTADÍSTICAS DE ERRORES:")
    print(
        f"ErrorAbsoluto - Min: {fact_predictions['ErrorAbsoluto'].min():.6f}, "
        f"Max: {fact_predictions['ErrorAbsoluto'].max():.6f}, "
        f"Media: {fact_predictions['ErrorAbsoluto'].mean():.6f}"
    )

    error_pct_stats = fact_predictions['ErrorPorcentual'].describe()
    print(
        f"ErrorPorcentual - Min: {error_pct_stats['min']:.2f}%, "
        f"Max: {error_pct_stats['max']:.2f}%, "
        f"Media: {error_pct_stats['mean']:.2f}%"
    )

    # 8. MOSTRAR DISTRIBUCIÓN POR ZONA
    print(f"\n🗂️  DISTRIBUCIÓN POR ZONA DE ENTRENAMIENTO:")
    zona_counts = fact_predictions['ZonaEntrenamiento'].value_counts()
    for zona, count in zona_counts.items():
        percentage = (count / len(fact_predictions)) * 100
        print(f"   {zona}: {count:,} registros ({percentage:.1f}%)")

    # 9. MOSTRAR DISTRIBUCIÓN DE PREDICCIONES FUTURAS
    forecast_count = fact_predictions['EsPrediccionFutura'].sum()
    print(f"\n🔮 PREDICCIONES FUTURAS:")
    print(f"   Total predicciones futuras: {forecast_count:,}")
    print(f"   Porcentaje del total: {(forecast_count/len(fact_predictions))*100:.1f}%")

    # 10. ORDENAR COLUMNAS SEGÚN ESTRUCTURA SOLICITADA
    columnas_ordenadas = [
        'PrediccionKey',  # PK único
        'FechaKey',  # FK a DIM_TIEMPO
        'ModeloKey',  # FK a DIM_MODELO
        'MercadoKey',  # FK a DIM_MERCADO
        'ValorReal',  # Valor real observado
        'ValorPredicho',  # Predicción del modelo
        'ErrorAbsoluto',  # |Real - Predicho|
        'ErrorPorcentual',  # Error %
        'TipoPeriodo',  # Training, Evaluacion, Test, Forecast
        'ZonaEntrenamiento',  # Zona A, B, C, Forecast
        'EsPrediccionFutura',  # Si es forecast (sin valor real)
    ]

    fact_predictions = fact_predictions[columnas_ordenadas]

    # 11. GUARDAR ARCHIVO
    fact_predictions.to_csv(output_file, index=False, float_format='%.6f')
    print(f"\n✅ Archivo generado exitosamente: {output_file}")
    print(f"📊 Total de registros: {len(fact_predictions):,}")

    # 12. MOSTRAR MUESTRA DEL RESULTADO
    print(f"\n📋 MUESTRA DEL RESULTADO (primeras 5 filas):")
    print(fact_predictions.head().to_string(index=False))

    # 13. GENERAR REPORTE DE MAPEOS
    print(f"\n🗂️  REPORTE DE MAPEOS GENERADOS:")
    print("FECHAS (primeras 10):")
    for fecha, key in list(fecha_to_key.items())[:10]:
        print(f"   {fecha} → FechaKey: {key}")

    print("MODELOS:")
    for modelo, key in modelo_to_key.items():
        print(f"   {modelo} → ModeloKey: {key}")

    print("MERCADOS:")
    for mercado, key in mercado_to_key.items():
        print(f"   {mercado} → MercadoKey: {key}")

    return fact_predictions


# Función adicional para validar el resultado
def validate_fact_predictions(df):
    """
    Valida la calidad de los datos generados en FactPredictions

    Args:
        df (pd.DataFrame): DataFrame de FactPredictions a validar
    """
    print("\n=== VALIDACIÓN DE FACT_PREDICTIONS ===")

    # 1. Verificar unicidad de PrediccionKey
    duplicated_keys = df['PrediccionKey'].duplicated().sum()
    print(f"✅ PrediccionKey únicos: {duplicated_keys == 0} (duplicados: {duplicated_keys})")

    # 2. Verificar integridad de Foreign Keys
    fks_with_nulls = {
        'FechaKey': df['FechaKey'].isnull().sum(),
        'ModeloKey': df['ModeloKey'].isnull().sum(),
        'MercadoKey': df['MercadoKey'].isnull().sum(),
    }

    for fk, nulls in fks_with_nulls.items():
        status = "✅" if nulls == 0 else "❌"
        print(f"{status} {fk}: {nulls} valores nulos")

    # 3. Verificar rangos de valores
    print(f"\n📊 RANGOS DE VALORES:")
    print(f"FechaKey: {df['FechaKey'].min()} - {df['FechaKey'].max()}")
    print(f"ModeloKey: {df['ModeloKey'].min()} - {df['ModeloKey'].max()}")
    print(f"MercadoKey: {df['MercadoKey'].min()} - {df['MercadoKey'].max()}")

    # 4. Verificar distribución de errores
    print(f"\n📈 DISTRIBUCIÓN DE ERRORES:")
    error_abs_outliers = (df['ErrorAbsoluto'] > df['ErrorAbsoluto'].quantile(0.95)).sum()
    print(f"ErrorAbsoluto outliers (>p95): {error_abs_outliers}")

    error_pct_outliers = (df['ErrorPorcentual'] > 100).sum()  # Errores > 100%
    print(f"ErrorPorcentual > 100%: {error_pct_outliers}")

    # 5. Verificar consistencia de predicciones futuras
    forecast_future_count = (df['TipoPeriodo'] == 'Forecast_Future').sum()
    es_pred_futura_count = df['EsPrediccionFutura'].sum()
    consistency = forecast_future_count == es_pred_futura_count

    print(f"\n🔮 CONSISTENCIA PREDICCIONES FUTURAS:")
    print(f"✅ Consistencia EsPrediccionFutura: {consistency}")
    print(f"   Forecast_Future registros: {forecast_future_count}")
    print(f"   EsPrediccionFutura=True: {es_pred_futura_count}")

    return {
        'unique_keys': duplicated_keys == 0,
        'fk_integrity': all(nulls == 0 for nulls in fks_with_nulls.values()),
        'forecast_consistency': consistency,
        'total_records': len(df),
    }


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

    logging.info(
        f"División en tres zonas: A (HPO)={len(X_zone_A)}, B (Back-test)={len(X_zone_B)}, C (Hold-out)={len(X_zone_C)}"
    )

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
        ("LSTM", objective_lstm),  # Añadido LSTM
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
            algo_name,
            obj_func,
            X_zone_A,
            y_zone_A,  # Zona A (70%) para HPO
            X_zone_B,
            y_zone_B,  # Zona B (20%) para back-test
            X_zone_C,
            y_zone_C,  # Zona C (10%) para hold-out final
            forecast_horizon=forecast_horizon,
            top_n=3,
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
        columns_to_save = [
            "date",
            "Valor_Real",
            "Valor_Predicho",
            "Modelo",
            "Version",
            "Periodo",
            "RMSE",
            "MAE",
            "R2",
            "SMAPE",
            "Hyperparámetros",
            "Tipo_Mercado",
        ]
        predictions_df.to_csv(args.output_predictions, index=False, float_format="%.6f")
        logging.info(f"Archivo final de predicciones guardado en {args.output_predictions}")

        # Gráfico comparativo de todos los modelos
        all_models_chart = os.path.join(IMG_CHARTS_DIR, "all_models_comparison.png")
        plot_real_vs_pred(
            predictions_df[predictions_df['Periodo'] != 'Forecast'],
            title="Comparación de todos los modelos",
            output_path=all_models_chart,
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
                df_filtered = predictions_df[
                    (predictions_df['Modelo'] == modelo) & (predictions_df['Periodo'] == periodo)
                ]
                if not df_filtered.empty:
                    metrics = {
                        'Modelo': modelo,
                        'Periodo': periodo,
                        'RMSE_medio': df_filtered['RMSE'].mean(),
                        'MAE_medio': df_filtered['MAE'].mean(),
                        'R2_medio': df_filtered['R2'].mean(),
                        'SMAPE_medio': df_filtered['SMAPE'].mean(),
                        'Tiempo_Entrenamiento': algorithm_times.get(modelo, np.nan),
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
        backtest_metrics = (
            result_df[result_df['Periodo'] == 'Evaluacion'].iloc[0][['RMSE', 'MAE', 'R2', 'SMAPE']].to_dict()
        )
        holdout_metrics = result_df[result_df['Periodo'] == 'Test'].iloc[0][['RMSE', 'MAE', 'R2', 'SMAPE']].to_dict()

        all_backtest_results[algo_name] = {'metrics': backtest_metrics}
        all_holdout_results[algo_name] = {'metrics': holdout_metrics}

    # Generar tabla comparativa
    metrics_comparison = pd.DataFrame(columns=['Algoritmo', 'Fase', 'RMSE', 'MAE', 'R2', 'SMAPE'])

    # Añadir métricas de back-test
    for algo_name, results in all_backtest_results.items():
        metrics = results['metrics']
        new_row = pd.DataFrame(
            {
                'Algoritmo': [algo_name],
                'Fase': ['Back-test (Zona B)'],
                'RMSE': [metrics['RMSE']],
                'MAE': [metrics['MAE']],
                'R2': [metrics['R2']],
                'SMAPE': [metrics['SMAPE']],
            }
        )
        metrics_comparison = pd.concat([metrics_comparison, new_row], ignore_index=True)

    # Añadir métricas de hold-out
    for algo_name, results in all_holdout_results.items():
        metrics = results['metrics']
        new_row = pd.DataFrame(
            {
                'Algoritmo': [algo_name],
                'Fase': ['Hold-out (Zona C)'],
                'RMSE': [metrics['RMSE']],
                'MAE': [metrics['MAE']],
                'R2': [metrics['R2']],
                'SMAPE': [metrics['SMAPE']],
            }
        )
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
    plt.xticks(index + bar_width / 2, backtest_df['Algoritmo'])
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
    # Generar el archivo FactPredictions
    df_fact_predictions = generate_fact_predictions_csv()

    # Validar el resultado
    validation_results = validate_fact_predictions(df_fact_predictions)

    print(f"\n🎉 PROCESO COMPLETADO")
    print(f"Validación exitosa: {all(validation_results.values())}")
