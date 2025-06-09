import os
import glob
import random
import numpy as np
import pandas as pd
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from feature_engine.selection import SelectByShuffling
import pandas_market_calendars as mcal

# Importar configuraciones centralizadas
from src.config.base import ProjectConfig

config = ProjectConfig.from_env()

PROJECT_ROOT = config.project_root
PROCESSED_DIR = config.processed_dir
TRAINING_DIR = config.training_dir
DATE_COL = config.date_col
CV_SPLITS = config.cv_splits
FORECAST_HORIZON_1MONTH = config.forecast_horizon_1month
FORECAST_HORIZON_3MONTHS = config.forecast_horizon_3months
CV_GAP_1MONTH = config.cv_gap_1month
CV_GAP_3MONTHS = config.cv_gap_3months
FPI_THRESHOLD = config.fpi_threshold
CATBOOST_PARAMS = config.catboost_params
SCORER = config.scorer
RANDOM_SEED = config.random_seed

# ------------------------------
# CONFIGURACIÓN LOCAL
# ------------------------------

# Directorio para logs detallados y visualizaciones
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# Directorios de entrada y salida (mantener compatibilidad con código anterior)
INPUT_DIR = PROCESSED_DIR
OUTPUT_DIR = TRAINING_DIR

# Gap para validación cruzada (por defecto 1MONTH, 20 días hábiles)
FORECAST_PERIOD = "1MONTH"  # Puede ser '1MONTH' o '3MONTHS'
GAP = CV_GAP_1MONTH if FORECAST_PERIOD == "1MONTH" else CV_GAP_3MONTHS

# Si quieres forzar el target por nombre, ponlo aquí; en caso contrario, se usará la última columna
TARGET_COL_NAME = None

# Control de escalado
APPLY_SCALING = False

# Ajusta este threshold para hacer la selección de features más o menos agresiva
# Cuanto más alto, mayor caída de performance se requiere para descartar una feature
THRESHOLD = FPI_THRESHOLD

# ------------------------------
# INICIALIZACIÓN
# ------------------------------

# Asegurar que los directorios existen
for directory in [OUTPUT_DIR, LOG_DIR, PLOTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directorio creado: {directory}")

# Configurar logging con timestamp en el nombre del archivo
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"fpi_selection_{timestamp}.log")

# Configuración de logging a archivo y consola
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Configuración adicional para capturar más información
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Fijar semillas para reproducibilidad
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Log de inicio con información de configuración
logging.info("=" * 80)
logging.info("INICIANDO PROCESO DE SELECCIÓN DE FEATURES POR FPI")
logging.info("=" * 80)
logging.info(f"Timestamp: {timestamp}")
logging.info(f"Directorio de entrada: {INPUT_DIR}")
logging.info(f"Directorio de salida: {OUTPUT_DIR}")
logging.info(f"Directorio de logs: {LOG_DIR}")
logging.info(f"Archivo de log: {log_file}")
logging.info(f"Período de forecasting: {FORECAST_PERIOD}")
logging.info(f"Gap para CV: {GAP} días")
logging.info(f"Número de splits CV: {CV_SPLITS}")
logging.info(f"Threshold FPI: {THRESHOLD}")
logging.info(f"Aplicar escalado: {APPLY_SCALING}")
logging.info(f"Semilla aleatoria: {RANDOM_SEED}")
logging.info(f"Parámetros CatBoost: {CATBOOST_PARAMS}")
logging.info("=" * 80)

# ------------------------------
# FUNCIONES AUXILIARES
# ------------------------------

def get_most_recent_file(directory, extension='.xlsx'):
    """
    Obtiene el archivo más reciente en un directorio con la extensión especificada.
    
    Args:
        directory (str): Ruta al directorio
        extension (str): Extensión del archivo incluyendo el punto
        
    Returns:
        str: Ruta completa al archivo más reciente
    """
    files = glob.glob(os.path.join(directory, f'*{extension}'))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def plot_cv_splits(X, tscv, output_path):
    """
    Visualiza los splits de validación cruzada temporal.
    
    Args:
        X (DataFrame): Features
        tscv (TimeSeriesSplit): Objeto de validación cruzada
        output_path (str): Ruta donde guardar el gráfico
    """
    fig, ax = plt.figure(figsize=(15, 5)), plt.gca()
    
    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Graficar índices de entrenamiento
        ax.scatter(train_idx, [i + 0.5] * len(train_idx), 
                  c='blue', marker='_', s=40, label='Train' if i == 0 else "")
        
        # Graficar índices de validación
        ax.scatter(val_idx, [i + 0.5] * len(val_idx), 
                  c='red', marker='_', s=40, label='Validation' if i == 0 else "")
        
        # Añadir textos informativos
        ax.text(X.shape[0] + 5, i + 0.5, f"Split {i+1}: {len(train_idx)} train, {len(val_idx)} val",
               va='center', ha='left')
    
    # Añadir leyenda y etiquetas
    ax.legend(loc='upper right')
    ax.set_xlabel('Índice de muestra')
    ax.set_yticks(range(1, CV_SPLITS + 1))
    ax.set_yticklabels([f'Split {i+1}' for i in range(CV_SPLITS)])
    ax.set_title(f'Validación Cruzada Temporal (CV_SPLITS={CV_SPLITS}, GAP={GAP})')
    
    # Guardar gráfico
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logging.info(f"Gráfico de CV splits guardado en: {output_path}")

def plot_performance_drift(features, drifts, selected, threshold, output_path):
    """
    Visualiza los performance drifts de las features.
    
    Args:
        features (list): Lista de nombres de features
        drifts (list): Lista de performance drifts
        selected (list): Lista de features seleccionadas
        threshold (float): Umbral de selección
        output_path (str): Ruta donde guardar el gráfico
    """
    # Asegurarnos que drifts sea una lista o array
    if isinstance(drifts, dict):
        # Si es un diccionario, convertirlo a lista manteniendo el orden de features
        drifts_list = [drifts.get(feature, 0) for feature in features]
    else:
        drifts_list = drifts
        
    # Crear DataFrame con los datos
    df = pd.DataFrame({'feature': features, 'drift': drifts_list})
    df['selected'] = df['feature'].isin(selected)
    df = df.sort_values('drift', ascending=False)
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, max(8, len(features)/5)))
    
    # Graficar barras con colores según selección
    colors = ['green' if sel else 'red' for sel in df['selected']]
    bars = ax.barh(df['feature'], df['drift'], color=colors)
    
    # Añadir línea de threshold
    ax.axvline(x=threshold, color='black', linestyle='--', 
              label=f'Threshold ({threshold:.4f})')
    
    # Añadir etiquetas
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label = f"{width:.4f}"
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               label, va='center', ha='left')
    
    # Configurar leyenda y etiquetas
    ax.legend()
    ax.set_xlabel('Performance Drift')
    ax.set_ylabel('Feature')
    ax.set_title('Performance Drift por Feature (FPI)')
    
    # Guardar gráfico
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logging.info(f"Gráfico de performance drift guardado en: {output_path}")


def select_features_fpi(X, y, cv_splits=CV_SPLITS, gap=GAP, threshold=THRESHOLD):
    """
    Realiza Feature Permutation Importance (FPI) usando CatBoostRegressor y SelectByShuffling.
    Usa un threshold para hacer la selección menos agresiva.
    
    La validación usa TimeSeriesSplit con gap explícito para evitar data leakage,
    especialmente importante cuando hay features rezagadas o acumuladas.
    
    Args:
        X (DataFrame): Features
        y (Series): Target
        cv_splits (int): Número de folds para cross-validation
        gap (int): Tamaño del gap en días entre train y valid (debe coincidir con horizonte)
        threshold (float): Valor límite para considerar una feature relevante
        
    Returns:
        tuple: (lista de nombres de features seleccionadas, array con drift scores)
    """
    start_time = time.time()
    logging.info("=" * 50)
    logging.info("[FPI] INICIANDO ANÁLISIS DE FEATURE PERMUTATION IMPORTANCE")
    logging.info("=" * 50)
    logging.info(f"[FPI] Dimensiones de datos - X: {X.shape}, y: {y.shape}")
    logging.info(f"[FPI] Rango de valores target - Min: {y.min():.6f}, Max: {y.max():.6f}, Mean: {y.mean():.6f}")
    logging.info(f"[FPI] Parámetros - CV splits: {cv_splits}, gap: {gap}, threshold: {threshold}")
    
    # Validación temporal con gap explícito igual al horizonte de predicción
    tscv = TimeSeriesSplit(n_splits=cv_splits, gap=gap)
    
    # Visualizar los splits de CV
    cv_plot_path = os.path.join(PLOTS_DIR, f"cv_splits_{timestamp}.png")
    plot_cv_splits(X, tscv, cv_plot_path)
    
    # Log información detallada de los splits
    logging.info("[FPI] Detalle de los splits de validación cruzada temporal:")
    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        train_start, train_end = min(train_idx), max(train_idx)
        val_start, val_end = min(val_idx), max(val_idx)
        train_size, val_size = len(train_idx), len(val_idx)
        
        logging.info(f"[FPI] Split {i+1}:")
        logging.info(f"  - Train: {train_size} muestras (índices {train_start} a {train_end})")
        logging.info(f"  - Validation: {val_size} muestras (índices {val_start} a {val_end})")
        logging.info(f"  - Gap efectivo: {val_start - train_end - 1} muestras")
    
    # Crear regressor con los parámetros definidos
    logging.info("[FPI] Configurando CatBoostRegressor con los siguientes parámetros:")
    for param, value in CATBOOST_PARAMS.items():
        logging.info(f"  - {param}: {value}")
    
    regressor = CatBoostRegressor(**CATBOOST_PARAMS)

    # Configurar el selector
    logging.info(f"[FPI] Configurando SelectByShuffling con threshold={threshold}")
    logging.info(f"[FPI] Scorer utilizado: {SCORER}")
    
    selector = SelectByShuffling(
        estimator=regressor,
        scoring=SCORER,
        cv=tscv,
        threshold=threshold
    )

    # Iniciar proceso de fitting
    logging.info("[FPI] Iniciando proceso de fit con SelectByShuffling...")
    fit_start_time = time.time()
    
    # Verificar por NaN o infinitos
    if X.isna().any().any() or np.isinf(X).any().any():
        logging.warning("[FPI] ⚠️ Detectados valores NaN o infinitos en X. Puede causar problemas.")
        
    try:
        selector.fit(X, y)
        fit_time = time.time() - fit_start_time
        logging.info(f"[FPI] Proceso de fit completado en {fit_time:.2f} segundos")
    except Exception as e:
        logging.error(f"[FPI] ❌ Error durante el proceso de fit: {e}")
        return [], []

    # Obtener resultados
    selected_features = selector.get_feature_names_out()
    performance_drifts = selector.performance_drifts_
    
    if hasattr(selector, 'baseline_score_'):
        logging.info(f"[FPI] Baseline score (sin permutación): {selector.baseline_score_:.6f}")
    
    logging.info(f"[FPI] Se calcularon drift scores para {len(X.columns)} columnas.")
    logging.info(f"[FPI] Features seleccionadas por FPI: {len(selected_features)} de {len(X.columns)} ({len(selected_features)/len(X.columns)*100:.1f}%)")

    # Verificar longitudes
    if len(performance_drifts) != len(X.columns):
        logging.error(f"[FPI] ❌ Mismatch entre número de columnas de X ({len(X.columns)}) y performance drifts ({len(performance_drifts)}). Abortando.")
        return [], []

    # Construir DataFrame con los drifts
    drift_df = pd.DataFrame(
        list(zip(X.columns, performance_drifts)),
        columns=['feature', 'performance_drift']
    ).sort_values('performance_drift', ascending=False)

    logging.info("[FPI] Top 10 features por importancia (performance drift):")
    for _, row in drift_df.head(10).iterrows():
        feature = row['feature']
        drift = row['performance_drift']
        status = "✅ SELECCIONADA" if feature in selected_features else "❌ DESCARTADA"
        if isinstance(drift, (int, float)):
            logging.info(f"  - {feature}: {drift:.6f} [{status}]")
        else:
            logging.info(f"  - {feature}: {drift} [{status}]")
    
    logging.info("[FPI] Bottom 10 features con menor importancia:")
    for _, row in drift_df.tail(10).iterrows():
        feature = row['feature']
        drift = row['performance_drift']
        status = "✅ SELECCIONADA" if feature in selected_features else "❌ DESCARTADA"
        if isinstance(drift, (int, float)):
            logging.info(f"  - {feature}: {drift:.6f} [{status}]")
        else:
            logging.info(f"  - {feature}: {drift} [{status}]")
    
    # Guardar DataFrame completo de drifts
    drift_csv_path = os.path.join(PLOTS_DIR, f"fpi_drifts_{timestamp}.csv")
    drift_df.to_csv(drift_csv_path, index=False)
    logging.info(f"[FPI] CSV con todos los drift scores guardado en: {drift_csv_path}")
    
    # Visualizar performance drifts
    drift_plot_path = os.path.join(PLOTS_DIR, f"performance_drift_{timestamp}.png")
    plot_performance_drift(X.columns, performance_drifts, selected_features, threshold, drift_plot_path)
    
    # Tiempo total
    total_time = time.time() - start_time
    logging.info(f"[FPI] Proceso FPI completado en {total_time:.2f} segundos")
    logging.info(f"[FPI] Número de features seleccionadas: {len(selected_features)}")
    logging.info("=" * 50)

    return list(selected_features), performance_drifts

# ------------------------------
# FUNCIÓN PRINCIPAL
# ------------------------------

def main():
    """
    Función principal que ejecuta la selección de features FPI.
    
    Proceso:
    1. Carga el archivo más reciente
    2. Preprocesa los datos (fechas, filtrado por días hábiles)
    3. Aplica Feature Permutation Importance con validación temporal
    4. Guarda el dataset final con las features seleccionadas
    """
    main_start_time = time.time()
    
    # 1) Obtener el archivo más reciente en la carpeta de entrada
    input_file = get_most_recent_file(INPUT_DIR)
    if not input_file:
        logging.error(f"❌ No se encontraron archivos Excel en {INPUT_DIR}")
        return
    
    # Extraer el nombre base del archivo para generar el nombre de salida
    base_name = os.path.basename(input_file)
    file_name, file_ext = os.path.splitext(base_name)
    output_file = os.path.join(OUTPUT_DIR, f"{file_name}_FPI{file_ext}")
    
    logging.info("=" * 60)
    logging.info(f"INICIANDO SELECCIÓN DE FEATURES FPI")
    logging.info("=" * 60)
    logging.info(f"Usando el archivo más reciente: {input_file}")
    logging.info(f"La salida se guardará en: {output_file}")
    
    try:
        df_original = pd.read_excel(input_file)
        logging.info(f"Archivo '{input_file}' cargado con forma {df_original.shape}.")
        logging.info(f"Memoria utilizada: {df_original.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    except Exception as e:
        logging.error(f"❌ Error al cargar '{input_file}': {e}")
        return

    logging.info(f"Columnas del DataFrame original: {list(df_original.columns)}")
    logging.info(f"Tipos de datos: \n{df_original.dtypes}")
    logging.info(f"Primeras 3 filas (head) del DataFrame original:\n{df_original.head(3)}")
    logging.info(f"Últimas 3 filas (tail) del DataFrame original:\n{df_original.tail(3)}")
    
    # Estadísticas básicas del DataFrame original
    logging.info("Estadísticas básicas de algunas columnas numéricas:")
    num_cols = df_original.select_dtypes(include=[np.number]).columns[:5]  # Limitar a 5 columnas
    for col in num_cols:
        stats = df_original[col].describe()
        logging.info(f"  - {col}: min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    # 2) Verificar que existe la columna de fecha
    if DATE_COL not in df_original.columns:
        logging.error(f"❌ No se encontró la columna de fecha '{DATE_COL}' en el dataset.")
        return

    # 3) Copia para transformaciones
    df = df_original.copy()

    # 4) Convertir a datetime y normalizar (quitando componente de hora)
    logging.info(f"Convirtiendo columna '{DATE_COL}' a datetime...")
    original_dtype = df[DATE_COL].dtype
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce').dt.normalize()
    logging.info(f"Conversión de fechas: {original_dtype} → {df[DATE_COL].dtype}")

    # 5) Eliminar filas con fechas inválidas
    df_before_drop = df.shape[0]
    na_dates = df[DATE_COL].isna().sum()
    logging.info(f"Fechas inválidas (NaT) encontradas: {na_dates}")
    
    if na_dates > 0:
        df.dropna(subset=[DATE_COL], inplace=True)
        logging.info(f"Filas eliminadas por fecha inválida (NaT): {df_before_drop - df.shape[0]}")

    # 6) Rango de fechas tras limpieza
    data_min_date = df[DATE_COL].min()
    data_max_date = df[DATE_COL].max()
    num_unique_dates = df[DATE_COL].nunique()
    total_days = (data_max_date - data_min_date).days + 1
    
    logging.info(f"Rango de fechas después de limpiar: {data_min_date} a {data_max_date}")
    logging.info(f"Cantidad de fechas únicas: {num_unique_dates}")
    logging.info(f"Días calendario totales en el rango: {total_days}")
    logging.info(f"Cobertura de fechas: {num_unique_dates/total_days*100:.1f}% del rango total")

    # 7) Construir calendario de NYSE y filtrar solo días hábiles de trading
    logging.info("Construyendo calendario de días hábiles para NYSE...")
    nyse_cal = mcal.get_calendar('NYSE')
    schedule = nyse_cal.schedule(start_date=data_min_date, end_date=data_max_date)
    trading_days = mcal.date_range(schedule, frequency='1D').normalize()
    trading_days = trading_days.tz_localize(None)  # quita timezone

    num_trading_days = len(trading_days)
    logging.info(f"Días hábiles (NYSE) en el rango: {num_trading_days}")
    
    if num_trading_days > 0:
        logging.info(f"Primeros 5 días hábiles: {trading_days[:5].to_list()}")
        logging.info(f"Últimos 5 días hábiles: {trading_days[-5:].to_list()}")
        logging.info(f"Rango total días hábiles: {trading_days.min()} a {trading_days.max()}")
        
        # Verificar días faltantes
        dates_in_df = set(df[DATE_COL].dt.normalize())
        missing_trading_days = [d for d in trading_days if d not in dates_in_df]
        if missing_trading_days:
            logging.warning(f"⚠️ Faltan {len(missing_trading_days)} días hábiles en el dataset")
            logging.warning(f"Ejemplos de días hábiles faltantes: {missing_trading_days[:5]}")
    else:
        logging.warning("⚠️ No se generaron días hábiles para el rango especificado.")

    before_filter_shape = df.shape
    df = df[df[DATE_COL].isin(trading_days)]
    after_filter_shape = df.shape
    logging.info(f"Filas totales antes de filtrar por días hábiles: {before_filter_shape[0]}")
    logging.info(f"Filas totales después de filtrar por días hábiles: {after_filter_shape[0]}")
    logging.info(f"Filas eliminadas (no días hábiles): {before_filter_shape[0] - after_filter_shape[0]}")
    logging.info(f"Dataset tras filtrar días hábiles: {df.shape}")

    # 8) Verificar si hay datos suficientes para TimeSeriesSplit
    horizon = FORECAST_HORIZON_1MONTH if FORECAST_PERIOD == "1MONTH" else FORECAST_HORIZON_3MONTHS
    if df.shape[0] < CV_SPLITS * horizon:
        logging.warning(
            f"⚠️ El dataset tiene {df.shape[0]} filas. "
            f"Puede ser insuficiente para {CV_SPLITS} splits con horizonte={horizon} días."
        )
        logging.warning(f"Ideal: al menos {CV_SPLITS * horizon * 2} filas para CV robusto con ventana deslizante.")

    # 9) Determinar la columna target
    if TARGET_COL_NAME is not None and TARGET_COL_NAME in df.columns:
        target_col = TARGET_COL_NAME
        logging.info(f"Se forzó la columna target a '{target_col}' (definido en el script).")
    else:
        target_col = df.columns[-1]
        logging.info(f"Columna target tomada como la última columna: '{target_col}'")

    # Log de estadísticas del target
    target_stats = df[target_col].describe()
    logging.info(f"Estadísticas del target '{target_col}':")
    logging.info(f"  - count: {target_stats['count']}")
    logging.info(f"  - mean: {target_stats['mean']:.6f}")
    logging.info(f"  - std: {target_stats['std']:.6f}")
    logging.info(f"  - min: {target_stats['min']:.6f}")
    logging.info(f"  - 25%: {target_stats['25%']:.6f}")
    logging.info(f"  - 50%: {target_stats['50%']:.6f}")
    logging.info(f"  - 75%: {target_stats['75%']:.6f}")
    logging.info(f"  - max: {target_stats['max']:.6f}")
    
    # 10) Separar target y features
    y = df[target_col]
    X = df.drop(columns=[target_col], errors='ignore')
    logging.info(f"Separación de datos - X: {X.shape}, y: {y.shape}")

    # 11) Extraer columna de fecha de las features
    if DATE_COL in X.columns:
        logging.info(f"Extrayendo '{DATE_COL}' de las features para conservarla aparte.")
        date_data = X.pop(DATE_COL)
    else:
        date_data = None
        logging.warning(f"⚠️ No se encontró columna de fecha '{DATE_COL}' en las features.")

    # 12) Quedarnos solo con columnas numéricas
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        logging.warning(f"⚠️ Eliminando {len(non_numeric_cols)} columnas no numéricas: {non_numeric_cols}")
    
    X_numeric = X.select_dtypes(include=[np.number])
    logging.info(f"Número de columnas numéricas: {len(X_numeric.columns)} de {len(X.columns)} originales")
    logging.info(f"Primeras 10 columnas numéricas: {list(X_numeric.columns[:10])}")
    
    if len(X_numeric.columns) > 10:
        logging.info(f"... y {len(X_numeric.columns) - 10} más")

    if X_numeric.empty or X_numeric.shape[0] == 0:
        logging.error("❌ No hay datos numéricos disponibles para la selección de features. Abortando.")
        return

    # 13) Escalado (opcional)
    if APPLY_SCALING:
        logging.info("Aplicando StandardScaler a las features...")
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=X_numeric.columns)
        
        # Comparar estadísticas antes y después del escalado
        logging.info("Estadísticas antes/después del escalado (para primeras 3 columnas):")
        for col in X_numeric.columns[:3]:
            before_mean, before_std = X_numeric[col].mean(), X_numeric[col].std()
            after_mean, after_std = X_scaled[col].mean(), X_scaled[col].std()
            logging.info(f"  - {col}: antes [mean={before_mean:.4f}, std={before_std:.4f}], después [mean={after_mean:.4f}, std={after_std:.4f}]")
    else:
        X_scaled = X_numeric.copy()
        logging.info("No se aplicó escalado; se mantienen los valores originales.")

    # 14) Eliminar filas donde el target sea NaN (si las hay)
    if y.isna().any():
        num_nans = y.isna().sum()
        logging.warning(f"⚠️ El target contiene {num_nans} valores NaN ({num_nans/len(y)*100:.2f}%); se eliminarán esas filas.")
        valid_idx = ~y.isna()
        y = y[valid_idx]
        X_scaled = X_scaled[valid_idx]
        if date_data is not None:
            date_data = date_data[valid_idx]
        logging.info(f"Dimensiones tras eliminar NaNs - X: {X_scaled.shape}, y: {y.shape}")

    # Importante: forzar que todos tengan el mismo índice consecutivo 0..N-1
    y = y.reset_index(drop=True)
    X_scaled = X_scaled.reset_index(drop=True)
    if date_data is not None:
        date_data = date_data.reset_index(drop=True)
    logging.info("Índices reseteados para asegurar alineación 0..N-1")

    # 15) Logs antes de hacer la selección de features
    logging.info(f"Dimensión de X_scaled antes de FPI: {X_scaled.shape}")
    logging.info(f"Dimensión de y antes de FPI: {y.shape}")
    if date_data is not None:
        logging.info(f"Dimensión de date_data antes de FPI: {date_data.shape}")
    
    # Verificar valores extremos o problemas
    for col in X_scaled.columns[:5]:  # Primeras 5 columnas
        na_count = X_scaled[col].isna().sum()
        inf_count = np.isinf(X_scaled[col]).sum()
        if na_count > 0 or inf_count > 0:
            logging.warning(f"⚠️ Columna '{col}' tiene {na_count} NaNs y {inf_count} infinitos")
    
    logging.info(f"Últimas 3 filas de X_scaled:\n{X_scaled.tail(3)}")
    logging.info(f"Últimas 3 filas de y:\n{y.tail(3)}")
    if date_data is not None:
        logging.info(f"Últimas 3 filas de date_data:\n{date_data.tail(3)}")

    # 16) Selección de features con FPI - usando gap específico según forecast period
    logging.info("=" * 60)
    logging.info(f"INICIANDO SELECCIÓN DE FEATURES CON FPI")
    logging.info("=" * 60)
    logging.info(f"Número total de features antes de FPI: {X_scaled.shape[1]}")
    logging.info(f"Forecast period: {FORECAST_PERIOD}")
    gap = CV_GAP_1MONTH if FORECAST_PERIOD == "1MONTH" else CV_GAP_3MONTHS
    logging.info(f"Gap usado para TimeSeriesSplit: {gap} días")
    logging.info(f"Threshold FPI: {THRESHOLD}")
    
    # Verificar valores NaN o inf en el target
    if y.isna().any() or np.isinf(y).any().any():
        logging.error("❌ El target contiene valores NaN o infinitos. FPI fallará.")
        return
    
    fpi_start_time = time.time()
    selected_features, performance_drifts = select_features_fpi(
        X_scaled, y, cv_splits=CV_SPLITS, gap=gap, threshold=THRESHOLD
    )
    fpi_time = time.time() - fpi_start_time
    logging.info(f"Proceso FPI completado en {fpi_time:.2f} segundos")

    if not selected_features:
        logging.error("❌ No se logró seleccionar ninguna feature (posible error en FPI). Terminando.")
        return

    logging.info(f"Features finales seleccionadas ({len(selected_features)}/{X_scaled.shape[1]} = {len(selected_features)/X_scaled.shape[1]*100:.1f}%):")
    
    # Agrupar features por prefijos para mejor visualización
    feature_prefixes = {}
    for feature in selected_features:
        # Intentar extraer prefijo (suponiendo formato como prefix_featurename)
        parts = feature.split('_', 1)
        prefix = parts[0] if len(parts) > 1 else "otros"
        
        if prefix not in feature_prefixes:
            feature_prefixes[prefix] = []
        feature_prefixes[prefix].append(feature)
    
    # Mostrar features agrupadas por prefijo
    for prefix, features in feature_prefixes.items():
        logging.info(f"  - Grupo '{prefix}': {len(features)} features")
        logging.info(f"    {features[:5]}{'...' if len(features) > 5 else ''}")
    
    # 17) Reconstruir X final
    final_X = X_scaled[selected_features].copy()

    # (Opcional) Asegurarte de que no se haya alterado la longitud
    logging.info(f"Shape final_X: {final_X.shape}, y: {y.shape}")
    if date_data is not None:
        logging.info(f"Shape date_data: {date_data.shape}")

    # 18) Hacer un reset_index y concatenar
    final_X = final_X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    if date_data is not None:
        date_data = date_data.reset_index(drop=True)

    logging.info("Verificando longitudes antes de concat:")
    logging.info(f"Len final_X={len(final_X)}, Len y={len(y)}, Len date_data={len(date_data) if date_data is not None else 'None'}")

    if date_data is not None:
        final_df = pd.concat([date_data, final_X, y], axis=1)
        final_df.columns = [DATE_COL] + list(final_X.columns) + [target_col]
    else:
        final_df = pd.concat([final_X, y], axis=1)

    # (Opcional) Si deseas eliminar filas que no tengan fecha ni target, descomenta:
    # final_df.dropna(subset=[DATE_COL, target_col], inplace=True)

    logging.info(f"Shape final tras la selección de features y concat: {final_df.shape}")
    logging.info(f"Columnas del DataFrame final: {list(final_df.columns)}")
    logging.info(f"Primeras 3 filas del DataFrame final:\n{final_df.head(3)}")
    logging.info(f"Últimas 3 filas del DataFrame final:\n{final_df.tail(3)}")
    
    # Log de la reducción de dimensionalidad
    original_features = X.shape[1]
    final_features = final_X.shape[1]
    reduction_pct = (original_features - final_features) / original_features * 100
    logging.info(f"Reducción de dimensionalidad: {original_features} → {final_features} features ({reduction_pct:.1f}% reducción)")
    
    # Verificar nulls en el dataset final
    null_counts = final_df.isna().sum()
    if null_counts.sum() > 0:
        logging.warning("⚠️ El dataset final contiene valores nulos:")
        for col, count in null_counts[null_counts > 0].items():
            logging.warning(f"  - {col}: {count} valores nulos")

    # 19) Guardar a Excel
    try:
        save_start_time = time.time()
        final_df.to_excel(output_file, index=False)
        save_time = time.time() - save_start_time
        logging.info(f"Dataset final guardado en '{output_file}' con forma {final_df.shape} en {save_time:.2f} segundos.")
        
        # Tamaño de archivo
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logging.info(f"Tamaño del archivo: {file_size_mb:.2f} MB")
    except Exception as e:
        logging.error(f"❌ Error al guardar el dataset final: {e}")
    
    # Tiempo total de ejecución
    total_time = time.time() - main_start_time
    mins, secs = divmod(total_time, 60)
    logging.info("=" * 60)
    logging.info(f"PROCESO COMPLETADO EN {int(mins)}m {secs:.2f}s")
    logging.info(f"Features originales: {original_features}")
    logging.info(f"Features seleccionadas: {final_features} ({final_features/original_features*100:.1f}%)")
    logging.info(f"Reducción: {original_features - final_features} features ({reduction_pct:.1f}%)")
    logging.info(f"Archivo final: {output_file}")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()