import os
import glob
import random
import numpy as np
import pandas as pd
from pandas import DataFrame
import logging
from sp500_analysis.shared.logging.logger import configurar_logging
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pandas_market_calendars as mcal

# Importar configuraciones centralizadas
from sp500_analysis.config.settings import settings


from sp500_analysis.application.feature_engineering import get_most_recent_file, plot_cv_splits, plot_performance_drift, select_features_fpi
PROJECT_ROOT = settings.project_root
PROCESSED_DIR = settings.processed_dir
TRAINING_DIR = settings.training_dir
DATE_COL = settings.date_col
CV_SPLITS = settings.cv_splits
FORECAST_HORIZON_1MONTH = settings.forecast_horizon_1month
FORECAST_HORIZON_3MONTHS = settings.forecast_horizon_3months
CV_GAP_1MONTH = settings.cv_gap_1month
CV_GAP_3MONTHS = settings.cv_gap_3months
FPI_THRESHOLD = settings.fpi_threshold
CATBOOST_PARAMS = settings.catboost_params
SCORER = settings.scorer
RANDOM_SEED = settings.random_seed

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
configurar_logging(log_file)

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
# FUNCIÓN PRINCIPAL
# ------------------------------


def main() -> None:
    """
    Función principal que ejecuta la selección de features FPI.

    Proceso:
    1. Carga el archivo más reciente
    2. Preprocesa los datos (fechas, filtrado por días hábiles)
    3. Aplica Feature Permutation Importance con validación temporal
    4. Guarda el dataset final con las features seleccionadas
    """
    main_start_time = time.time()

    # 1) CORRECCIÓN CRÍTICA: Usar archivo de TRAINING en lugar de buscar el más reciente
    # El problema era que se usaba INFERENCE que no tiene targets válidos
    training_file = os.path.join(PROCESSED_DIR, "datos_economicos_1month_SP500_TRAINING.xlsx")
    
    if os.path.exists(training_file):
        input_file = training_file
        logging.info(f"✅ Usando archivo de TRAINING: {input_file}")
    else:
        # Fallback: buscar archivo más reciente como antes
        input_file = get_most_recent_file(INPUT_DIR)
        if not input_file:
            logging.error(f"❌ No se encontraron archivos Excel en {INPUT_DIR}")
            return
        logging.warning(f"⚠️ No se encontró archivo de TRAINING, usando: {input_file}")

    # Extraer el nombre base del archivo para generar el nombre de salida
    base_name = os.path.basename(input_file)
    file_name, file_ext = os.path.splitext(base_name)
    output_file = os.path.join(OUTPUT_DIR, f"{file_name}_FPI{file_ext}")

    logging.info("=" * 60)
    logging.info("INICIANDO SELECCIÓN DE FEATURES FPI")
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
        logging.info(
            f"  - {col}: min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}"
        )

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
    logging.info(f"Conversión de fechas: {original_dtype} -> {df[DATE_COL].dtype}")

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
        logging.warning(f"WARNING: No se encontró columna de fecha '{DATE_COL}' en las features.")

    # 12) Quedarnos solo con columnas numéricas
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        logging.warning(f"WARNING: Eliminando {len(non_numeric_cols)} columnas no numéricas: {non_numeric_cols}")

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
            logging.info(
                f"  - {col}: antes [mean={before_mean:.4f}, std={before_std:.4f}], después [mean={after_mean:.4f}, std={after_std:.4f}]"
            )
    else:
        X_scaled = X_numeric.copy()
        logging.info("No se aplicó escalado; se mantienen los valores originales.")

    # 14) Eliminar filas donde el target sea NaN (si las hay)
    if y.isna().any():
        num_nans = y.isna().sum()
        logging.warning(
            f"WARNING: El target contiene {num_nans} valores NaN ({num_nans/len(y)*100:.2f}%); se eliminarán esas filas."
        )
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
            logging.warning(f"WARNING: Columna '{col}' tiene {na_count} NaNs y {inf_count} infinitos")

    logging.info(f"Últimas 3 filas de X_scaled:\n{X_scaled.tail(3)}")
    logging.info(f"Últimas 3 filas de y:\n{y.tail(3)}")
    if date_data is not None:
        logging.info(f"Últimas 3 filas de date_data:\n{date_data.tail(3)}")

    # 16) Selección de features con FPI - usando gap específico según forecast period
    logging.info("=" * 60)
    logging.info("INICIANDO SELECCIÓN DE FEATURES CON FPI")
    logging.info("=" * 60)
    logging.info(f"Número total de features antes de FPI: {X_scaled.shape[1]}")
    logging.info(f"Forecast period: {FORECAST_PERIOD}")
    gap = CV_GAP_1MONTH if FORECAST_PERIOD == "1MONTH" else CV_GAP_3MONTHS
    logging.info(f"Threshold FPI: {THRESHOLD}")

    # Verificar valores NaN o inf en el target
    if y.isna().any() or np.isinf(y).any().any():
        logging.error("❌ El target contiene valores NaN o infinitos. FPI fallará.")
        return

    fpi_start_time = time.time()
    selected_features, performance_drifts = select_features_fpi(
        X_scaled, y, cv_splits=CV_SPLITS, gap=gap, threshold=THRESHOLD,
        catboost_params=CATBOOST_PARAMS, scorer=SCORER,
        plots_dir=PLOTS_DIR, timestamp=timestamp
    )
    fpi_time = time.time() - fpi_start_time
    logging.info(f"Proceso FPI completado en {fpi_time:.2f} segundos")

    if not selected_features:
        logging.error("❌ No se logró seleccionar ninguna feature (posible error en FPI). Terminando.")
        return

    logging.info(
        f"Features finales seleccionadas ({len(selected_features)}/{X_scaled.shape[1]} = {len(selected_features)/X_scaled.shape[1]*100:.1f}%):"
    )

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
    logging.info(
        f"Len final_X={len(final_X)}, Len y={len(y)}, Len date_data={len(date_data) if date_data is not None else 'None'}"
    )

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
    logging.info(
        f"Reducción de dimensionalidad: {original_features} → {final_features} features ({reduction_pct:.1f}% reducción)"
    )

    # Verificar nulls en el dataset final
    null_counts = final_df.isna().sum()
    if null_counts.sum() > 0:
        logging.warning("WARNING: El dataset final contiene valores nulos:")
        for col, count in null_counts[null_counts > 0].items():
            logging.warning(f"  - {col}: {count} valores nulos")

    # 19) Guardar a Excel
    try:
        save_start_time = time.time()
        final_df.to_excel(output_file, index=False)
        save_time = time.time() - save_start_time
        logging.info(
            f"Dataset final guardado en '{output_file}' con forma {final_df.shape} en {save_time:.2f} segundos."
        )

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
