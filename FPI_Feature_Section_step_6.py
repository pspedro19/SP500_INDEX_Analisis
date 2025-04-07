

import pandas as pd
import numpy as np
import logging
import random
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from feature_engine.selection import SelectByShuffling
import pandas_market_calendars as mcal

# ------------------------------
# CONFIGURACIÓN GLOBAL
# ------------------------------
INPUT_FILE = "Data/Macro/Eliminar_relaciones/EUR.xlsx"
OUTPUT_FILE = "Data/Macro/FPI_Feature_Section/EUR_final_FPI.xlsx"

# Si quieres forzar el target por nombre, ponlo aquí; en caso contrario, se usará la última columna.
# TARGET_COL_NAME = 'Target'
TARGET_COL_NAME = None  

DATE_COL = 'date'
CV_SPLITS = 5
GAP = 20
SCORER = 'neg_mean_squared_error'
APPLY_SCALING = False

# Ajusta este threshold para hacer la selección de features más o menos agresiva:
# Cuanto más alto el threshold, mayor caída de performance se requiere para descartar una feature.
THRESHOLD = 0.000001

CATBOOST_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.01,
    'max_depth': 8,
    'random_seed': 42,
    'verbose': 0
}

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def select_features_fpi(X, y, cv_splits=CV_SPLITS, gap=GAP, threshold=THRESHOLD):
    """
    Realiza Feature Permutation Importance (FPI) usando CatBoostRegressor y SelectByShuffling.
    Usa un threshold para hacer la selección menos agresiva.
    Retorna:
      selected_features: lista de nombres de features finales
      performance_drifts: array con los drift scores (uno por feature)
    """
    tscv = TimeSeriesSplit(n_splits=cv_splits, gap=gap)
    regressor = CatBoostRegressor(**CATBOOST_PARAMS)

    selector = SelectByShuffling(
        estimator=regressor,
        scoring=SCORER,
        cv=tscv,
        threshold=threshold
    )

    logging.info(f"[FPI] Iniciando SelectByShuffling con threshold={threshold}")
    selector.fit(X, y)

    selected_features = selector.get_feature_names_out()
    performance_drifts = selector.performance_drifts_

    logging.info(f"[FPI] Se calcularon drift scores para {len(X.columns)} columnas.")
    logging.info(f"[FPI] Features seleccionadas por FPI: {len(selected_features)} de {len(X.columns)}")

    # Verifica longitudes
    if len(performance_drifts) != len(X.columns):
        logging.error("[FPI] Mismatch entre número de columnas de X y performance drifts. Abortando.")
        return [], []

    # Construye DataFrame con los drifts
    drift_df = pd.DataFrame(
        list(zip(X.columns, performance_drifts)),
        columns=['feature', 'performance_drift']
    ).sort_values('performance_drift', ascending=False)

    logging.info("[FPI] Importancia (performance drift) de cada feature:")
    logging.info(f"\n{drift_df}\n")

    return list(selected_features), performance_drifts

def main():
    # 1) Lectura del Excel original
    try:
        df_original = pd.read_excel(INPUT_FILE)
        logging.info(f"Archivo '{INPUT_FILE}' cargado con forma {df_original.shape}.")
    except Exception as e:
        logging.error(f"Error al cargar '{INPUT_FILE}': {e}")
        return

    logging.info(f"Columnas del DataFrame original: {list(df_original.columns)}")
    logging.info(f"Últimas 5 filas (tail) del DataFrame original:\n{df_original.tail(5)}")

    # 2) Verificar que existe la columna de fecha
    if DATE_COL not in df_original.columns:
        logging.error(f"No se encontró la columna de fecha '{DATE_COL}' en el dataset.")
        return

    # 3) Copia para transformaciones
    df = df_original.copy()

    # 4) Convertir a datetime y normalizar (quitando componente de hora)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce').dt.normalize()

    # 5) Eliminar filas con fechas inválidas
    df_before_drop = df.shape[0]
    df.dropna(subset=[DATE_COL], inplace=True)
    logging.info(f"Filas eliminadas por fecha inválida (NaT): {df_before_drop - df.shape[0]}")

    # 6) Rango de fechas tras limpieza
    data_min_date = df[DATE_COL].min()
    data_max_date = df[DATE_COL].max()
    logging.info(f"Rango de fechas después de limpiar fechas inválidas: {data_min_date} a {data_max_date}")
    logging.info(f"Cantidad de fechas únicas ahora: {df[DATE_COL].nunique()}")

    # 7) Construir calendario de NYSE y filtrar solo días hábiles de trading
    nyse_cal = mcal.get_calendar('NYSE')
    schedule = nyse_cal.schedule(start_date=data_min_date, end_date=data_max_date)
    trading_days = mcal.date_range(schedule, frequency='1D').normalize()
    trading_days = trading_days.tz_localize(None)  # quita timezone

    if len(trading_days) > 0:
        logging.info(f"Primeros 5 días hábiles: {trading_days[:5].to_list()}")
        logging.info(f"Últimos 5 días hábiles: {trading_days[-5:].to_list()}")
        logging.info(f"Rango total días hábiles: {trading_days.min()} a {trading_days.max()}")
    else:
        logging.warning("No se generaron días hábiles para el rango especificado.")

    before_filter_shape = df.shape
    df = df[df[DATE_COL].isin(trading_days)]
    after_filter_shape = df.shape
    logging.info(f"Filas totales antes de filtrar por días hábiles: {before_filter_shape[0]}")
    logging.info(f"Filas totales después de filtrar por días hábiles: {after_filter_shape[0]}")
    logging.info(f"Dataset tras filtrar días hábiles: {df.shape}")

    # 8) Verificar si hay datos suficientes para TimeSeriesSplit
    if df.shape[0] < CV_SPLITS * GAP:
        logging.warning(
            f"El dataset tiene {df.shape[0]} filas. "
            f"Puede que sea poco para {CV_SPLITS} splits con gap={GAP}."
        )

    # 9) Determinar la columna target
    if TARGET_COL_NAME is not None and TARGET_COL_NAME in df.columns:
        target_col = TARGET_COL_NAME
        logging.info(f"Se forzó la columna target a '{target_col}' (definido en el script).")
    else:
        target_col = df.columns[-1]
        logging.info(f"Columna target tomada como la última columna: '{target_col}'")

    # 10) Separar target y features
    y = df[target_col]
    X = df.drop(columns=[target_col], errors='ignore')

    # 11) Extraer columna de fecha de las features
    if DATE_COL in X.columns:
        logging.info(f"Extrayendo '{DATE_COL}' de las features para conservarla aparte.")
        date_data = X.pop(DATE_COL)
    else:
        date_data = None

    # 12) Quedarnos solo con columnas numéricas
    X_numeric = X.select_dtypes(include=[np.number])
    logging.info(f"Número de columnas numéricas: {len(X_numeric.columns)}")
    logging.info(f"Columnas numéricas: {list(X_numeric.columns)}")

    if X_numeric.empty or X_numeric.shape[0] == 0:
        logging.error("No hay datos numéricos disponibles para la selección de features. Abortando.")
        return

    # 13) Escalado (opcional)
    if APPLY_SCALING:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=X_numeric.columns)
        logging.info("Se aplicó escalado StandardScaler a las features.")
    else:
        X_scaled = X_numeric.copy()
        logging.info("No se aplicó escalado; se mantienen los valores originales.")

    # 14) Eliminar filas donde el target sea NaN (si las hay)
    if y.isna().any():
        num_nans = y.isna().sum()
        logging.warning(f"El target contiene {num_nans} valores NaN; se eliminarán esas filas.")
        valid_idx = ~y.isna()
        y = y[valid_idx]
        X_scaled = X_scaled[valid_idx]
        if date_data is not None:
            date_data = date_data[valid_idx]

    # Importante: forzar que todos tengan el mismo índice consecutivo 0..N-1
    y = y.reset_index(drop=True)
    X_scaled = X_scaled.reset_index(drop=True)
    if date_data is not None:
        date_data = date_data.reset_index(drop=True)

    # 15) Logs antes de hacer la selección de features
    logging.info(f"Dimensión de X_scaled antes de FPI: {X_scaled.shape}")
    logging.info(f"Dimensión de y antes de FPI: {y.shape}")
    if date_data is not None:
        logging.info(f"Dimensión de date_data antes de FPI: {date_data.shape}")
    logging.info(f"Últimas 5 filas de X_scaled:\n{X_scaled.tail(5)}")
    logging.info(f"Últimas 5 filas de y:\n{y.tail(5)}")
    if date_data is not None:
        logging.info(f"Últimas 5 filas de date_data:\n{date_data.tail(5)}")

    # 16) Selección de features con FPI
    selected_features, performance_drifts = select_features_fpi(
        X_scaled, y, cv_splits=CV_SPLITS, gap=GAP, threshold=THRESHOLD
    )

    if not selected_features:
        logging.error("No se logró seleccionar ninguna feature (posible error en FPI). Terminando.")
        return

    logging.info(f"Features finales seleccionadas: {selected_features}")

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
    logging.info(f"Últimas 5 filas del DataFrame final:\n{final_df.tail(5)}")

    # 19) Guardar a Excel
    try:
        final_df.to_excel(OUTPUT_FILE, index=False)
        logging.info(f"Dataset final guardado en '{OUTPUT_FILE}' con forma {final_df.shape}.")
    except Exception as e:
        logging.error(f"Error al guardar el dataset final: {e}")

if __name__ == "__main__":
    main()