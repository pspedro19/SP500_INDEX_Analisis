import os
import logging
import pandas as pd
import numpy as np
import joblib
import json
import glob
import time
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Importar configuraciones centralizadas
from sp500_analysis.config.settings import settings
from pipelines.ml.config import ensure_directories

PROJECT_ROOT = settings.project_root
MODELS_DIR = settings.models_dir
TRAINING_DIR = settings.training_dir
RESULTS_DIR = settings.results_dir
IMG_CHARTS = settings.img_charts_dir
DATE_COL = settings.date_col
LOCAL_REFINEMENT_DAYS = settings.local_refinement_days
TRAIN_TEST_SPLIT_RATIO = settings.train_test_split_ratio
FORECAST_HORIZON_1MONTH = settings.forecast_horizon_1month
FORECAST_HORIZON_3MONTHS = settings.forecast_horizon_3months

# Importar funciones de visualización
from sp500_analysis.shared.visualization.plotters import plot_forecast

# Configuración de logging
log_file = os.path.join(PROJECT_ROOT, "logs", f"inference_{time.strftime('%Y%m%d_%H%M%S')}.log")
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


def load_all_models():
    """
    Carga todos los modelos disponibles (individuales y ensemble).

    Returns:
        dict: Diccionario con los modelos cargados {nombre_modelo: modelo}
    """
    t0 = time.perf_counter()
    models = {}

    # Cargar ensemble si existe
    ensemble_path = os.path.join(MODELS_DIR, "ensemble_greedy.pkl")
    if os.path.exists(ensemble_path):
        try:
            models["Ensemble"] = joblib.load(ensemble_path)
            logging.info(f"Modelo ensemble cargado desde {ensemble_path}")
        except Exception as e:
            logging.error(f"Error al cargar ensemble: {e}")

    # Cargar modelos individuales
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    for model_path in model_files:
        model_name = os.path.basename(model_path).replace('.pkl', '')
        # Saltar el ensemble que ya cargamos y posibles backup/temp files
        if "ensemble" in model_name.lower() or model_name.startswith('.'):
            continue

        try:
            models[model_name] = joblib.load(model_path)
            logging.info(f"Modelo {model_name} cargado desde {model_path}")
        except Exception as e:
            logging.error(f"Error al cargar {model_name}: {e}")

    if not models:
        logging.error("No se pudo cargar ningún modelo")
        return {}

    logging.info(f"Total de modelos cargados: {len(models)}")
    t1 = time.perf_counter()
    logging.info(f"Tiempo de carga de modelos: {t1-t0:.2f}s")
    return models


def refinamiento_local(model, X, y):
    """
    Realiza refinamiento local del modelo usando los últimos n_days.

    Args:
        model: Modelo a refinar
        X: Features completos
        y: Target completo

    Returns:
        object: Modelo refinado
    """
    t0 = time.perf_counter()
    # Asegurarse de que tenemos un modelo que se puede ajustar
    if not hasattr(model, 'fit'):
        logging.warning("El modelo no tiene método 'fit', no se puede refinar localmente")
        return model

    # Usar solo los últimos LOCAL_REFINEMENT_DAYS días
    if len(X) > LOCAL_REFINEMENT_DAYS:
        X_local = X.tail(LOCAL_REFINEMENT_DAYS).copy()
        y_local = y.tail(LOCAL_REFINEMENT_DAYS).copy()
        logging.info(f"Refinamiento local usando los últimos {LOCAL_REFINEMENT_DAYS} días")
    else:
        X_local = X.copy()
        y_local = y.copy()
        logging.info(f"Refinamiento local usando todos los {len(X)} días disponibles")

    # Dividir en train/test (80/20 sin shuffle)
    train_size = int(len(X_local) * TRAIN_TEST_SPLIT_RATIO)
    X_train = X_local.iloc[:train_size]
    y_train = y_local.iloc[:train_size]
    X_test = X_local.iloc[train_size:]
    y_test = y_local.iloc[train_size:]

    # Refinar modelo con datos de entrenamiento
    try:
        model.fit(X_train, y_train)
        logging.info("Modelo refinado localmente con éxito")

        # Evaluar refinamiento local
        if len(X_test) > 0:
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            logging.info(f"RMSE local tras refinamiento: {rmse:.4f}")
        else:
            logging.warning("No hay datos suficientes para evaluación local")

        t1 = time.perf_counter()
        logging.info(f"Tiempo de refinamiento local: {t1-t0:.2f}s")
        return model
    except Exception as e:
        logging.error(f"Error al refinar modelo localmente: {e}")
        return model  # Devolver el modelo original si hay error


def get_inference_for_all_models(models, dataset, date_inference=None, forecast_period="1MONTH"):
    """
    Realiza inferencia con todos los modelos para una fecha específica, con refinamiento local.

    Args:
        models: Diccionario de modelos {nombre_modelo: modelo}
        dataset: DataFrame con features y target
        date_inference: Fecha para la inferencia (por defecto la más reciente)
        forecast_period: Periodo de pronóstico ("1MONTH" o "3MONTHS")

    Returns:
        dict: Resultados de inferencia para todos los modelos
    """
    t0 = time.perf_counter()
    results = {}
    forecasts_df = {}

    if not models:
        logging.error("No se proporcionaron modelos válidos para la inferencia")
        return results, forecasts_df

    # Determinar el horizonte según forecast_period
    if forecast_period == "1MONTH":
        forecast_horizon = FORECAST_HORIZON_1MONTH
    elif forecast_period == "3MONTHS":
        forecast_horizon = FORECAST_HORIZON_3MONTHS
    else:
        forecast_horizon = FORECAST_HORIZON_1MONTH  # Default

    # Convertir fechas a datetime
    dataset[DATE_COL] = pd.to_datetime(dataset[DATE_COL])

    # Si no se especifica fecha, usar la más reciente
    if date_inference is None:
        date_inference = dataset[DATE_COL].max()
    else:
        date_inference = pd.to_datetime(date_inference)

    logging.info(f"Realizando inferencia para fecha: {date_inference.strftime('%Y-%m-%d')}")

    # Verificar que existe la fecha de inferencia
    if date_inference not in dataset[DATE_COL].values:
        logging.error(f"No hay datos para la fecha de inferencia: {date_inference}")
        return results, forecasts_df

    # Identificar la columna target (última columna)
    if "_Target" in dataset.columns[-1]:
        target_col = dataset.columns[-1]
    else:
        target_cols = [col for col in dataset.columns if col.endswith("_Target")]
        if target_cols:
            target_col = target_cols[0]
        else:
            target_col = dataset.columns[-1]
            logging.warning(f"No se encontró columna con sufijo '_Target'. Usando última columna: {target_col}")

    # Extraer el nombre base de la columna target (sin "_Target")
    if target_col.endswith("_Target"):
        valor_real_col = target_col[:-7]  # Quitar "_Target"
    else:
        valor_real_col = target_col

    logging.info(f"Columna target identificada: {target_col}")
    logging.info(f"Columna de valor real: {valor_real_col}")

    # Separar datos históricos para refinamiento
    historical_df = dataset[dataset[DATE_COL] < date_inference].copy()
    historical_df = historical_df.sort_values(DATE_COL)

    # Extraer features y target para refinamiento
    X_hist = historical_df.drop(columns=[target_col, DATE_COL])
    y_hist = historical_df[target_col]

    # Extraer la fila de inferencia
    inference_row = dataset[dataset[DATE_COL] == date_inference]
    X_inference = inference_row.drop(columns=[target_col, DATE_COL])

    # Para generar fechas futuras (días hábiles)
    dates_future = pd.date_range(start=date_inference + pd.Timedelta(days=1), periods=forecast_horizon, freq="B")

    # Realizar inferencia para cada modelo
    for model_name, model in models.items():
        try:
            # Refinar modelo localmente
            refined_model = refinamiento_local(model, X_hist, y_hist)

            # Realizar predicción para el día actual (inferencia)
            try:
                prediction = refined_model.predict(X_inference)[0]
                logging.info(f"Predicción {model_name}: {prediction:.6f}")
            except Exception as e:
                logging.error(f"Error al predecir con {model_name}: {e}")
                continue

            # Calcular fecha objetivo (fecha_inferencia + horizonte días hábiles)
            target_date = date_inference + pd.Timedelta(days=int(forecast_horizon * 1.4))

            # Crear resultado para este modelo
            results[model_name] = {
                'date_inference': date_inference.strftime('%Y-%m-%d'),
                'target_date': target_date.strftime('%Y-%m-%d'),
                'prediction': float(prediction),
                'forecast_period': forecast_period,
                'forecast_horizon': forecast_horizon,
                'model_type': type(model).__name__,
                'model_name': model_name,
                'refinement_days': LOCAL_REFINEMENT_DAYS,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }

            # Ahora generar forecast usando el último valor disponible
            last_features = X_inference.values.reshape(1, -1)
            future_predictions = [prediction]  # Primer valor es la inferencia actual

            # Generar predicciones para el horizonte
            current_features = last_features.copy()

            for i in range(forecast_horizon - 1):
                # Algunas estrategias podrían ser:
                # - Mantener las features constantes (la más simple)
                # - Aplicar un modelo autorregresivo simple
                # - Usar un esquema recursivo específico para el modelo

                # Aquí usamos la estrategia más simple: mantener features constantes
                try:
                    next_pred = refined_model.predict(current_features)[0]
                    future_predictions.append(next_pred)
                except Exception as e:
                    logging.error(f"Error en forecast paso {i+1} para {model_name}: {e}")
                    future_predictions.append(np.nan)

            # Crear DataFrame con forecast para este modelo
            # Primero creamos la parte histórica
            hist_dates = historical_df[DATE_COL].tolist() + [date_inference]
            hist_real = historical_df[valor_real_col].tolist() + [inference_row[valor_real_col].values[0]]
            hist_pred = [np.nan] * len(historical_df) + [prediction]

            # Luego añadimos el futuro
            future_dates = dates_future.tolist()
            future_real = [np.nan] * len(future_dates)

            # Crear DataFrame completo (histórico + forecast)
            forecast_df = pd.DataFrame(
                {
                    DATE_COL: hist_dates + future_dates,
                    'Valor_Real': hist_real + future_real,
                    'Valor_Predicho': hist_pred + future_predictions,
                    'Modelo': model_name,
                    'Periodo': ['Historico'] * len(hist_dates) + ['Forecast'] * len(future_dates),
                }
            )

            forecasts_df[model_name] = forecast_df

        except Exception as e:
            logging.error(f"Error al procesar modelo {model_name}: {e}")

    t1 = time.perf_counter()
    logging.info(f"Tiempo total de inferencia: {t1-t0:.2f}s")

    return results, forecasts_df


def save_all_inference_results(results, forecasts_df, output_dir=None):
    """
    Guarda los resultados de inferencia de todos los modelos en formato JSON.

    Args:
        results (dict): Resultados de inferencia para todos los modelos
        forecasts_df (dict): DataFrames con forecasts para cada modelo
        output_dir (str): Directorio para guardar resultados

    Returns:
        dict: Paths a los archivos guardados {modelo: path}
    """
    t0 = time.perf_counter()
    if not results:
        logging.error("No hay resultados para guardar")
        return {}

    # Crear directorio si no se especifica
    if output_dir is None:
        date_str = list(results.values())[0]['date_inference'].replace('-', '')
        output_dir = os.path.join(RESULTS_DIR, f"inference_{date_str}")

    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)

    saved_files = {}

    # Guardar resultado consolidado con todos los modelos
    consolidated_file = os.path.join(output_dir, "all_models_inference.json")
    with open(consolidated_file, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Resultados consolidados guardados en: {consolidated_file}")

    # Guardar CSV consolidado para PowerBI
    fecha_inferencia = list(results.values())[0]['date_inference']
    consolidated_df = pd.DataFrame(
        [
            {
                'Fecha_Inferencia': fecha_inferencia,
                'Modelo': model_name,
                'Prediccion': result['prediction'],
                'Fecha_Objetivo': result['target_date'],
                'Horizonte_Dias': result['forecast_horizon'],
                'Tiempo_Ejecucion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            for model_name, result in results.items()
        ]
    )

    consolidated_csv = os.path.join(output_dir, "predictions_api.csv")
    consolidated_df.to_csv(consolidated_csv, index=False)
    logging.info(f"CSV consolidado guardado en: {consolidated_csv}")

    # También guardar como JSON para API
    api_json = os.path.join(RESULTS_DIR, "predictions_api.json")
    with open(api_json, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"JSON para API guardado en: {api_json}")

    # Guardar resultados individuales
    for model_name, result in results.items():
        model_file = os.path.join(output_dir, f"{model_name}_inference.json")
        with open(model_file, 'w') as f:
            json.dump(result, f, indent=4)
        saved_files[model_name] = model_file
        logging.info(f"Resultado de {model_name} guardado en: {model_file}")

        # Guardar gráfico de forecast
        if model_name in forecasts_df:
            chart_path = os.path.join(IMG_CHARTS, f"{model_name}_forecast.png")

            # Calcular métrica RMSE usando solo datos históricos donde hay predicción
            df_model = forecasts_df[model_name]
            df_eval = df_model.dropna(subset=['Valor_Real', 'Valor_Predicho'])

            metrics = {}
            if len(df_eval) > 0:
                rmse = np.sqrt(mean_squared_error(df_eval['Valor_Real'], df_eval['Valor_Predicho']))
                metrics['RMSE'] = rmse

            plot_forecast(
                forecasts_df[model_name],
                inference_date=result['date_inference'],
                title=f"Forecast {model_name} - Horizonte: {result['forecast_horizon']} días",
                metrics=metrics,
                model_name=model_name,
                output_path=chart_path,
            )
            logging.info(f"Gráfico de forecast guardado en: {chart_path}")

    # Guardar todas las predicciones en un solo DataFrame
    all_forecasts = pd.concat(forecasts_df.values())
    all_forecasts_csv = os.path.join(output_dir, "all_forecasts.csv")
    all_forecasts.to_csv(all_forecasts_csv, index=False)
    logging.info(f"CSV con todos los forecasts guardado en: {all_forecasts_csv}")

    # Generar gráfico comparativo con todos los modelos
    if len(forecasts_df) > 1:
        # Filtrar solo período de forecast
        future_forecasts = {}
        for model_name, df in forecasts_df.items():
            future_forecasts[model_name] = df[df['Periodo'] == 'Forecast']

        # Crear figura para comparar forecasts
        plt.figure(figsize=(12, 8))

        # Plotear datos históricos solo una vez
        hist_data = next(iter(forecasts_df.values()))
        hist_data = hist_data[hist_data['Periodo'] == 'Historico']
        if not hist_data.empty:
            plt.plot(hist_data[DATE_COL], hist_data['Valor_Real'], label='Histórico (real)', color='blue', linewidth=2)

        # Plotear forecast para cada modelo
        for model_name, df in future_forecasts.items():
            plt.plot(df[DATE_COL], df['Valor_Predicho'], label=f'Forecast {model_name}', marker='o', linewidth=1.5)

        # Fecha de inferencia
        inference_date = pd.to_datetime(list(results.values())[0]['date_inference'])
        plt.axvline(
            x=inference_date,
            color='black',
            linestyle='--',
            alpha=0.7,
            label=f'Fecha de corte: {inference_date.strftime("%Y-%m-%d")}',
        )

        plt.title(f"Comparación de Forecasts - {len(future_forecasts)} modelos")
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        comparison_path = os.path.join(IMG_CHARTS, "all_models_forecast_comparison.png")
        plt.savefig(comparison_path, dpi=300)
        plt.close()
        logging.info(f"Gráfico comparativo guardado en: {comparison_path}")

    t1 = time.perf_counter()
    logging.info(f"Tiempo para guardar resultados: {t1-t0:.2f}s")

    return saved_files


def main():
    """
    Función principal para realizar inferencia con todos los modelos (individuales y ensemble).

    Proceso:
    1. Cargar todos los modelos disponibles
    2. Cargar datos
    3. Realizar inferencia para la fecha más reciente con todos los modelos
    4. Generar visualizaciones de forecast
    5. Guardar resultados individuales y consolidados
    """
    t0 = time.perf_counter()
    logging.info("Iniciando proceso de inferencia para todos los modelos...")

    # Parámetros configurables
    forecast_period = "1MONTH"  # "1MONTH" o "3MONTHS"
    date_inference = None  # None para usar la fecha más reciente

    # 1. Cargar todos los modelos
    t1 = time.perf_counter()
    models = load_all_models()
    if not models:
        logging.error("No se pudo cargar ningún modelo. Abortando.")
        return

    # 2. Cargar datos
    t2 = time.perf_counter()
    dataset_path = get_most_recent_file(TRAINING_DIR, pattern="*FPI.xlsx")
    if not dataset_path:
        logging.error(f"No se encontró ningún archivo de datos en {TRAINING_DIR}")
        return

    logging.info(f"Cargando datos desde: {os.path.basename(dataset_path)}")
    try:
        dataset = pd.read_excel(dataset_path)
        logging.info(f"Datos cargados: {dataset.shape[0]} filas, {dataset.shape[1]} columnas")
    except Exception as e:
        logging.error(f"Error al cargar datos: {e}")
        return

    # 3. Realizar inferencia con todos los modelos
    t3 = time.perf_counter()
    results, forecasts_df = get_inference_for_all_models(
        models=models, dataset=dataset, date_inference=date_inference, forecast_period=forecast_period
    )

    if not results:
        logging.error("No se pudo realizar la inferencia. Abortando.")
        return

    # 4. Guardar resultados y generar visualizaciones
    t4 = time.perf_counter()
    saved_files = save_all_inference_results(results, forecasts_df)
    t5 = time.perf_counter()

    # Tiempos de ejecución
    logging.info(f"Tiempo carga de modelos: {t2-t1:.2f}s")
    logging.info(f"Tiempo carga de datos: {t3-t2:.2f}s")
    logging.info(f"Tiempo inferencia: {t4-t3:.2f}s")
    logging.info(f"Tiempo guardar resultados: {t5-t4:.2f}s")
    logging.info(f"Tiempo total: {t5-t0:.2f}s")

    logging.info(f"Proceso de inferencia completado exitosamente para {len(results)} modelos.")
    print(f"✅ Inferencia completada para {len(results)} modelos.")
    print(f"✅ Visualizaciones de forecast generadas en: {IMG_CHARTS}")
    print(f"✅ Resultados guardados en API: {os.path.join(RESULTS_DIR, 'predictions_api.json')}")


if __name__ == "__main__":
    main()
