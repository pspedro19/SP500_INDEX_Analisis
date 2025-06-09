import numpy as np
import pandas as pd
import os
import logging
import glob
import json
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# Importar configuraciones centralizadas
from src.config.base import ProjectConfig

config = ProjectConfig.from_env()

PROJECT_ROOT = config.project_root
RESULTS_DIR = config.results_dir
METRICS_DIR = config.metrics_dir
METRICS_CHARTS = config.metrics_charts_dir
SUBPERIODS_CHARTS = config.subperiods_charts_dir
DATE_COL = config.date_col
ensure_directories = config.ensure_dirs

# Importar funciones de visualización
from src.pipelines.ml.utils.plots import (
    plot_real_vs_pred, plot_metrics_by_subperiod, plot_radar_metrics,
    generate_report_figures
)

# ------------------------------
# CONFIGURACIÓN DE LOGGING
# ------------------------------
LOG_DIR = config.log_dir
log_file = os.path.join(LOG_DIR, f"backtest_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Asegurar que los directorios existen
ensure_directories()

# ------------------------------
# FUNCIONES AUXILIARES
# ------------------------------
def get_most_recent_file(directory, pattern='*.csv'):
    """
    Obtiene el archivo más reciente en un directorio que coincide con un patrón.
    
    Args:
        directory (str): Ruta al directorio
        pattern (str): Patrón para buscar archivos
        
    Returns:
        str: Ruta al archivo más reciente
    """
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

# ------------------------------
# FUNCIONES DE MÉTRICAS
# ------------------------------
def calcular_smape(real, pred):
    """
    Symmetric Mean Absolute Percentage Error, usado en ambos pipelines.
    Métrica simétrica que penaliza errores porcentuales en ambas direcciones.
    
    Args:
        real (array): Valores reales
        pred (array): Valores predichos
        
    Returns:
        float: Valor SMAPE (0-100)
    """
    return 100 * np.mean(2 * np.abs(pred - real) / (np.abs(real) + np.abs(pred)))

def calcular_ultra_metric(real, pred, w1=0.5, w2=0.3, w3=0.2):
    """
    Métrica combinada como en el pipeline de series temporales.
    Pondera RMSE, MAE y SMAPE con pesos configurables.
    
    Args:
        real (array): Valores reales
        pred (array): Valores predichos
        w1, w2, w3 (float): Pesos para cada componente
        
    Returns:
        float: Valor de la métrica combinada
    """
    rmse = np.sqrt(mean_squared_error(real, pred))
    mae = mean_absolute_error(real, pred)
    smape = calcular_smape(real, pred)
    
    # Normalizar para que estén en escalas similares
    norm_rmse = rmse / np.std(real) if np.std(real) > 0 else 0
    norm_mae = mae / np.mean(np.abs(real)) if np.mean(np.abs(real)) > 0 else 0
    norm_smape = smape / 100
    
    return w1 * norm_rmse + w2 * norm_mae + w3 * norm_smape

def calcular_amplitud_score(real, pred):
    """
    Calcula cuán similares son las amplitudes de las series.
    Utiliza transformada de Hilbert para extraer la envolvente.
    
    Args:
        real (array): Valores reales
        pred (array): Valores predichos
        
    Returns:
        float: Score de similitud de amplitud (0-1)
    """
    envolvente_real = np.abs(hilbert(real))
    envolvente_pred = np.abs(hilbert(pred))
    iqr_real = np.percentile(envolvente_real, 75) - np.percentile(envolvente_real, 25)
    iqr_pred = np.percentile(envolvente_pred, 75) - np.percentile(envolvente_pred, 25)
    
    if max(iqr_real, iqr_pred) == 0:
        return 1.0  # Evitar división por cero
        
    return np.clip(1 - abs(iqr_real - iqr_pred) / max(iqr_real, iqr_pred), 0, 1)

def calcular_fase_score(real, pred):
    """
    Calcula cuán similares son las fases de las series.
    Utiliza transformada de Hilbert para extraer la fase.
    
    Args:
        real (array): Valores reales
        pred (array): Valores predichos
        
    Returns:
        float: Score de similitud de fase (0-1)
    """
    fase_real = np.unwrap(np.angle(hilbert(real)))
    fase_pred = np.unwrap(np.angle(hilbert(pred)))
    error_fase = np.std(fase_real - fase_pred)
    return np.clip(1 / (1 + error_fase), 0, 1)

def calcular_hit_direction(real, pred):
    """
    Calcula el porcentaje de aciertos en la dirección del movimiento.
    
    Args:
        real (array): Valores reales
        pred (array): Valores predichos
        
    Returns:
        float: Porcentaje de acierto en dirección (0-100)
    """
    if len(real) <= 1 or len(pred) <= 1:
        return np.nan
        
    # Calcular cambios
    real_changes = np.diff(real)
    pred_changes = np.diff(pred)
    
    # Determinar dirección (1=sube, 0=baja)
    real_direction = (real_changes > 0).astype(int)
    pred_direction = (pred_changes > 0).astype(int)
    
    # Calcular aciertos
    hits = (real_direction == pred_direction).sum()
    return 100 * hits / len(real_direction)

def calcular_todas_metricas(real, pred):
    """
    Calcula todas las métricas disponibles para un conjunto de valores reales y predichos.
    
    Args:
        real (array): Valores reales
        pred (array): Valores predichos
        
    Returns:
        dict: Diccionario con todas las métricas calculadas
    """
    # Métricas básicas
    mae = mean_absolute_error(real, pred)
    mse = mean_squared_error(real, pred)
    rmse = np.sqrt(mse)
    
    # Métricas porcentuales
    mape = np.mean(np.abs((real - pred) / real)) * 100
    smape = calcular_smape(real, pred)
    
    # Métricas de ajuste
    r2 = r2_score(real, pred)
    r2_adj = 1 - ((1 - r2) * (len(real) - 1)) / (len(real) - 1 - 1)
    
    # Métricas especiales
    amplitud = calcular_amplitud_score(real, pred)
    fase = calcular_fase_score(real, pred)
    ultra_metric = calcular_ultra_metric(real, pred)
    hit_direction = calcular_hit_direction(real, pred)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape,
        'R2': r2,
        'R2_adjusted': r2_adj,
        'Amplitud_Score': amplitud,
        'Fase_Score': fase,
        'Ultra_Metric': ultra_metric,
        'Hit_Direction': hit_direction
    }

def calcular_metricas_por_subperiodos(df, real_col='Valor_Real', pred_col='Valor_Predicho',
                                      date_col=DATE_COL, freq='Q'):
    """
    Calcula métricas separando por subperíodos (trimestres, semestres, años).
    
    Args:
        df (DataFrame): DataFrame con valores reales y predichos
        real_col (str): Nombre de la columna con valores reales
        pred_col (str): Nombre de la columna con valores predichos
        date_col (str): Nombre de la columna de fecha
        freq (str): Frecuencia para subperíodos ('Q'=trimestral, 'A'=anual)
        
    Returns:
        DataFrame: DataFrame con métricas por subperíodo
    """
    # Asegurar que fecha está en formato datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Crear columna de período
    if freq == 'Q':
        df['periodo'] = df[date_col].dt.to_period('Q').astype(str)
        periodo_nombre = "Trimestre"
    elif freq == '6M':
        df['periodo'] = df[date_col].dt.to_period('6M').astype(str)
        periodo_nombre = "Semestre"
    elif freq == 'A':
        df['periodo'] = df[date_col].dt.to_period('A').astype(str)
        periodo_nombre = "Año"
    else:
        df['periodo'] = df[date_col].dt.to_period(freq).astype(str)
        periodo_nombre = "Período"
    
    # Agrupar por período y modelo
    resultados = []
    
    for modelo in df['Modelo'].unique():
        df_modelo = df[df['Modelo'] == modelo]
        
        for periodo, grupo in df_modelo.groupby('periodo'):
            if len(grupo) < 5:  # Saltear períodos con pocos datos
                continue
                
            real = grupo[real_col].values
            pred = grupo[pred_col].values
            
            # Calcular métricas
            metricas = calcular_todas_metricas(real, pred)
            
            # Crear fila de resultado
            resultado = {
                'Modelo': modelo,
                'Periodo': periodo,
                'Num_Observaciones': len(grupo),
                **metricas
            }
            
            resultados.append(resultado)
    
    # Convertir a DataFrame
    df_resultados = pd.DataFrame(resultados)
    
    # Ordenar por período y modelo
    if 'Periodo' in df_resultados.columns:
        df_resultados = df_resultados.sort_values(['Periodo', 'Modelo'])
    
    return df_resultados, periodo_nombre

def main():
    """
    Función principal que realiza el backtest de los modelos entrenados.
    
    Proceso:
    1. Lee predicciones desde el archivo de PowerBI
    2. Para cada combinación modelo/mercado:
       - Convierte los datos y calcula métricas
       - Genera visualizaciones de comparación
       - Guarda resultados individuales y consolidados
    """
    t0 = time.perf_counter()
    logging.info("Iniciando proceso de backtest y generación de reportes...")
    
    # Rutas de archivos - usando constantes de config
    input_dir = RESULTS_DIR
    output_dir = METRICS_DIR
    output_charts = METRICS_CHARTS
    subperiods_dir = SUBPERIODS_CHARTS
    
    # Buscar el archivo más reciente de tipo PowerBI
    input_file = get_most_recent_file(input_dir, pattern='*powerbi*.csv')
    
    if not input_file:
        logging.error(f"No se encontró ningún archivo PowerBI en {input_dir}")
        return
    
    logging.info(f"Usando el archivo: {os.path.basename(input_file)}")
    
    # Tiempo de carga de archivo
    t1 = time.perf_counter()
    
    try:
        # Leer el CSV con separador de punto y coma
        df = pd.read_csv(input_file, sep=';')
        logging.info(f"Archivo cargado correctamente: {len(df)} filas")
        
        # Convertir columna de fecha y filtrar
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
        df = df.dropna(subset=[DATE_COL])
        df = df[df[DATE_COL] >= pd.to_datetime('2014-01-01')]
        df = df.sort_values(DATE_COL)
        logging.info(f"Después de filtrar por fecha: {len(df)} filas")
        
        # Convertir valores con comas a puntos y luego a float
        for col in ['Valor_Real', 'Valor_Predicho', 'RMSE']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        
        # Tiempo preprocesamiento
        t2 = time.perf_counter()
        logging.info(f"Preprocesamiento completado en {t2-t1:.2f}s")
        
        # Agrupar por modelo y tipo de mercado
        combinaciones = df.groupby(['Modelo', 'Tipo_Mercado'])
        logging.info(f"Analizando {len(combinaciones)} combinaciones de modelo y mercado")
        
        resultados = []
        for (modelo, mercado), grupo in combinaciones:
            logging.info(f"Procesando: {modelo} - {mercado}")
            
            # Filtrar NaNs
            grupo = grupo.dropna(subset=['Valor_Real', 'Valor_Predicho'])
            
            real = grupo['Valor_Real'].values
            pred = grupo['Valor_Predicho'].values
            
            # Filtrar valores idénticos (evita divisiones por cero)
            mask = np.abs(real - pred) > 1e-6
            if sum(mask) == 0:
                logging.warning(f"No hay datos válidos para {modelo} - {mercado} (valores idénticos)")
                continue
                
            real = real[mask]
            pred = pred[mask]
            
            if len(real) == 0:
                logging.warning(f"No hay datos válidos para {modelo} - {mercado}")
                continue
            
            # Calcular todas las métricas
            metricas = calcular_todas_metricas(real, pred)
            
            # Añadir información de modelo y mercado
            resultado = {
                'Tipo_de_Mercado': mercado,
                'Modelo': modelo,
                **metricas
            }
            
            # Guardar CSV individual
            modelo_filename = modelo.replace(" ", "_").replace("/", "_")
            mercado_filename = mercado.replace(" ", "_").replace("/", "_")
            individual_file = os.path.join(output_dir, f"{mercado_filename}_{modelo_filename}_metricas.csv")
            pd.DataFrame([resultado]).to_csv(individual_file, index=False)
            
            # Guardar también como JSON para compatibilidad con pipeline de series temporales
            json_file = os.path.join(output_dir, f"{mercado_filename}_{modelo_filename}_metricas.json")
            with open(json_file, 'w') as f:
                json.dump(resultado, f, indent=4)
            
            # Generar visualización de comparación real vs predicho
            chart_path = os.path.join(output_charts, f"{mercado_filename}_{modelo_filename}_comparison.png")
            plot_real_vs_pred(
                grupo, 
                title=f"Comparación {mercado} - {modelo}",
                metrics=metricas,
                model_name=modelo,
                output_path=chart_path
            )
            logging.info(f"Gráfico comparativo guardado: {os.path.basename(chart_path)}")
            
            # Calcular métricas por subperíodos (trimestral y anual)
            for freq, freq_name in [('Q', 'trimestre'), ('A', 'año')]:
                df_subperiodos, periodo_label = calcular_metricas_por_subperiodos(
                    grupo, freq=freq
                )
                
                if not df_subperiodos.empty:
                    # Guardar CSV de subperíodos
                    subperiod_file = os.path.join(
                        subperiods_dir, 
                        f"{mercado_filename}_{modelo_filename}_por_{freq_name}.csv"
                    )
                    df_subperiodos.to_csv(subperiod_file, index=False)
                    
                    # Generar gráfico de evolución por subperíodo
                    for metric in ['RMSE', 'MAE', 'Hit_Direction']:
                        if metric in df_subperiodos.columns:
                            chart_path = os.path.join(
                                subperiods_dir,
                                f"{mercado_filename}_{modelo_filename}_{metric.lower()}_evolution.png"
                            )
                            
                            plt.figure(figsize=(10, 6))
                            plt.plot(df_subperiodos['Periodo'], df_subperiodos[metric], 
                                    marker='o', linestyle='-')
                            plt.title(f"Evolución de {metric} por {periodo_label} - {modelo} ({mercado})")
                            plt.xlabel(periodo_label)
                            plt.ylabel(metric)
                            plt.grid(True, alpha=0.3)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            plt.savefig(chart_path, dpi=300)
                            plt.close()
                            
                            logging.info(f"Gráfico de evolución {metric} guardado: {os.path.basename(chart_path)}")
                    
                    # Dashboard de métricas por subperíodo
                    dashboard_path = os.path.join(
                        subperiods_dir,
                        f"{mercado_filename}_{modelo_filename}_metrics_dashboard.png"
                    )
                    
                    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
                    metrics_to_plot = [
                        ('RMSE', 'RMSE por período', axs[0, 0]),
                        ('MAE', 'MAE por período', axs[0, 1]),
                        ('Hit_Direction', 'Hit Direction por período', axs[1, 0]),
                        ('R2', 'R² por período', axs[1, 1])
                    ]
                    
                    for metric, title, ax in metrics_to_plot:
                        if metric in df_subperiodos.columns:
                            ax.plot(df_subperiodos['Periodo'], df_subperiodos[metric], 
                                   marker='o', linestyle='-')
                            ax.set_title(title)
                            ax.set_xlabel(periodo_label)
                            ax.set_ylabel(metric)
                            ax.grid(True, alpha=0.3)
                            ax.tick_params(axis='x', rotation=45)
                    
                    plt.suptitle(f"Dashboard de Métricas por {periodo_label} - {modelo} ({mercado})")
                    plt.tight_layout()
                    plt.savefig(dashboard_path, dpi=300)
                    plt.close()
                    
                    logging.info(f"Dashboard guardado: {os.path.basename(dashboard_path)}")
            
            logging.info(f"Métricas guardadas para {modelo} - {mercado}")
            
            # Agregar al consolidado
            resultados.append(resultado)
        
        # Tiempo procesamiento individual
        t3 = time.perf_counter()
        logging.info(f"Procesamiento individual completado en {t3-t2:.2f}s")
        
        # Guardar archivo consolidado
        if resultados:
            consolidado_file = os.path.join(output_dir, "resultados_totales.csv")
            df_resultado = pd.DataFrame(resultados)
            df_resultado.to_csv(consolidado_file, index=False)
            
            # También como Excel para mejor visualización
            excel_consolidado = os.path.join(output_dir, "resultados_totales.xlsx")
            df_resultado.to_excel(excel_consolidado, index=False)
            
            # También como JSON para compatibilidad
            json_consolidado = os.path.join(output_dir, "resultados_totales.json")
            df_resultado.to_json(json_consolidado, orient='records', indent=4)
            
            logging.info(f"Resultados consolidados guardados en: {consolidado_file}")
            
            # Generar radar charts para comparación de modelos por tipo de mercado
            mercados = df_resultado['Tipo_de_Mercado'].unique()
            for mercado in mercados:
                df_mercado = df_resultado[df_resultado['Tipo_de_Mercado'] == mercado]
                
                # Convertir a formato para radar chart
                metrics_dict = {}
                metric_cols = ['RMSE', 'MAE', 'SMAPE', 'R2', 'Hit_Direction', 'Amplitud_Score', 'Fase_Score']
                for _, row in df_mercado.iterrows():
                    model = row['Modelo']
                    metrics = {col: row[col] for col in metric_cols if col in row and not pd.isna(row[col])}
                    metrics_dict[model] = metrics
                
                if len(metrics_dict) > 1:  # Solo si hay más de un modelo
                    radar_path = os.path.join(output_charts, f"{mercado.replace(' ', '_')}_radar_comparison.png")
                    plot_radar_metrics(
                        metrics_dict, 
                        title=f"Comparación de métricas - {mercado}",
                        output_path=radar_path
                    )
                    logging.info(f"Radar chart guardado: {os.path.basename(radar_path)}")
            
            # Generar gráficos consolidados de barras para RMSE, MAE, etc.
            main_metrics = ['RMSE', 'MAE', 'SMAPE', 'Hit_Direction']
            for metric in main_metrics:
                if metric in df_resultado.columns:
                    for mercado in mercados:
                        df_mercado = df_resultado[df_resultado['Tipo_de_Mercado'] == mercado]
                        df_sorted = df_mercado.sort_values(metric)
                        
                        chart_path = os.path.join(output_charts, f"{mercado.replace(' ', '_')}_{metric.lower()}_comparison.png")
                        plt.figure(figsize=(10, 6))
                        plt.bar(df_sorted['Modelo'], df_sorted[metric])
                        plt.title(f"Comparación de {metric} entre modelos - {mercado}")
                        plt.xlabel('Modelo')
                        plt.ylabel(metric)
                        plt.xticks(rotation=45, ha='right')
                        plt.grid(axis='y', alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(chart_path, dpi=300)
                        plt.close()
                        
                        logging.info(f"Gráfico de barras {metric} guardado: {os.path.basename(chart_path)}")
                        
            # Tiempo procesamiento consolidado
            t4 = time.perf_counter()
            logging.info(f"Procesamiento consolidado completado en {t4-t3:.2f}s")
            
            print(f"✅ Resultados exportados correctamente a: {output_dir}")
            print(f"✅ Visualizaciones generadas en: {output_charts}")
            print(f"✅ Análisis por subperíodos en: {subperiods_dir}")
        else:
            logging.warning("No se generaron resultados para guardar")
            
    except Exception as e:
        logging.error(f"Error al procesar el archivo: {e}", exc_info=True)
    
    # Tiempo total
    t_final = time.perf_counter()
    logging.info(f"Proceso completo terminado en {t_final-t0:.2f}s")

if __name__ == "__main__":
    main()