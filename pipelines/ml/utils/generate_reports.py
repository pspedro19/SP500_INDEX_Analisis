"""
Script para generar un reporte final consolidado de todo el pipeline en formato PDF.
Recopila gráficos, tablas de métricas, tiempos de ejecución y resultados de los modelos.
"""
import os
import sys
import json
import glob
import time
import pandas as pd
import numpy as np
import logging
from sp500_analysis.shared.logging.logger import configurar_logging
from datetime import datetime
from pathlib import Path

# Asegurar que podemos importar desde el directorio actual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar configuraciones centralizadas
from sp500_analysis.config.settings import settings
from pipelines.ml.config import ensure_directories

PROJECT_ROOT = settings.project_root
MODELS_DIR = settings.models_dir
RESULTS_DIR = settings.results_dir
METRICS_DIR = settings.metrics_dir
IMG_CHARTS = settings.img_charts_dir
METRICS_CHARTS = settings.metrics_charts_dir
REPORTS_DIR = settings.reports_dir
CSV_REPORTS = settings.csv_reports_dir

# Configuración de logging
log_file = os.path.join(PROJECT_ROOT, "logs", f"report_generation_{time.strftime('%Y%m%d_%H%M%S')}.log")
configurar_logging(log_file)

def find_files(directory, pattern):
    """
    Encuentra archivos que coinciden con un patrón en un directorio.
    
    Args:
        directory (str): Directorio donde buscar
        pattern (str): Patrón glob para archivos
        
    Returns:
        list: Lista de rutas a archivos
    """
    return glob.glob(os.path.join(directory, pattern))

def get_most_recent_file(directory, pattern='*.csv'):
    """
    Obtiene el archivo más reciente que coincide con un patrón.
    
    Args:
        directory (str): Directorio donde buscar
        pattern (str): Patrón glob para archivos
        
    Returns:
        str: Ruta al archivo más reciente
    """
    files = find_files(directory, pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def gather_metrics():
    """
    Recopila métricas de archivos CSV y JSON.
    
    Returns:
        dict: Métricas consolidadas
    """
    metrics = {}
    
    # Buscar archivo de resultados totales
    resultados_file = os.path.join(METRICS_DIR, "resultados_totales.csv")
    if os.path.exists(resultados_file):
        try:
            df_metrics = pd.read_csv(resultados_file)
            metrics['backtest'] = df_metrics.to_dict(orient='records')
            logging.info(f"Métricas de backtest cargadas: {len(df_metrics)} registros")
        except Exception as e:
            logging.error(f"Error al cargar resultados_totales.csv: {e}")
    
    # Buscar información del ensemble
    ensemble_info_file = os.path.join(RESULTS_DIR, "ensemble_info.json")
    if os.path.exists(ensemble_info_file):
        try:
            with open(ensemble_info_file, 'r') as f:
                ensemble_info = json.load(f)
            metrics['ensemble'] = ensemble_info
            logging.info("Información del ensemble cargada")
        except Exception as e:
            logging.error(f"Error al cargar información del ensemble: {e}")
    
    # Buscar métricas de entrenamiento
    training_file = os.path.join(RESULTS_DIR, "resumen_metricas.csv")
    if os.path.exists(training_file):
        try:
            df_training = pd.read_csv(training_file)
            metrics['training'] = df_training.to_dict(orient='records')
            logging.info(f"Métricas de entrenamiento cargadas: {len(df_training)} registros")
        except Exception as e:
            logging.error(f"Error al cargar resumen_metricas.csv: {e}")
    
    # Buscar tiempos de ejecución
    timings_file = os.path.join(REPORTS_DIR, "pipeline_timings.json")
    if os.path.exists(timings_file):
        try:
            with open(timings_file, 'r') as f:
                timings = json.load(f)
            metrics['timings'] = timings
            logging.info("Tiempos de ejecución cargados")
        except Exception as e:
            logging.error(f"Error al cargar tiempos de ejecución: {e}")
    
    # Buscar información de inferencia
    inference_file = os.path.join(RESULTS_DIR, "predictions_api.json")
    if os.path.exists(inference_file):
        try:
            with open(inference_file, 'r') as f:
                inference = json.load(f)
            metrics['inference'] = inference
            logging.info("Resultados de inferencia cargados")
        except Exception as e:
            logging.error(f"Error al cargar resultados de inferencia: {e}")
    
    return metrics

def gather_visualizations():
    """
    Recopila todas las visualizaciones disponibles.
    
    Returns:
        dict: Visualizaciones agrupadas por categoría
    """
    visualizations = {
        'models': [],
        'backtest': [],
        'inference': [],
        'ensemble': [],
        'timelines': [],
        'subperiods': []
    }
    
    # Charts de modelos
    for img in find_files(IMG_CHARTS, "*.png"):
        filename = os.path.basename(img)
        
        # Clasificar por nombre
        if "ensemble" in filename.lower():
            visualizations['ensemble'].append(img)
        elif "timeline" in filename.lower():
            visualizations['timelines'].append(img)
        elif "forecast" in filename.lower():
            visualizations['inference'].append(img)
        elif "comparison" in filename.lower():
            visualizations['backtest'].append(img)
        else:
            visualizations['models'].append(img)
    
    # Metrics charts
    for img in find_files(METRICS_CHARTS, "*.png"):
        filename = os.path.basename(img)
        
        if "radar" in filename.lower():
            visualizations['backtest'].append(img)
        elif "comparison" in filename.lower():
            visualizations['backtest'].append(img)
        else:
            visualizations['backtest'].append(img)
    
    # Subperiods
    subperiods_dir = os.path.join(METRICS_CHARTS, "subperiods")
    if os.path.exists(subperiods_dir):
        for img in find_files(subperiods_dir, "*.png"):
            visualizations['subperiods'].append(img)
    
    # Contar imágenes encontradas
    total_images = sum(len(v) for v in visualizations.values())
    logging.info(f"Total de visualizaciones encontradas: {total_images}")
    for cat, images in visualizations.items():
        logging.info(f"  - {cat}: {len(images)} imágenes")
    
    return visualizations

def generate_html_report(metrics, visualizations):
    """
    Genera un reporte HTML completo.
    
    Args:
        metrics (dict): Métricas recopiladas
        visualizations (dict): Visualizaciones recopiladas
        
    Returns:
        str: Ruta al archivo HTML generado
    """
    from jinja2 import Template
    
    # Plantilla HTML
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reporte Final de Pipeline ML</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
            h1, h2, h3 { color: #333; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .section { margin-bottom: 40px; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 14px; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .gallery { display: flex; flex-wrap: wrap; gap: 20px; }
            .gallery-item { margin-bottom: 20px; max-width: 500px; }
            .gallery-item img { max-width: 100%; border: 1px solid #ddd; box-shadow: 0 0 5px rgba(0,0,0,0.1); }
            .highlight { background-color: #f9f9f9; padding: 15px; border-radius: 5px; }
            .metrics-table { font-size: 14px; }
            .metrics-table th, .metrics-table td { padding: 5px; }
            .caption { font-style: italic; margin-top: 5px; text-align: center; }
            .page-break { page-break-after: always; }
            @media print {
                .gallery-item { page-break-inside: avoid; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Reporte Final de Pipeline ML</h1>
                <p>Generado el {{ date }}</p>
            </div>
            
            <div class="section page-break">
                <h2>Resumen Ejecutivo</h2>
                <div class="highlight">
                    <h3>Objetivos del Modelo</h3>
                    <p>Pipeline de pronóstico para {{ market_type }} con horizonte de {{ forecast_period }}.</p>
                    
                    <h3>Rendimiento del Modelo</h3>
                    {% if best_model %}
                    <p>El mejor modelo es <strong>{{ best_model }}</strong> con RMSE = {{ best_rmse }}.</p>
                    {% endif %}
                    
                    {% if ensemble_info %}
                    <p>El ensemble combina {{ ensemble_models_count }} modelos y logra un RMSE de {{ ensemble_rmse }}.</p>
                    {% endif %}
                    
                    <h3>Métricas Clave</h3>
                    <table class="metrics-table">
                        <tr>
                            <th>Modelo</th>
                            <th>RMSE</th>
                            <th>MAE</th>
                            <th>R²</th>
                            <th>Hit Direction</th>
                        </tr>
                        {% for model in top_models %}
                        <tr>
                            <td>{{ model.Modelo }}</td>
                            <td>{{ model.RMSE|round(4) }}</td>
                            <td>{{ model.MAE|round(4) }}</td>
                            <td>{{ model.R2|round(4) }}</td>
                            <td>{{ model.Hit_Direction|round(2) }}%</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
            </div>

            <div class="section page-break">
                <h2>Inferencia y Pronóstico</h2>
                <p>Resultados de la inferencia más reciente ({{ inference_date }}):</p>
                
                <table>
                    <tr>
                        <th>Modelo</th>
                        <th>Predicción</th>
                        <th>Fecha Objetivo</th>
                    </tr>
                    {% for model, data in inference.items() %}
                    <tr>
                        <td>{{ model }}</td>
                        <td>{{ data.prediction|round(4) }}</td>
                        <td>{{ data.target_date }}</td>
                    </tr>
                    {% endfor %}
                </table>
                
                <h3>Visualizaciones de Pronóstico</h3>
                <div class="gallery">
                    {% for img in visualizations.inference %}
                    <div class="gallery-item">
                        <img src="{{ img }}" alt="Forecast">
                        <div class="caption">{{ img.split('/')[-1] }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="section page-break">
                <h2>Desempeño del Ensemble</h2>
                {% if ensemble_info %}
                <p>Modelos seleccionados: {{ ensemble_info.selected_models|join(', ') }}</p>
                <p>Métricas: RMSE={{ ensemble_info.metrics.RMSE|round(4) }}, R²={{ ensemble_info.metrics.R2|round(4) }}</p>
                
                <div class="gallery">
                    {% for img in visualizations.ensemble %}
                    <div class="gallery-item">
                        <img src="{{ img }}" alt="Ensemble Chart">
                        <div class="caption">{{ img.split('/')[-1] }}</div>
                    </div>
                    {% endfor %}
                </div>
                {% else %}
                <p>No se encontró información del ensemble.</p>
                {% endif %}
            </div>
            
            <div class="section page-break">
                <h2>Backtest y Evaluación</h2>
                <p>Resultados de evaluación para diferentes modelos:</p>
                
                <div class="gallery">
                    {% for img in visualizations.backtest[:4] %}
                    <div class="gallery-item">
                        <img src="{{ img }}" alt="Backtest Chart">
                        <div class="caption">{{ img.split('/')[-1] }}</div>
                    </div>
                    {% endfor %}
                </div>
                
                <h3>Métricas de Backtest</h3>
                <table>
                    <tr>
                        <th>Modelo</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                        <th>SMAPE</th>
                        <th>R²</th>
                        <th>Hit Direction</th>
                    </tr>
                    {% for model in metrics.backtest[:10] %}
                    <tr>
                        <td>{{ model.Modelo }}</td>
                        <td>{{ model.RMSE|round(4) }}</td>
                        <td>{{ model.MAE|round(4) }}</td>
                        <td>{{ model.SMAPE|round(2) }}%</td>
                        <td>{{ model.R2|round(4) }}</td>
                        <td>{{ model.Hit_Direction|round(2) }}%</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="section page-break">
                <h2>Análisis por Subperíodos</h2>
                <p>Desempeño de los modelos en diferentes períodos:</p>
                
                <div class="gallery">
                    {% for img in visualizations.subperiods[:6] %}
                    <div class="gallery-item">
                        <img src="{{ img }}" alt="Subperiod Analysis">
                        <div class="caption">{{ img.split('/')[-1] }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="section page-break">
                <h2>Tiempos de Ejecución</h2>
                <p>Tiempo total del pipeline: {{ total_time }}</p>
                
                <table>
                    <tr>
                        <th>Paso</th>
                        <th>Tiempo (segundos)</th>
                    </tr>
                    {% for step, time in timings.items() %}
                    <tr>
                        <td>{{ step }}</td>
                        <td>{{ time|round(2) }}</td>
                    </tr>
                    {% endfor %}
                </table>
                
                {% if visualizations.timelines %}
                <div class="gallery">
                    <div class="gallery-item">
                        <img src="{{ visualizations.timelines[0] }}" alt="Timeline">
                        <div class="caption">Línea de tiempo del pipeline</div>
                    </div>
                </div>
                {% endif %}
            </div>
        </div>
    </body>
    </html>
    """
    
    # Preparar datos para la plantilla
    template_data = {
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'market_type': "S&P500",  # Ajustar según sea necesario
        'forecast_period': "1 mes",  # Ajustar según sea necesario
        'visualizations': visualizations,
        'inference': {},
        'ensemble_info': {},
        'top_models': [],
        'best_model': "",
        'best_rmse': 0,
        'ensemble_models_count': 0,
        'ensemble_rmse': 0,
        'inference_date': datetime.now().strftime("%Y-%m-%d"),
        'timings': {},
        'total_time': "0s"
    }
    
    # Extraer datos de las métricas recopiladas
    if 'backtest' in metrics and metrics['backtest']:
        # Top modelos por RMSE
        df_backtest = pd.DataFrame(metrics['backtest'])
        df_backtest = df_backtest.sort_values('RMSE')
        template_data['top_models'] = df_backtest.to_dict('records')[:5]
        
        # Mejor modelo
        best_model_row = df_backtest.iloc[0]
        template_data['best_model'] = best_model_row['Modelo']
        template_data['best_rmse'] = round(best_model_row['RMSE'], 4)
    
    if 'ensemble' in metrics:
        template_data['ensemble_info'] = metrics['ensemble']
        if 'selected_models' in metrics['ensemble']:
            template_data['ensemble_models_count'] = len(metrics['ensemble']['selected_models'])
        if 'metrics' in metrics['ensemble'] and 'RMSE' in metrics['ensemble']['metrics']:
            template_data['ensemble_rmse'] = round(metrics['ensemble']['metrics']['RMSE'], 4)
    
    if 'inference' in metrics:
        template_data['inference'] = metrics['inference']
        if metrics['inference']:
            first_model = list(metrics['inference'].values())[0]
            template_data['inference_date'] = first_model.get('date_inference', 'N/A')
    
    if 'timings' in metrics:
        template_data['timings'] = metrics['timings']
        total_seconds = sum(metrics['timings'].values())
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            template_data['total_time'] = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        elif minutes > 0:
            template_data['total_time'] = f"{int(minutes)}m {seconds:.2f}s"
        else:
            template_data['total_time'] = f"{seconds:.2f}s"
    
    # Crear el reporte HTML
    template = Template(template_str)
    html_content = template.render(**template_data)
    
    # Guardar archivo HTML
    html_path = os.path.join(REPORTS_DIR, "reporte_final.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"Reporte HTML generado: {html_path}")
    return html_path

def convert_html_to_pdf(html_path):
    """
    Convierte un archivo HTML a PDF usando weasyprint.
    
    Args:
        html_path (str): Ruta al archivo HTML
        
    Returns:
        str: Ruta al archivo PDF generado
    """
    try:
        from weasyprint import HTML
        
        # Determinar ruta de salida
        pdf_path = html_path.replace('.html', '.pdf')
        
        # Convertir a PDF
        HTML(html_path).write_pdf(pdf_path)
        logging.info(f"PDF generado correctamente: {pdf_path}")
        return pdf_path
    except ImportError:
        logging.warning("Módulo weasyprint no disponible. No se pudo generar PDF.")
        logging.warning("Para generar PDFs, instala weasyprint: pip install weasyprint")
        return None
    except Exception as e:
        logging.error(f"Error al generar PDF: {e}")
        return None

def main():
    """
    Función principal para generar el reporte final.
    """
    logging.info("Iniciando generación de reporte final...")
    start_time = time.time()
    
    # Asegurar que los directorios existen
    ensure_directories()
    
    # Recopilar métricas y visualizaciones
    metrics = gather_metrics()
    visualizations = gather_visualizations()
    
    # Generar reporte HTML
    html_path = generate_html_report(metrics, visualizations)
    
    # Convertir a PDF
    pdf_path = convert_html_to_pdf(html_path)
    
    # Calcular tiempo total
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Reporte generado en {elapsed_time:.2f} segundos")
    
    if pdf_path:

if __name__ == "__main__":
    main()