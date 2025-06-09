#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Orquestador principal del pipeline ML.
Ejecuta todos los pasos en secuencia y genera un reporte final.
"""
import os
import sys
import time
import json
import logging
import importlib
import subprocess
from datetime import datetime
from pathlib import Path

# Asegurar que podemos importar módulos desde el directorio actual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config.base import ProjectConfig

config = ProjectConfig.from_env()

ROOT = config.project_root
DATA_RAW = config.data_raw
DATA_PREP = config.data_prep
PROCESSED_DIR = config.processed_dir
TRAINING_DIR = config.training_dir
RESULTS_DIR = config.results_dir
METRICS_DIR = config.metrics_dir
LOG_DIR = config.log_dir
REPORTS_DIR = config.reports_dir
IMG_CHARTS = config.img_charts_dir
METRICS_CHARTS = config.metrics_charts_dir
CSV_REPORTS = config.csv_reports_dir
ensure_directories = config.ensure_dirs


# Crear directorio para logs si no existe
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# Configurar logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"pipeline_run_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def run_step(step_module, step_name=None):
    """
    Ejecuta un paso del pipeline y registra el tiempo de ejecución.
    
    Args:
        step_module (str): Ruta al módulo Python a ejecutar
        step_name (str): Nombre descriptivo para el paso (opcional)
    
    Returns:
        tuple: (éxito, tiempo de ejecución, errores)
    """
    if step_name is None:
        step_name = os.path.basename(step_module).replace('.py', '')
    
    logging.info(f"Iniciando {step_name}...")
    start_time = time.time()
    
    try:
        # Intentar importar y ejecutar como módulo
        if step_module.endswith('.py'):
            module_path = step_module.replace('/', '.').replace('\\', '.').replace('.py', '')
            module = importlib.import_module(module_path)
            if hasattr(module, 'main'):
                module.main()
            else:
                logging.warning(f"El módulo {module_path} no tiene función main(), ejecutando como script...")
                result = subprocess.run([sys.executable, step_module], check=True)
                if result.returncode != 0:
                    raise Exception(f"Script terminó con código {result.returncode}")
        else:
            # Ejecutar como script directamente
            result = subprocess.run([sys.executable, step_module], check=True)
            if result.returncode != 0:
                raise Exception(f"Script terminó con código {result.returncode}")
        
        elapsed_time = time.time() - start_time
        logging.info(f"✅ {step_name} completado en {elapsed_time:.2f}s")
        return True, elapsed_time, None
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"❌ Error en {step_name}: {str(e)}")
        return False, elapsed_time, str(e)

def generate_timeline_chart(timings):
    """
    Genera un gráfico de línea de tiempo con los tiempos de ejecución.
    
    Args:
        timings (dict): Diccionario con {nombre_paso: tiempo_segundos}
    
    Returns:
        str: Ruta al archivo del gráfico generado
    """
    try:
        from pipelines.ml.utils.plots import plot_pipeline_timeline
        chart_path = Path(REPORTS_DIR) / "pipeline_timeline.png"
        plot_pipeline_timeline(timings, chart_path)
        logging.info(f"Gráfico de timeline generado: {chart_path}")
        return str(chart_path)
    except Exception as e:
        logging.error(f"Error al generar timeline: {e}")
        return None

def generate_html_report(timings, results, start_time):
    """
    Genera un informe HTML simple con los resultados del pipeline.
    
    Args:
        timings (dict): Diccionario con tiempos de ejecución
        results (dict): Resultados de cada paso
        start_time (float): Tiempo de inicio del pipeline
    
    Returns:
        str: Ruta al archivo HTML generado
    """
    try:
        from jinja2 import Template
        
        # Preparar datos para la plantilla
        total_time = sum(timings.values())
        end_time = time.time()
        pipeline_time = end_time - start_time
        
        # Encontrar imágenes generadas
        chart_files = []
        for chart_dir in [IMG_CHARTS_DIR, METRICS_CHARTS_DIR]:
            if os.path.exists(chart_dir):
                chart_files.extend([
                    str(chart_dir / f) for f in os.listdir(chart_dir) 
                    if f.endswith(('.png', '.jpg', '.svg'))
                ])
        
        # Encontrar archivos CSV generados
        csv_files = []
        for csv_dir in [CSV_REPORTS, RESULTS_DIR, METRICS_DIR]:
            if os.path.exists(csv_dir):
                csv_files.extend([
                    str(csv_dir / f) for f in os.listdir(csv_dir) 
                    if f.endswith(('.csv', '.xlsx'))
                ])
        
        # Plantilla HTML
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Pipeline ML Report</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
                h1, h2, h3 { color: #333; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 30px; }
                .section { margin-bottom: 30px; }
                table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
                th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .success { color: green; }
                .error { color: red; }
                .gallery { display: flex; flex-wrap: wrap; gap: 20px; }
                .gallery img { max-width: 350px; border: 1px solid #ddd; box-shadow: 0 0 5px rgba(0,0,0,0.1); }
                .files-list { list-style-type: none; padding: 0; }
                .files-list li { margin-bottom: 5px; }
                .highlight { background-color: #f9f9f9; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Reporte de Ejecución del Pipeline ML</h1>
                    <p>Ejecutado el {{ date }} | Duración total: {{ pipeline_time }} segundos</p>
                </div>
                
                <div class="section">
                    <h2>Resumen de Ejecución</h2>
                    <table>
                        <tr>
                            <th>Paso</th>
                            <th>Estado</th>
                            <th>Tiempo (seg)</th>
                            <th>Detalles</th>
                        </tr>
                        {% for step, data in results.items() %}
                        <tr>
                            <td>{{ step }}</td>
                            <td class="{{ 'success' if data.success else 'error' }}">
                                {{ "✅ Exitoso" if data.success else "❌ Error" }}
                            </td>
                            <td>{{ "%.2f"|format(data.time) }}</td>
                            <td>{{ data.error if data.error else "" }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                <div class="section">
                    <h2>Línea de Tiempo</h2>
                    <div class="highlight">
                        {% if timeline_path %}
                        <img src="{{ timeline_path }}" alt="Timeline" style="max-width:100%;">
                        {% else %}
                        <p>No se pudo generar la línea de tiempo.</p>
                        {% endif %}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Visualizaciones Generadas</h2>
                    <div class="gallery">
                        {% for chart in charts %}
                        <div>
                            <img src="{{ chart }}" alt="Chart">
                            <p>{{ chart.split('/')[-1] }}</p>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Archivos de Resultados</h2>
                    <ul class="files-list">
                        {% for file in csv_files %}
                        <li>{{ file }}</li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        template = Template(template_str)
        html_content = template.render(
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pipeline_time=f"{pipeline_time:.2f}",
            results=results,
            timeline_path=generate_timeline_chart(timings),
            charts=chart_files,
            csv_files=csv_files
        )
        
        # Guardar archivo HTML
        report_path = Path(REPORTS_DIR) / "pipeline_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info(f"Reporte HTML generado: {report_path}")
        return str(report_path)
    
    except Exception as e:
        logging.error(f"Error al generar reporte HTML: {e}")
        return None

def main():
    """Función principal que ejecuta todo el pipeline."""
    # Asegurar que existen todos los directorios necesarios
    ensure_directories()
    
    # Registrar tiempo de inicio global
    pipeline_start_time = time.time()
    
    # Definir los pasos del pipeline
    pipeline_steps = [
        ("Paso 1: Merge de Excels", "src/pipelines/ml/step_1_merge_excels.py"),
        ("Paso 2: Generación de Categorías", "src/pipelines/ml/step_2_generate_categories.py"),
        ("Paso 3: Limpieza de Columnas", "src/pipelines/ml/step_3_clean_columns.py"),
        ("Paso 4: Transformación de Features", "src/pipelines/ml/step_4_transform_features.py"),
        ("Paso 5: Eliminación de Relaciones", "src/pipelines/ml/step_5_remove_relations.py"),
        ("Paso 6: Selección FPI", "src/pipelines/ml/step_6_fpi_selection.py"),
        ("Paso 7: Entrenamiento de Modelos", "src/pipelines/ml/step_7_0_train_models.py"),
        ("Paso 7.5: Ensamblado", "src/pipelines/ml/step_7_5_ensemble.py"),
        ("Paso 8: Preparación de Salida", "src/pipelines/ml/step_8_prepare_output.py"),
        ("Paso 9: Backtest", "src/pipelines/ml/step_9_backtest.py"),
        ("Paso 10: Inferencia", "src/pipelines/ml/step_10_inference.py")
    ]
    
    # Almacenar resultados y tiempos
    results = {}
    timings = {}
    
    # Ejecutar cada paso del pipeline
    for step_name, step_module in pipeline_steps:
        success, elapsed_time, error = run_step(step_module, step_name)
        results[step_name] = {
            "success": success,
            "time": elapsed_time,
            "error": error
        }
        timings[step_name] = elapsed_time
        
        # Si el paso falló y es crítico, detener el pipeline
        if not success and step_name in ["Paso 1", "Paso 2", "Paso 3", "Paso 4", "Paso 5", "Paso 6"]:
            logging.error(f"Paso crítico {step_name} falló. Deteniendo el pipeline.")
            break
    
    # Guardar tiempos en JSON para referencia
    timings_file = Path(REPORTS_DIR) / "pipeline_timings.json"
    with open(timings_file, 'w') as f:
        json.dump(timings, f, indent=4)
    logging.info(f"Tiempos del pipeline guardados en: {timings_file}")
    
    # Generar reporte HTML
    report_path = generate_html_report(timings, results, pipeline_start_time)
    if report_path:
        print(f"\n✅ Pipeline completado. Reporte generado en: {report_path}\n")
    else:
        print("\n⚠️ Pipeline completado pero no se pudo generar el reporte.\n")
    
    # Calcular y mostrar tiempo total
    total_time = time.time() - pipeline_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s" if hours > 0 else f"{int(minutes)}m {seconds:.2f}s"
    
    logging.info(f"Pipeline completado en {time_str}")
    print(f"⏱️ Tiempo total de ejecución: {time_str}")
    
    # Resumen de éxito/error
    success_count = sum(1 for r in results.values() if r["success"])
    error_count = len(results) - success_count
    print(f"📊 Resumen: {success_count} pasos exitosos, {error_count} con errores.")
    
    # Mostrar los pasos con error
    if error_count > 0:
        print("\nPasos con errores:")
        for step, data in results.items():
            if not data["success"]:
                print(f"  - {step}: {data['error']}")
    
    return results

if __name__ == "__main__":
    main()