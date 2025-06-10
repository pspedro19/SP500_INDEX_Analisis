import os
import joblib
import numpy as np
import pandas as pd
import logging
import glob
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Importar configuraciones
from config import (
    PROJECT_ROOT, MODELS_DIR, RESULTS_DIR, TRAINING_DIR, IMG_CHARTS,
    LOCAL_REFINEMENT_DAYS, FORECAST_HORIZON_1MONTH, 
    TRAIN_TEST_SPLIT_RATIO, DATE_COL, RANDOM_SEED, ensure_directories
)

# Importar funciones de visualización
from ml.utils.plots import plot_real_vs_pred

# ------------------------------
# CONFIGURACIÓN DE LOGGING
# ------------------------------
log_file = os.path.join(PROJECT_ROOT, "logs", f"ensemble_{time.strftime('%Y%m%d_%H%M%S')}.log")
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

class GreedyEnsembleRegressor:
    """
    Implementación de ensamble greedy similar al pipeline de series temporales.
    Añade iterativamente los modelos que más mejoran la métrica.
    """
    def __init__(self, models, metric='rmse', model_names=None):
        self.models = models
        self.model_names = model_names if model_names else [f"Model_{i}" for i in range(len(models))]
        self.selected_models = []
        self.selected_indices = []
        self.weights = []
        self.metric = metric
        self.best_score = float('inf')
        self.model_contributions = {}  # Para almacenar contribución de cada modelo
    
    def fit(self, X, y):
        """
        Selecciona greedily los mejores modelos.
        """
        t0 = time.perf_counter()
        # Temporal split para validación
        n = len(X)
        val_size = int(n * 0.2)
        X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
        y_train, y_val = y.iloc[:-val_size], y.iloc[-val_size:]
        
        logging.info(f"Entrenando ensemble usando {len(self.models)} modelos base")
        logging.info(f"División de datos: train={len(X_train)}, validación={len(X_val)}")
        
        # Entrena todos los modelos
        all_preds = []
        all_scores = []
        model_training_times = []
        
        for i, model in enumerate(self.models):
            model_start = time.perf_counter()
            model_name = self.model_names[i]
            logging.info(f"Entrenando modelo base {model_name} ({i+1}/{len(self.models)})")
            
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            all_preds.append(preds)
            
            # Evaluar modelo individual
            score = np.sqrt(mean_squared_error(y_val, preds))
            all_scores.append(score)
            
            model_end = time.perf_counter()
            model_time = model_end - model_start
            model_training_times.append(model_time)
            
            logging.info(f"Modelo {model_name} entrenado en {model_time:.2f}s - RMSE: {score:.4f}")
        
        # Guardar métricas individuales
        self.individual_scores = dict(zip(self.model_names, all_scores))
        logging.info(f"Métricas individuales: {self.individual_scores}")
        
        # Selección greedy
        remaining_models = list(range(len(self.models)))
        selected = []
        selected_names = []
        score_history = []
        
        while remaining_models and len(selected) < len(self.models):
            best_score = float('inf')
            best_idx = -1
            
            for idx in remaining_models:
                # Añadir este modelo a los seleccionados
                current = selected + [idx]
                
                # Calcular predicción promedio
                ensemble_pred = np.mean([all_preds[i] for i in current], axis=0)
                
                # Evaluar
                score = np.sqrt(mean_squared_error(y_val, ensemble_pred))
                
                if score < best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx != -1:
                selected.append(best_idx)
                selected_names.append(self.model_names[best_idx])
                remaining_models.remove(best_idx)
                score_history.append(best_score)
                logging.info(f"Añadido modelo {self.model_names[best_idx]} - Score: {best_score:.4f}")
            
            # Si no hay mejora, terminar
            if len(selected) > 1 and best_score >= self.best_score:
                logging.info(f"No hay mejora, deteniendo selección en {len(selected)} modelos")
                break
            
            self.best_score = best_score
        
        # Guardar modelos seleccionados
        self.selected_indices = selected
        self.selected_models = [self.models[i] for i in selected]
        self.selected_names = selected_names
        
        # Reentrenar modelos seleccionados en todos los datos
        for i, model in enumerate(self.selected_models):
            model.fit(X, y)
            logging.info(f"Modelo {self.selected_names[i]} reentrenado con todos los datos")
        
        # Calcular métricas finales
        ensemble_preds = self.predict(X_val)
        final_rmse = np.sqrt(mean_squared_error(y_val, ensemble_preds))
        final_mae = mean_absolute_error(y_val, ensemble_preds)
        final_r2 = r2_score(y_val, ensemble_preds)
        
        self.metrics = {
            'RMSE': final_rmse,
            'MAE': final_mae,
            'R2': final_r2,
            'Num_Models': len(self.selected_models)
        }
        
        logging.info(f"Ensemble final: {len(self.selected_models)}/{len(self.models)} modelos")
        logging.info(f"Métricas finales: RMSE={final_rmse:.4f}, MAE={final_mae:.4f}, R2={final_r2:.4f}")
        
        # Visualizar evolución del score
        self.score_history = score_history
        
        t1 = time.perf_counter()
        logging.info(f"Tiempo total de entrenamiento del ensemble: {t1-t0:.2f}s")
        
        return self
    
    def predict(self, X):
        """
        Predice usando el promedio de modelos seleccionados.
        """
        if not self.selected_models:
            raise ValueError("El ensamble debe ser entrenado primero")
        
        predictions = np.array([model.predict(X) for model in self.selected_models])
        return np.mean(predictions, axis=0)
    
    def plot_score_evolution(self, output_path=None):
        """
        Genera un gráfico que muestra la evolución del score al añadir modelos.
        
        Args:
            output_path (str): Ruta donde guardar el gráfico
            
        Returns:
            matplotlib.figure.Figure: Figura generada
        """
        if not hasattr(self, 'score_history'):
            logging.warning("No hay historial de scores para visualizar")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(1, len(self.score_history) + 1)
        ax.plot(x, self.score_history, marker='o', linestyle='-')
        
        # Añadir etiquetas con nombres de modelos
        for i, (score, name) in enumerate(zip(self.score_history, self.selected_names)):
            ax.annotate(
                name, 
                xy=(i + 1, score),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        ax.set_xlabel('Número de modelos')
        ax.set_ylabel('RMSE')
        ax.set_title('Evolución del RMSE al añadir modelos al ensemble')
        ax.grid(True, alpha=0.3)
        
        # Guardar o mostrar
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Gráfico de evolución guardado en: {output_path}")
        
        return fig
    
    def plot_model_comparison(self, output_path=None):
        """
        Genera un gráfico que compara los modelos individuales vs el ensemble.
        
        Args:
            output_path (str): Ruta donde guardar el gráfico
            
        Returns:
            matplotlib.figure.Figure: Figura generada
        """
        if not hasattr(self, 'individual_scores'):
            logging.warning("No hay scores individuales para visualizar")
            return None
        
        # Agregar el ensemble a los scores
        all_scores = self.individual_scores.copy()
        all_scores['Ensemble'] = self.best_score
        
        # Ordenar por score
        sorted_scores = {k: v for k, v in sorted(all_scores.items(), key=lambda item: item[1])}
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Códigos de colores para diferenciar modelos seleccionados vs no seleccionados
        colors = []
        for model in sorted_scores.keys():
            if model == 'Ensemble':
                colors.append('green')
            elif model in self.selected_names:
                colors.append('blue')
            else:
                colors.append('gray')
        
        # Crear barras
        bars = ax.bar(sorted_scores.keys(), sorted_scores.values(), color=colors)
        
        # Añadir valores
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 puntos de desplazamiento vertical
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8
            )
        
        ax.set_xlabel('Modelo')
        ax.set_ylabel('RMSE')
        ax.set_title('Comparación RMSE: Modelos individuales vs Ensemble')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        # Añadir leyenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Ensemble'),
            Patch(facecolor='blue', label='Seleccionado'),
            Patch(facecolor='gray', label='No seleccionado')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Guardar o mostrar
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Gráfico comparativo guardado en: {output_path}")
        
        return fig

def visualize_ensemble_performance(ensemble, X, y, output_dir=None):
    """
    Genera visualizaciones para evaluar el desempeño del ensemble.
    
    Args:
        ensemble: Modelo de ensemble entrenado
        X: Features para predicción
        y: Valores reales
        output_dir: Directorio donde guardar las visualizaciones
        
    Returns:
        list: Rutas a las visualizaciones generadas
    """
    if output_dir is None:
        output_dir = IMG_CHARTS
    
    os.makedirs(output_dir, exist_ok=True)
    visualization_paths = []
    
    # 1. Evolución del score
    score_path = os.path.join(output_dir, "ensemble_score_evolution.png")
    ensemble.plot_score_evolution(score_path)
    visualization_paths.append(score_path)
    
    # 2. Comparación de modelos
    comparison_path = os.path.join(output_dir, "ensemble_model_comparison.png")
    ensemble.plot_model_comparison(comparison_path)
    visualization_paths.append(comparison_path)
    
    # 3. Predicciones del ensemble vs valores reales
    ensemble_predictions = ensemble.predict(X)
    
    # Crear DataFrame para visualización
    df_pred = pd.DataFrame({
        'date': X.index if hasattr(X, 'index') else range(len(X)),
        'Valor_Real': y,
        'Valor_Predicho': ensemble_predictions,
        'Modelo': 'Ensemble',
        'Periodo': 'Validation'
    })
    
    # Calcular métricas
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y, ensemble_predictions)),
        'MAE': mean_absolute_error(y, ensemble_predictions),
        'R2': r2_score(y, ensemble_predictions),
        'Models_Used': len(ensemble.selected_models)
    }
    
    # Generar visualización
    predictions_path = os.path.join(output_dir, "ensemble_predictions.png")
    plot_real_vs_pred(
        df_pred,
        title="Predicciones del Ensemble vs Valores Reales",
        metrics=metrics,
        model_name="Ensemble",
        output_path=predictions_path
    )
    visualization_paths.append(predictions_path)
    
    return visualization_paths

def main():
    """
    Función principal para crear y evaluar el ensemble.
    
    Proceso:
    1. Cargar modelos entrenados
    2. Cargar datos para evaluación
    3. Crear y entrenar el ensemble
    4. Generar visualizaciones
    5. Guardar el ensemble
    """
    logging.info("Iniciando creación de ensamble greedy...")
    start_time = time.perf_counter()
    
    # Asegurar que los directorios existan
    ensure_directories()
    
    # Cargar modelos entrenados
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    models = []
    model_names = []
    
    for model_file in model_files:
        model_name = os.path.basename(model_file).replace('.pkl', '')
        
        # Filtrar modelos - excluir ensemble y modelos temporales
        if "ensemble" in model_name.lower() or model_name.startswith('.'):
            continue
            
        try:
            model = joblib.load(model_file)
            models.append(model)
            model_names.append(model_name)
            logging.info(f"Modelo cargado: {model_name}")
        except Exception as e:
            logging.error(f"Error al cargar {model_file}: {e}")
    
    if not models:
        logging.error("No se encontraron modelos para crear el ensamble.")
        return None
    
    logging.info(f"Se cargaron {len(models)} modelos para el ensamble.")
    
    # Cargar datos
    input_file = glob.glob(os.path.join(TRAINING_DIR, "*FPI.xlsx"))
    if not input_file:
        logging.error("No se encontró archivo de training con features FPI.")
        return None
    
    input_file = max(input_file, key=os.path.getmtime)
    logging.info(f"Usando datos: {os.path.basename(input_file)}")
    
    try:
        df = pd.read_excel(input_file)
        logging.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    except Exception as e:
        logging.error(f"Error al cargar datos: {e}")
        return None
    
    # Procesar datos
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)
    
    # Separar features y target
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col, DATE_COL])
    y = df[target_col]
    
    # Manejar índices de fecha
    X.index = df[DATE_COL]
    
    # Crear y entrenar ensamble
    ensemble = GreedyEnsembleRegressor(models, model_names=model_names)
    ensemble.fit(X, y)
    
    # Generar visualizaciones
    visualization_paths = visualize_ensemble_performance(ensemble, X, y)
    
    # Guardar ensamble
    ensemble_path = os.path.join(MODELS_DIR, "ensemble_greedy.pkl")
    joblib.dump(ensemble, ensemble_path)
    
    # Guardar información sobre el ensemble
    ensemble_info = {
        'selected_models': ensemble.selected_names,
        'metrics': ensemble.metrics,
        'score_history': ensemble.score_history,
        'individual_scores': ensemble.individual_scores,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    ensemble_info_path = os.path.join(RESULTS_DIR, "ensemble_info.json")
    with open(ensemble_info_path, 'w') as f:
        json.dump(ensemble_info, f, indent=4)
    
    end_time = time.perf_counter()
    logging.info(f"Ensamble guardado en {ensemble_path}")
    logging.info(f"Información del ensamble guardada en {ensemble_info_path}")
    logging.info(f"Modelos seleccionados: {len(ensemble.selected_models)}/{len(models)}")
    logging.info(f"Visualizaciones generadas: {', '.join(visualization_paths)}")
    logging.info(f"Proceso completado en {end_time - start_time:.2f}s")
    
    print(f"✅ Ensemble creado con {len(ensemble.selected_models)} modelos seleccionados de {len(models)} totales")
    print(f"✅ Métricas del ensemble: RMSE={ensemble.metrics['RMSE']:.4f}, R2={ensemble.metrics['R2']:.4f}")
    print(f"✅ Visualizaciones guardadas en: {IMG_CHARTS}")
    
    return ensemble

if __name__ == "__main__":
    main()