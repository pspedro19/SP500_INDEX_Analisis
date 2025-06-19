"""Compatibility wrapper for visualization helpers.

Deprecated: use modules in ``sp500_analysis.shared.visualization`` instead."""

from __future__ import annotations

from sp500_analysis.shared.visualization.plotters import *  # noqa: F401,F403

"""
Funciones de visualización para ML pipelines
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import logging

# Configurar estilo
plt.style.use('default')
sns.set_palette("husl")

def plot_real_vs_pred(df, title="Real vs Predicho", output_path=None, figsize=(12, 8)):
    """
    Gráfico de valores reales vs predichos por modelo
    """
    try:
        plt.figure(figsize=figsize)
        
        if 'Modelo' in df.columns:
            models = df['Modelo'].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
            
            for i, model in enumerate(models):
                model_data = df[df['Modelo'] == model]
                if not model_data.empty:
                    plt.scatter(model_data['Valor_Real'], model_data['Valor_Predicho'], 
                              alpha=0.6, label=model, color=colors[i], s=20)
        else:
            plt.scatter(df['Valor_Real'], df['Valor_Predicho'], alpha=0.6, s=20)
        
        # Línea diagonal (predicción perfecta)
        min_val = min(df['Valor_Real'].min(), df['Valor_Predicho'].min())
        max_val = max(df['Valor_Real'].max(), df['Valor_Predicho'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        plt.xlabel('Valor Real')
        plt.ylabel('Valor Predicho')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"Gráfico guardado en: {output_path}")
        
        plt.close()
        
    except Exception as e:
        logging.error(f"Error creando gráfico real vs pred: {e}")

def plot_training_curves(history, output_path=None):
    """
    Gráfico de curvas de entrenamiento (para modelos con historial)
    """
    try:
        if history is None:
            return
            
        plt.figure(figsize=(12, 4))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.get('loss', []), label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy (si existe)
        plt.subplot(1, 2, 2)
        if 'accuracy' in history:
            plt.plot(history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in history:
                plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            # Placeholder si no hay accuracy
            plt.text(0.5, 0.5, 'No accuracy data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Additional Metrics')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"Curvas de entrenamiento guardadas en: {output_path}")
        
        plt.close()
        
    except Exception as e:
        logging.error(f"Error creando curvas de entrenamiento: {e}")

def plot_feature_importance(model, feature_names, output_path=None, top_n=20):
    """
    Gráfico de importancia de características
    """
    try:
        # Intentar obtener importancia según el tipo de modelo
        importance = None
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        elif hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
        
        if importance is None:
            logging.warning("No se pudo obtener importancia de características")
            return
        
        # Crear DataFrame para ordenar
        feature_df = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        })
        
        # Ordenar y tomar top_n
        feature_df = feature_df.sort_values('importance', ascending=True).tail(top_n)
        
        # Crear gráfico
        plt.figure(figsize=(10, max(6, len(feature_df) * 0.3)))
        plt.barh(range(len(feature_df)), feature_df['importance'])
        plt.yticks(range(len(feature_df)), feature_df['feature'])
        plt.xlabel('Importancia')
        plt.title(f'Top {len(feature_df)} Características Más Importantes')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"Importancia de características guardada en: {output_path}")
        
        plt.close()
        
    except Exception as e:
        logging.error(f"Error creando gráfico de importancia: {e}")

def plot_residuals(y_true, y_pred, output_path=None):
    """
    Gráfico de residuales
    """
    try:
        residuals = y_true - y_pred
        
        plt.figure(figsize=(12, 4))
        
        # Residuales vs predichos
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Valores Predichos')
        plt.ylabel('Residuales')
        plt.title('Residuales vs Predichos')
        plt.grid(True, alpha=0.3)
        
        # Histograma de residuales
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuales')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Residuales')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"Gráfico de residuales guardado en: {output_path}")
        
        plt.close()
        
    except Exception as e:
        logging.error(f"Error creando gráfico de residuales: {e}")
