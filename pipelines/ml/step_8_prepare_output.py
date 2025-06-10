import pandas as pd
import os
import logging

# Importar configuraciones centralizadas
from sp500_analysis.config.settings import settings

PROJECT_ROOT = settings.project_root
RESULTS_DIR = settings.results_dir

# ------------------------------
# CONFIGURACIÓN DE LOGGING
# ------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_powerbi_format(input_df):
    """
    Prepara un DataFrame para exportación a Power BI con formato español:
    - Convierte decimales de punto a coma
    - Ajusta las columnas de hiperparámetros 
    
    Args:
        input_df (DataFrame): DataFrame de entrada con valores numéricos
        
    Returns:
        DataFrame: DataFrame formateado para Power BI
    """
    df = input_df.copy()
    
    # Convertir columnas numéricas de punto a coma
    numeric_cols = ['Valor_Real', 'Valor_Predicho', 'RMSE']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].str.replace(".", ",")
    
    # Ajustar formato de la columna de hiperparámetros
    hiper_cols = [col for col in df.columns if 'parámetros' in col or 'parametros' in col.lower()]
    if hiper_cols:
        col_hiper = hiper_cols[0]
        df[col_hiper] = df[col_hiper].apply(lambda x: f'"{x}"' if not str(x).startswith('"') else x)
        logging.info(f"Columna de hiperparámetros ajustada: {col_hiper}")
    
    return df

def main():
    """
    Convierte el archivo de predicciones al formato adecuado para Power BI:
    - Cambia puntos decimales por comas
    - Ajusta la columna de hiperparámetros
    - Guarda con delimitador punto y coma
    """
    # Definir rutas usando constantes del módulo config
    input_file = os.path.join(RESULTS_DIR, "all_models_predictions.csv")
    output_file = os.path.join(RESULTS_DIR, "archivo_para_powerbi.csv")
    
    # Crear directorio si no existe
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        logging.info(f"Directorio creado: {RESULTS_DIR}")
    
    try:
        # Verificar existencia del archivo de entrada
        if not os.path.exists(input_file):
            logging.error(f"El archivo de entrada no existe: {input_file}")
            return False
            
        # Leer todo como texto para preservar formato
        logging.info(f"Leyendo archivo: {input_file}")
        df = pd.read_csv(input_file, encoding='utf-8', dtype=str)
        logging.info(f"Archivo leído correctamente: {len(df)} filas, {len(df.columns)} columnas")
        
        # Preparar formato para Power BI
        df_powerbi = prepare_powerbi_format(df)
        logging.info("Conversión de formato completada")
        
        # Guardar con delimitador punto y coma y decimal con coma
        df_powerbi.to_csv(output_file, index=False, sep=';', encoding='utf-8', quoting=1)
        logging.info(f"✅ CSV listo para Power BI (formato español): {output_file}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error en el procesamiento: {e}")
        return False

if __name__ == "__main__":
    main()