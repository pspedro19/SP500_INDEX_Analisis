import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from sp500_analysis.config.settings import settings
from sp500_analysis.shared.logging.logger import configurar_logging

# Configurar logging
log_file = os.path.join(settings.log_dir, f"limpiar_ultimo_nan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
configurar_logging(log_file)

def clean_dataframe_up_to_last_nan(file_path, output_path=None):
    """
    Elimina todas las filas hasta la última fila que contiene un NaN (inclusive).
    
    Args:
        file_path (str): Ruta al archivo Excel a procesar
        output_path (str, optional): Ruta donde guardar el archivo limpio. Si es None,
                                    se genera automáticamente añadiendo '_sin_nan' al nombre original.
    
    Returns:
        pd.DataFrame: DataFrame limpio sin NaNs
    """
    # Cargar el archivo
    logging.info(f"Cargando archivo: {file_path}")
    try:
        df = pd.read_excel(file_path)
        logging.info(f"Archivo cargado con forma: {df.shape}")
    except Exception as e:
        logging.error(f"Error al cargar el archivo: {e}")
        return None
    
    # Verificar si hay NaNs en el DataFrame
    if not df.isna().any().any():
        logging.info("No se detectaron valores NaN en el DataFrame. No es necesario aplicar filtro.")
        return df
    
    # Mostrar información de los NaNs antes de la limpieza
    nan_counts = df.isna().sum()
    logging.info(f"Columnas con NaNs:")
    for col, count in nan_counts.items():
        if count > 0:
            logging.info(f"  - '{col}': {count} valores NaN")
    
    # Mostrar las primeras filas que contienen NaN
    nan_rows = df[df.isna().any(axis=1)]
    if not nan_rows.empty:
        logging.info(f"Primeras 3 filas con NaN:\n{nan_rows.head(3)}")
    
    # Encontrar la posición del último NaN en todo el DataFrame
    logging.info("Buscando la posición del último NaN...")
    last_nan_position = df.isna().stack()[::-1].idxmax()  # Retorna (row_index, column)
    
    # Obtener el índice de la fila donde se encuentra el último NaN
    last_nan_row_index = last_nan_position[0]
    last_nan_column = last_nan_position[1]
    
    # Registrar información sobre el último NaN encontrado
    logging.info(f"Último NaN encontrado en fila {last_nan_row_index}, columna '{last_nan_column}'")
    
    # Mostrar la fila donde se encuentra el último NaN
    logging.info(f"Fila con el último NaN:\n{df.iloc[last_nan_row_index]}")
    
    # Calcular cuántas filas serán eliminadas
    rows_to_remove = last_nan_row_index + 1  # +1 porque eliminamos también la fila con el último NaN
    total_rows = df.shape[0]
    remaining_rows = total_rows - rows_to_remove
    
    logging.info(f"Se eliminarán {rows_to_remove} filas de {total_rows} ({rows_to_remove/total_rows:.2%})")
    logging.info(f"Quedarán {remaining_rows} filas ({remaining_rows/total_rows:.2%} del original)")
    
    # Filtrar el DataFrame para conservar solo las filas posteriores al último NaN
    df_cleaned = df.iloc[last_nan_row_index + 1:].reset_index(drop=True)
    
    # Verificar que no queden NaN en el DataFrame resultante
    if df_cleaned.isna().any().any():
        logging.error("¡Error! Aún quedan valores NaN en el DataFrame después de la limpieza.")
        remaining_nans = df_cleaned.isna().sum().sum()
        logging.error(f"Cantidad de NaNs restantes: {remaining_nans}")
        # Mostrar en qué columnas quedan NaNs
        for col, count in df_cleaned.isna().sum().items():
            if count > 0:
                logging.error(f"  - '{col}': {count} valores NaN")
    else:
        logging.info("Limpieza exitosa: No quedan valores NaN en el DataFrame.")
    
    # Guardar el resultado
    try:
        df_cleaned.to_excel(output_path, index=False)
        logging.info(f"DataFrame limpio guardado en: {output_path}")
    except Exception as e:
        logging.error(f"Error al guardar el archivo limpio: {e}")
    
    return df_cleaned

# Para ejecutar este script independientemente
if __name__ == "__main__":
    # Ruta del archivo a procesar
    input_file = r"C:/Users/natus/Documents/Trabajo/PEDRO_PEREZ/Proyecto_Mercado_de_Valores/SP500_INDEX_Analisis/Data/3_trainingdata/ULTIMO_S&P500_final_FPI.xlsx"
    
    # Crear nombre de salida personalizado para la prueba
    input_dir = os.path.dirname(input_file)
    input_filename = os.path.basename(input_file)
    name, ext = os.path.splitext(input_filename)
    
    # Nombre de archivo de salida personalizado con indicador de prueba
    output_file = os.path.join(input_dir, f"{name}_PRUEBA_LIMPIEZA_NAN{ext}")
    
    logging.info(f"Archivo de entrada: {input_file}")
    logging.info(f"Archivo de salida: {output_file}")
    
    # Procesar el archivo
    clean_dataframe_up_to_last_nan(input_file, output_file)