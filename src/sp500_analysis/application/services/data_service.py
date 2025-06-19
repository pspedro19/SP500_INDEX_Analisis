"""
Servicio de datos para el manejo de archivos Excel y fusión de datos.
BASADO EXACTAMENTE EN step_1_merge_excels.py
"""

import pandas as pd
import os
import logging
import time
from typing import List, Optional
from pathlib import Path

from sp500_analysis.config.settings import settings
from sp500_analysis.shared.logging.logger import setup_logging

logger = setup_logging(settings.log_dir, "data_service")


def cargar_archivo(ruta_archivo):
    """
    Carga un archivo Excel en un DataFrame de pandas.
    
    Args:
        ruta_archivo (str): Ruta al archivo Excel
        
    Returns:
        DataFrame: Datos del archivo cargado
    """
    try:
        t0 = time.perf_counter()
        logging.info(f"Cargando archivo: {os.path.basename(ruta_archivo)}")
        
        df = pd.read_excel(ruta_archivo)
        
        t1 = time.perf_counter()
        logging.info(f"Archivo {os.path.basename(ruta_archivo)} cargado en {t1-t0:.2f}s con {len(df)} filas.")
        return df
    except Exception as e:
        logging.error(f"Error al cargar {ruta_archivo}: {e}")
        return None


def filtrar_por_fecha(df, columna_fecha, fecha_inicio, fecha_fin):
    """
    Filtra un DataFrame por un rango de fechas.
    
    Args:
        df (DataFrame): DataFrame a filtrar
        columna_fecha (str): Nombre de la columna con fechas
        fecha_inicio (str): Fecha de inicio en formato 'YYYY-MM-DD'
        fecha_fin (str): Fecha de fin en formato 'YYYY-MM-DD'
        
    Returns:
        DataFrame: DataFrame filtrado
    """
    t0 = time.perf_counter()
    
    if df is None or columna_fecha not in df.columns:
        return df
    
    # Convertir a datetime si es necesario
    if not pd.api.types.is_datetime64_any_dtype(df[columna_fecha]):
        df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors='coerce')
    
    # Filtrar por rango de fechas
    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = pd.to_datetime(fecha_fin)
    
    df_filtrado = df[(df[columna_fecha] >= fecha_inicio) & (df[columna_fecha] <= fecha_fin)]
    
    t1 = time.perf_counter()
    logging.info(f"Filtrado por fecha completado en {t1-t0:.2f}s: {len(df)} -> {len(df_filtrado)} filas.")
    
    return df_filtrado


def imputar_valor_fecha_faltante(df, columna_fecha, fecha_base, fecha_objetivo):
    """
    Si 'fecha_objetivo' no existe en el DataFrame, copia la fila de 'fecha_base' y la agrega con la nueva fecha.
    
    Args:
        df (DataFrame): DataFrame a modificar
        columna_fecha (str): Nombre de la columna de fecha
        fecha_base (str): Fecha base para copiar valores
        fecha_objetivo (str): Fecha objetivo a imputar
        
    Returns:
        DataFrame: DataFrame con la fecha imputada si era necesario
    """
    if df is None or columna_fecha not in df.columns:
        return df

    fecha_base = pd.to_datetime(fecha_base)
    fecha_objetivo = pd.to_datetime(fecha_objetivo)
    
    if fecha_objetivo in df[columna_fecha].values:
        return df  # No hace falta imputar

    # Buscar fila base
    fila_base = df[df[columna_fecha] == fecha_base]
    if fila_base.empty:
        logging.warning(f"No se encontró data para la fecha base {fecha_base.date()} en este archivo.")
        return df

    # Copiar fila y cambiar la fecha
    nueva_fila = fila_base.copy()
    nueva_fila[columna_fecha] = fecha_objetivo

    # Concatenar al DataFrame
    df = pd.concat([df, nueva_fila], ignore_index=True)
    logging.info(f"Se imputó valor del {fecha_base.date()} para la fecha faltante {fecha_objetivo.date()}")
    return df


def merge_dataframes(dfs, columnas_merge, como_merge='outer'):
    """
    Combina múltiples DataFrames en uno solo.
    
    Args:
        dfs (list): Lista de DataFrames a combinar
        columnas_merge (list): Lista de columnas por las que hacer el merge
        como_merge (str): Tipo de merge (inner, outer, left, right)
        
    Returns:
        DataFrame: DataFrame combinado
    """
    t0 = time.perf_counter()
    
    if not dfs:
        return None
    
    # Filtrar DataFrames None
    dfs = [df for df in dfs if df is not None]
    
    if len(dfs) == 0:
        logging.warning("No hay DataFrames válidos para combinar.")
        return None
    
    if len(dfs) == 1:
        logging.info("Solo hay un DataFrame, no es necesario hacer merge.")
        return dfs[0]
    
    # Iniciar con el primer DataFrame
    resultado = dfs[0]
    filas_originales = [len(df) for df in dfs]
    columnas_originales = [len(df.columns) for df in dfs]
    
    # Combinar con el resto
    for i, df in enumerate(dfs[1:], 1):
        logging.info(f"Combinando DataFrame {i} con el resultado ({len(resultado)} filas + {len(df)} filas)...")
        resultado = pd.merge(resultado, df, on=columnas_merge, how=como_merge)
    
    t1 = time.perf_counter()
    
    # Estadísticas del merge
    stats = {
        'DFs_originales': len(dfs),
        'filas_originales': filas_originales,
        'filas_originales_total': sum(filas_originales),
        'columnas_originales': columnas_originales,
        'columnas_originales_total': sum(columnas_originales),
        'filas_resultado': len(resultado),
        'columnas_resultado': len(resultado.columns),
        'tiempo_merge': t1-t0
    }
    
    logging.info(f"Merge completado en {stats['tiempo_merge']:.2f}s: {stats['filas_resultado']} filas, {stats['columnas_resultado']} columnas.")
    
    return resultado


def run_data_merge() -> bool:
    """
    TU CÓDIGO ORIGINAL de step_1_merge_excels.py adaptado para settings
    """
    t_inicio = time.perf_counter()
    logging.info("=== INICIANDO PROCESO DE COMBINACIÓN DE ARCHIVOS EXCEL ===")
    
    # Configurar rutas usando settings
    DATA_RAW = str(settings.raw_dir)
    DATA_PREP = str(settings.preprocess_dir)
    DATE_COL = settings.date_col
    
    # Asegurar que existen los directorios
    Path(DATA_PREP).mkdir(parents=True, exist_ok=True)
    
    # Configuraciones
    columna_fecha = DATE_COL  
    columnas_merge = [DATE_COL]  
    fecha_inicio = "2014-01-01"
    fecha_fin = "2025-05-31"
    archivo_salida = os.path.join(DATA_PREP, "MERGEDEXCELS.xlsx")
    
    # Obtener lista de archivos
    archivos = [f for f in os.listdir(DATA_RAW) if f.endswith('.xlsx') and not f.startswith('~$')]
    logging.info(f"Se encontraron {len(archivos)} archivos Excel en la carpeta.")
    
    # Cargar y procesar archivos
    dfs = []
    archivos_ok = []
    archivos_error = []
    
    for archivo in archivos:
        ruta_completa = os.path.join(DATA_RAW, archivo)
        df = cargar_archivo(ruta_completa)
        
        if df is not None:
            logging.info(f"Archivo {archivo} cargado correctamente con {len(df)} filas")
            archivos_ok.append(archivo)
            
            # Renombrar fecha si es necesario
            if 'fecha' in df.columns:
                df = df.rename(columns={'fecha': columna_fecha})
                logging.info(f"Columna 'fecha' renombrada a '{columna_fecha}' en {archivo}")
            
            # Filtrar y procesar
            df_filtrado = filtrar_por_fecha(df, columna_fecha, fecha_inicio, fecha_fin)
            df_filtrado = imputar_valor_fecha_faltante(df_filtrado, columna_fecha, "2025-03-26", "2025-03-27")
            logging.info(f"Después de filtrar por fecha: {len(df_filtrado)} filas")
            
            dfs.append(df_filtrado)
        else:
            archivos_error.append(archivo)
    
    # Combinar los DataFrames
    df_combinado = merge_dataframes(dfs, columnas_merge)

    # Verificamos que el DataFrame no esté vacío
    if df_combinado is None:
        logging.error("No se pudo combinar los archivos")
        return False

    logging.info(f"Combinación exitosa. Resultado tiene {len(df_combinado)} filas y {len(df_combinado.columns)} columnas")

    # Asegurar que columna_fecha esté en formato datetime y ordenado
    df_combinado[columna_fecha] = pd.to_datetime(df_combinado[columna_fecha])
    df_combinado = df_combinado.sort_values(by=columna_fecha)

    # Crear rango completo de fechas desde el mínimo hasta el máximo
    fecha_min = df_combinado[columna_fecha].min()
    fecha_max = df_combinado[columna_fecha].max()
    fechas_completas = pd.date_range(start=fecha_min, end=fecha_max, freq='D')

    # Guardar el orden original de columnas
    columnas_originales = df_combinado.columns.tolist()

    # Reindexar usando fechas completas
    df_combinado = df_combinado.set_index(columna_fecha)
    df_combinado = df_combinado.reindex(fechas_completas)

    # Aplicar forward fill
    df_combinado = df_combinado.ffill()

    # Restaurar columna de fecha al inicio
    df_combinado[columna_fecha] = df_combinado.index
    columnas_ordenadas = [columna_fecha] + [col for col in columnas_originales if col != columna_fecha]
    df_combinado = df_combinado[columnas_ordenadas].reset_index(drop=True)

    logging.info(f"Forward fill aplicado y fechas normalizadas desde {fecha_min.date()} hasta {fecha_max.date()}")

    # Eliminar columnas sin nombre
    columnas_sin_nombre = [col for col in df_combinado.columns if col.startswith('Unnamed')]
    if columnas_sin_nombre:
        logging.info(f"Eliminando {len(columnas_sin_nombre)} columnas sin nombre")
        columnas_a_mantener = [col for col in df_combinado.columns if not col.startswith('Unnamed')]
        df_combinado = df_combinado[columnas_a_mantener]
    
    # Guardar el resultado
    try:
        t0 = time.perf_counter()
        df_combinado.to_excel(archivo_salida, index=False)
        t1 = time.perf_counter()
        logging.info(f"Archivo combinado guardado como: {archivo_salida} en {t1-t0:.2f}s")
        
        logging.info(f"Proceso completo en {time.perf_counter() - t_inicio:.2f}s")
        return True
        
    except Exception as e:
        logging.error(f"Error al guardar el archivo combinado: {e}")
        return False