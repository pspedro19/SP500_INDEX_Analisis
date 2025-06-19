import pandas as pd
import numpy as np
import os
import re
import time
import logging
from datetime import datetime, timedelta
from typing import Any
import warnings
warnings.filterwarnings('ignore')

from sp500_analysis.application.preprocessing.base import BaseProcessor


# Configuración de logging
def configurar_logging(log_file='otherdataprocessor.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('OtherDataProcessor')


# Función para convertir fechas en diversos formatos
def convertir_fecha(fecha_str):
    """
    Convierte diversos formatos de fecha a pd.Timestamp.
    """
    if isinstance(fecha_str, (pd.Timestamp, datetime)):
        return pd.Timestamp(fecha_str)

    if pd.isna(fecha_str):
        return None

    if isinstance(fecha_str, str):
        fecha_str = fecha_str.strip()
        formatos = [
            '%d.%m.%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y',
            '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y'
        ]

        for formato in formatos:
            try:
                return pd.to_datetime(fecha_str, format=formato)
            except (ValueError, TypeError):
                continue

        # Si ningún formato funciona, intentar pd.to_datetime automático
        try:
            return pd.to_datetime(fecha_str)
        except (ValueError, TypeError):
            return None

    return None


class EOEProcessor(BaseProcessor):
    """
    Procesador de datos "Other" (EOE) que maneja archivos específicos:
    - US_Empire_State_Index
    - AAII_Investor_Sentiment  
    - Put_Call_Ratio_SPY
    - Chicago_Fed_NFCI
    """
    
    def __init__(self, config_file: str, data_root: str = 'data/0_raw', output_path: str = None, log_file: str = 'eoe_universal_processor.log'):
        super().__init__(data_root=data_root, log_file=log_file)
        self.config_file = config_file
        self.output_path = output_path
        self.logger = configurar_logging(log_file)
        self.config_data = None
        self.datos_procesados = {}
        self.indice_diario = None
        self.df_combinado = None
        self.estadisticas = {}
        self.fecha_min_global = None
        self.fecha_max_global = None

        # Mapeo de archivos específicos
        self.archivos_manuales = {
            'US_Empire_State_Index': os.path.join('business_confidence', 'US_Empire_State_Index.csv'),
            'AAII_Investor_Sentiment': os.path.join('consumer_confidence', 'AAII_Investor_Sentiment.xls'),
            'Put_Call_Ratio_SPY': os.path.join('consumer_confidence', 'Put_Call_Ratio_SPY.csv'),
            'Chicago_Fed_NFCI': os.path.join('leading_economic_index', 'Chicago_Fed_NFCI.csv')
        }

        self.logger.info("=" * 80)
        self.logger.info("INICIANDO PROCESO: OtherDataProcessor")
        self.logger.info(f"Archivo de configuración: {config_file}")
        self.logger.info(f"Directorio raíz de datos: {data_root}")
        self.logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)

    def _validate_input(self, data: Any) -> bool:
        return True

    def _transform(self, data: Any) -> Any:
        self.logger.info('Running EOE processor')
        return data

    def leer_configuracion(self):
        """
        Lee y filtra la configuración del archivo Excel para la fuente "Other"
        """
        try:
            self.logger.info("Leyendo archivo de configuración...")
            df_config = pd.read_excel(self.config_file)
            
            # Filtrar solo por fuente "Other" (sin restricción de tipo de preprocesamiento)
            self.config_data = df_config[df_config['Fuente'] == 'Other'].copy()
            
            num_configs = len(self.config_data)
            self.logger.info(f"Se encontraron {num_configs} configuraciones para procesar con fuente 'Other'")
            
            if num_configs == 0:
                self.logger.warning("No se encontraron configuraciones que cumplan los criterios")
                return None
                
            return self.config_data
        except Exception as e:
            self.logger.error(f"Error al leer configuración: {str(e)}")
            return None

    def encontrar_ruta_archivo(self, variable, tipo_macro):
        """
        Encuentra la ruta del archivo usando búsqueda en el sistema de archivos
        """
        # Probar primero con el mapeo manual
        if variable in self.archivos_manuales:
            ruta_manual = os.path.join(self.data_root, self.archivos_manuales[variable])
            if os.path.exists(ruta_manual):
                return ruta_manual

        # Búsqueda automática
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if file.startswith(variable) and file.endswith(('.csv', '.xlsx', '.xls')):
                    return os.path.join(root, file)

        return None

    def procesar_empire_state_manualmente(self, variable, tipo_macro, target_col):
        """
        Procesa los datos de Empire State Index manualmente
        """
        self.logger.info(f"Procesando {variable} manualmente")

        # Construir ruta directa basada en la estructura conocida
        ruta_directa = os.path.join(self.data_root, 'business_confidence', f"{variable}.csv")

        # Si no existe, intentar con la búsqueda normal
        if not os.path.exists(ruta_directa):
            input_file = self.encontrar_ruta_archivo(variable, tipo_macro)
            if input_file is None:
                self.logger.error(f"Archivo de entrada no encontrado para {variable}")
                return variable, None
        else:
            input_file = ruta_directa

        self.logger.info(f"- Archivo de entrada: {input_file}")

        try:
            # Cargar el archivo CSV
            df = pd.read_csv(input_file)
            self.logger.info(f"- Archivo cargado: {len(df)} filas, {len(df.columns)} columnas")

            # Crear un nuevo DataFrame con las columnas necesarias
            nuevo_df = pd.DataFrame()

            # Buscar columna de fecha
            fecha_col = None
            for col in df.columns:
                if any(term in col.lower() for term in ['date', 'survey', 'week']):
                    fecha_col = col
                    break

            if fecha_col is None and len(df.columns) > 0:
                fecha_col = df.columns[0]  # Usar primera columna como fallback

            if fecha_col is None:
                self.logger.error(f"No se pudo identificar columna de fecha en {input_file}")
                return variable, None

            # Verificar si existe la columna surveyDate
            if 'surveyDate' in df.columns:
                fecha_col = 'surveyDate'

            nuevo_df['fecha'] = pd.to_datetime(df[fecha_col])

            # Verificar rango de fechas
            fecha_min = nuevo_df['fecha'].min()
            fecha_max = nuevo_df['fecha'].max()
            self.logger.info(f"- Rango de fechas: {fecha_min.strftime('%Y-%m-%d')} a {fecha_max.strftime('%Y-%m-%d')}")

            # Verificar si tiene datos desde 2014
            fecha_2014 = pd.Timestamp('2014-01-01')
            tiene_desde_2014 = fecha_min <= fecha_2014
            self.logger.info(f"- ¿Tiene datos desde 2014 o antes?: {'Sí' if tiene_desde_2014 else 'No'}")

            # Procesar columnas de datos
            columnas_procesadas = []
            for col in df.columns:
                if col != fecha_col and col in ['GACDISA', 'AWCDISA']:
                    nuevo_nombre = f"ESI_{col}_{variable}_{tipo_macro}"
                    nuevo_df[nuevo_nombre] = pd.to_numeric(df[col], errors='coerce')
                    columnas_procesadas.append(col)

            self.logger.info(f"- Columnas encontradas y procesadas: {', '.join(columnas_procesadas)}")

            # Si no se encontraron las columnas esperadas, procesar todas las numéricas
            if not columnas_procesadas:
                for col in df.columns:
                    if col != fecha_col:
                        try:
                            numeric_series = pd.to_numeric(df[col], errors='coerce')
                            if not numeric_series.isna().all():
                                nuevo_nombre = f"ESI_{col}_{variable}_{tipo_macro}"
                                nuevo_df[nuevo_nombre] = numeric_series
                                columnas_procesadas.append(col)
                        except:
                            continue

            if len(columnas_procesadas) == 0:
                self.logger.error(f"No se encontraron columnas de datos válidas en {input_file}")
                return variable, None

            # Eliminar filas con todas las columnas de datos NaN
            data_cols = [col for col in nuevo_df.columns if col != 'fecha']
            nuevo_df = nuevo_df.dropna(subset=data_cols, how='all')

            if nuevo_df.empty:
                self.logger.error(f"No hay datos válidos después del procesamiento para {variable}")
                return variable, None

            # Actualizar estadísticas
            total_filas = len(nuevo_df)
            valores_validos = nuevo_df[data_cols].notna().sum().sum()

            self.estadisticas[variable] = {
                'tipo_macro': tipo_macro,
                'columnas_target': columnas_procesadas,
                'total_filas': total_filas,
                'valores_validos': valores_validos,
                'fecha_min': fecha_min,
                'fecha_max': fecha_max
            }

            self.logger.info(f"- Archivo procesado exitosamente: {total_filas} filas, {len(data_cols)} columnas de datos")
            
            return variable, nuevo_df

        except Exception as e:
            self.logger.error(f"Error procesando {variable}: {str(e)}")
            return variable, None

    def procesar_aaii_investor_sentiment(self, variable, tipo_macro, target_col):
        """
        Procesa los datos de AAII Investor Sentiment
        """
        self.logger.info(f"Procesando {variable} manualmente")

        input_file = self.encontrar_ruta_archivo(variable, tipo_macro)
        if input_file is None:
            self.logger.error(f"Archivo de entrada no encontrado para {variable}")
            return variable, None

        self.logger.info(f"- Archivo de entrada: {input_file}")

        try:
            # Cargar el archivo (puede ser .xls o .xlsx)
            if input_file.endswith('.xls'):
                df = pd.read_excel(input_file, engine='xlrd')
            else:
                df = pd.read_excel(input_file)
                
            self.logger.info(f"- Archivo cargado: {len(df)} filas, {len(df.columns)} columnas")

            # Crear un nuevo DataFrame
            nuevo_df = pd.DataFrame()

            # Buscar columna de fecha
            fecha_col = None
            for col in df.columns:
                if any(term in col.lower() for term in ['date', 'week']):
                    fecha_col = col
                    break

            if fecha_col is None and len(df.columns) > 0:
                fecha_col = df.columns[0]

            if fecha_col is None:
                self.logger.error(f"No se pudo identificar columna de fecha en {input_file}")
                return variable, None

            # Convertir fechas
            nuevo_df['fecha'] = pd.to_datetime(df[fecha_col])

            # Verificar rango de fechas
            fecha_min = nuevo_df['fecha'].min()
            fecha_max = nuevo_df['fecha'].max()
            self.logger.info(f"- Rango de fechas: {fecha_min.strftime('%Y-%m-%d')} a {fecha_max.strftime('%Y-%m-%d')}")

            # Procesar columnas de datos
            columnas_procesadas = []
            for col in df.columns:
                if col != fecha_col and any(term in col.lower() for term in ['bullish', 'bearish', 'neutral']):
                    nuevo_nombre = f"AAII_{col}_{variable}_{tipo_macro}"
                    nuevo_df[nuevo_nombre] = pd.to_numeric(df[col], errors='coerce')
                    columnas_procesadas.append(col)

            if not columnas_procesadas:
                # Procesar todas las columnas numéricas
                for col in df.columns:
                    if col != fecha_col:
                        try:
                            numeric_series = pd.to_numeric(df[col], errors='coerce')
                            if not numeric_series.isna().all():
                                nuevo_nombre = f"AAII_{col}_{variable}_{tipo_macro}"
                                nuevo_df[nuevo_nombre] = numeric_series
                                columnas_procesadas.append(col)
                        except:
                            continue

            if len(columnas_procesadas) == 0:
                self.logger.error(f"No se encontraron columnas de datos válidas en {input_file}")
                return variable, None

            # Eliminar filas con todas las columnas de datos NaN
            data_cols = [col for col in nuevo_df.columns if col != 'fecha']
            nuevo_df = nuevo_df.dropna(subset=data_cols, how='all')

            if nuevo_df.empty:
                self.logger.error(f"No hay datos válidos después del procesamiento para {variable}")
                return variable, None

            # Actualizar estadísticas
            total_filas = len(nuevo_df)
            valores_validos = nuevo_df[data_cols].notna().sum().sum()

            self.estadisticas[variable] = {
                'tipo_macro': tipo_macro,
                'columnas_target': columnas_procesadas,
                'total_filas': total_filas,
                'valores_validos': valores_validos,
                'fecha_min': fecha_min,
                'fecha_max': fecha_max
            }

            self.logger.info(f"- Archivo procesado exitosamente: {total_filas} filas, {len(data_cols)} columnas de datos")
            
            return variable, nuevo_df

        except Exception as e:
            self.logger.error(f"Error procesando {variable}: {str(e)}")
            return variable, None

    def procesar_put_call_ratio(self, variable, tipo_macro, target_col):
        """
        Procesa los datos de Put/Call Ratio SPY
        """
        self.logger.info(f"Procesando {variable} manualmente")

        input_file = self.encontrar_ruta_archivo(variable, tipo_macro)
        if input_file is None:
            self.logger.error(f"Archivo de entrada no encontrado para {variable}")
            return variable, None

        self.logger.info(f"- Archivo de entrada: {input_file}")

        try:
            # Cargar el archivo CSV
            df = pd.read_csv(input_file)
            self.logger.info(f"- Archivo cargado: {len(df)} filas, {len(df.columns)} columnas")

            # Crear un nuevo DataFrame
            nuevo_df = pd.DataFrame()

            # Buscar columna de fecha
            fecha_col = None
            for col in df.columns:
                if any(term in col.lower() for term in ['date', 'time']):
                    fecha_col = col
                    break

            if fecha_col is None and len(df.columns) > 0:
                fecha_col = df.columns[0]

            if fecha_col is None:
                self.logger.error(f"No se pudo identificar columna de fecha en {input_file}")
                return variable, None

            # Convertir fechas
            nuevo_df['fecha'] = pd.to_datetime(df[fecha_col])

            # Verificar rango de fechas
            fecha_min = nuevo_df['fecha'].min()
            fecha_max = nuevo_df['fecha'].max()
            self.logger.info(f"- Rango de fechas: {fecha_min.strftime('%Y-%m-%d')} a {fecha_max.strftime('%Y-%m-%d')}")

            # Procesar columnas de datos
            columnas_procesadas = []
            for col in df.columns:
                if col != fecha_col and any(term in col.lower() for term in ['ratio', 'put', 'call']):
                    nuevo_nombre = f"PCR_{col}_{variable}_{tipo_macro}"
                    nuevo_df[nuevo_nombre] = pd.to_numeric(df[col], errors='coerce')
                    columnas_procesadas.append(col)

            if not columnas_procesadas:
                # Procesar todas las columnas numéricas
                for col in df.columns:
                    if col != fecha_col:
                        try:
                            numeric_series = pd.to_numeric(df[col], errors='coerce')
                            if not numeric_series.isna().all():
                                nuevo_nombre = f"PCR_{col}_{variable}_{tipo_macro}"
                                nuevo_df[nuevo_nombre] = numeric_series
                                columnas_procesadas.append(col)
                        except:
                            continue

            if len(columnas_procesadas) == 0:
                self.logger.error(f"No se encontraron columnas de datos válidas en {input_file}")
                return variable, None

            # Eliminar filas con todas las columnas de datos NaN
            data_cols = [col for col in nuevo_df.columns if col != 'fecha']
            nuevo_df = nuevo_df.dropna(subset=data_cols, how='all')

            if nuevo_df.empty:
                self.logger.error(f"No hay datos válidos después del procesamiento para {variable}")
                return variable, None

            # Actualizar estadísticas
            total_filas = len(nuevo_df)
            valores_validos = nuevo_df[data_cols].notna().sum().sum()

            self.estadisticas[variable] = {
                'tipo_macro': tipo_macro,
                'columnas_target': columnas_procesadas,
                'total_filas': total_filas,
                'valores_validos': valores_validos,
                'fecha_min': fecha_min,
                'fecha_max': fecha_max
            }

            self.logger.info(f"- Archivo procesado exitosamente: {total_filas} filas, {len(data_cols)} columnas de datos")
            
            return variable, nuevo_df

        except Exception as e:
            self.logger.error(f"Error procesando {variable}: {str(e)}")
            return variable, None

    def procesar_chicago_fed_manualmente(self, variable, tipo_macro, target_col):
        """
        Procesa los datos de Chicago Fed NFCI
        """
        self.logger.info(f"Procesando {variable} manualmente")

        input_file = self.encontrar_ruta_archivo(variable, tipo_macro)
        if input_file is None:
            self.logger.error(f"Archivo de entrada no encontrado para {variable}")
            return variable, None

        self.logger.info(f"- Archivo de entrada: {input_file}")

        try:
            # Cargar el archivo CSV
            df = pd.read_csv(input_file)
            self.logger.info(f"- Archivo cargado: {len(df)} filas, {len(df.columns)} columnas")

            # Crear un nuevo DataFrame
            nuevo_df = pd.DataFrame()

            # Buscar columna de fecha (generalmente la primera)
            fecha_col = df.columns[0]
            self.logger.info(f"- Usando columna de fecha: {fecha_col}")

            # Convertir fechas
            nuevo_df['fecha'] = pd.to_datetime(df[fecha_col])

            # Verificar rango de fechas
            fecha_min = nuevo_df['fecha'].min()
            fecha_max = nuevo_df['fecha'].max()
            self.logger.info(f"- Rango de fechas: {fecha_min.strftime('%Y-%m-%d')} a {fecha_max.strftime('%Y-%m-%d')}")

            # Procesar columna TARGET específica
            if target_col and target_col in df.columns:
                nuevo_nombre = f"{target_col}_{variable}_{tipo_macro}"
                nuevo_df[nuevo_nombre] = pd.to_numeric(df[target_col], errors='coerce')
                columnas_procesadas = [target_col]
            else:
                # Procesar todas las columnas numéricas
                columnas_procesadas = []
                for col in df.columns:
                    if col != fecha_col:
                        try:
                            numeric_series = pd.to_numeric(df[col], errors='coerce')
                            if not numeric_series.isna().all():
                                nuevo_nombre = f"{col}_{variable}_{tipo_macro}"
                                nuevo_df[nuevo_nombre] = numeric_series
                                columnas_procesadas.append(col)
                        except:
                            continue

            if len(columnas_procesadas) == 0:
                self.logger.error(f"No se encontraron columnas de datos válidas en {input_file}")
                return variable, None

            # Eliminar filas con todas las columnas de datos NaN
            data_cols = [col for col in nuevo_df.columns if col != 'fecha']
            nuevo_df = nuevo_df.dropna(subset=data_cols, how='all')

            if nuevo_df.empty:
                self.logger.error(f"No hay datos válidos después del procesamiento para {variable}")
                return variable, None

            # Actualizar estadísticas
            total_filas = len(nuevo_df)
            valores_validos = nuevo_df[data_cols].notna().sum().sum()

            self.estadisticas[variable] = {
                'tipo_macro': tipo_macro,
                'columnas_target': columnas_procesadas,
                'total_filas': total_filas,
                'valores_validos': valores_validos,
                'fecha_min': fecha_min,
                'fecha_max': fecha_max
            }

            self.logger.info(f"- Archivo procesado exitosamente: {total_filas} filas, {len(data_cols)} columnas de datos")
            
            return variable, nuevo_df

        except Exception as e:
            self.logger.error(f"Error procesando {variable}: {str(e)}")
            return variable, None

    def procesar_archivo(self, config_row):
        """
        Procesa un archivo individual basado en la configuración
        """
        variable = config_row['Variable']
        tipo_macro = config_row['Tipo Macro']
        target_col = config_row['TARGET']
        
        self.logger.info(f"\nProcesando: {variable} ({tipo_macro})")
        self.logger.info(f"- Columna TARGET: {target_col}")
        self.logger.info(f"- Tipo de preprocesamiento: Normal")

        # Procesar según el tipo de archivo
        if variable == 'US_Empire_State_Index':
            return self.procesar_empire_state_manualmente(variable, tipo_macro, target_col)
        elif variable == 'AAII_Investor_Sentiment':
            return self.procesar_aaii_investor_sentiment(variable, tipo_macro, target_col)
        elif variable == 'Put_Call_Ratio_SPY':
            return self.procesar_put_call_ratio(variable, tipo_macro, target_col) 
        elif variable == 'Chicago_Fed_NFCI':
            return self.procesar_chicago_fed_manualmente(variable, tipo_macro, target_col)
        else:
            # Si no hay un método específico, intentamos un procesamiento genérico
            self.logger.warning(f"No existe procesador específico para {variable}, intentando procesamiento genérico")
            return self.procesar_archivo_generico(variable, tipo_macro, target_col)

    def procesar_archivo_generico(self, variable, tipo_macro, target_col):
        """
        Procesamiento genérico para archivos no especificados
        """
        input_file = self.encontrar_ruta_archivo(variable, tipo_macro)
        if input_file is None:
            self.logger.error(f"Archivo de entrada no encontrado para {variable}")
            return variable, None

        try:
            # Cargar el archivo
            if input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
            else:
                df = pd.read_excel(input_file)

            # Crear un nuevo DataFrame
            nuevo_df = pd.DataFrame()

            # Buscar columna de fecha
            fecha_col = df.columns[0]
            nuevo_df['fecha'] = pd.to_datetime(df[fecha_col])

            # Procesar columnas de datos
            for col in df.columns:
                if col != fecha_col:
                    try:
                        numeric_series = pd.to_numeric(df[col], errors='coerce')
                        if not numeric_series.isna().all():
                            nuevo_nombre = f"{col}_{variable}_{tipo_macro}"
                            nuevo_df[nuevo_nombre] = numeric_series
                    except:
                        continue

            return variable, nuevo_df

        except Exception as e:
            self.logger.error(f"Error procesando {variable}: {str(e)}")
            return variable, None

    def generar_indice_diario(self):
        """
        Genera un DataFrame con índice diario entre fechas mínima y máxima globales
        """
        # Filtrar para usar solo DataFrames válidos (no None)
        dfs_validos = {var: df for var, df in self.datos_procesados.items() if df is not None}
        
        if not dfs_validos:
            self.logger.error("No hay datos procesados válidos para generar el índice diario")
            return None
            
        # Calcular fechas mínima y máxima considerando solo los DataFrames válidos
        self.fecha_min_global = None
        self.fecha_max_global = None
        
        for variable, df in dfs_validos.items():
            fecha_min = df['fecha'].min()
            fecha_max = df['fecha'].max()
            
            if self.fecha_min_global is None or fecha_min < self.fecha_min_global:
                self.fecha_min_global = fecha_min
                
            if self.fecha_max_global is None or fecha_max > self.fecha_max_global:
                self.fecha_max_global = fecha_max
        
        self.logger.info("\nGenerando índice temporal diario...")
        self.logger.info(f"- Rango de fechas global: {self.fecha_min_global.strftime('%Y-%m-%d')} a {self.fecha_max_global.strftime('%Y-%m-%d')}")
        
        # Generar todas las fechas diarias
        todas_fechas = pd.date_range(start=self.fecha_min_global, end=self.fecha_max_global, freq='D')
        self.indice_diario = pd.DataFrame({'fecha': todas_fechas})
        
        self.logger.info(f"- Total de fechas diarias generadas: {len(self.indice_diario)}")
        return self.indice_diario

    def combinar_datos(self):
        """
        Combina todos los indicadores procesados con el índice diario
        """
        # Filtrar para usar solo DataFrames válidos (no None)
        dfs_validos = {var: df for var, df in self.datos_procesados.items() if df is not None}
        
        if not dfs_validos:
            self.logger.error("No hay datos procesados válidos para combinar")
            return None
            
        if self.indice_diario is None:
            self.logger.error("No se ha generado el índice diario")
            return None
            
        self.logger.info("\nCombinando datos con índice diario...")
        
        # Comenzar con el índice diario
        df_combinado = self.indice_diario.copy()
        
        # Para cada indicador, realizar el merge y aplicar ffill
        for variable, df in dfs_validos.items():
            self.logger.info(f"- Combinando: {variable}")
            
            # Normalizar la columna de fecha
            df['fecha'] = pd.to_datetime(df['fecha']).dt.normalize()
            
            # Realizar merge con el índice diario
            df_combinado = pd.merge(df_combinado, df, on='fecha', how='left')
            
            # Aplicar forward fill (ffill) para todas las columnas de datos
            for col in df.columns:
                if col != 'fecha':
                    df_combinado[col] = df_combinado[col].ffill()
        
        self.df_combinado = df_combinado
        self.logger.info(f"- DataFrame combinado: {len(df_combinado)} filas, {len(df_combinado.columns)} columnas")
        
        return self.df_combinado

    def analizar_cobertura_final(self):
        """
        Genera un informe detallado de cobertura final
        """
        # Filtrar para usar solo estadísticas de DataFrames válidos (no None)
        stats_validos = {var: stats for var, stats in self.estadisticas.items() 
                        if var in self.datos_procesados and self.datos_procesados[var] is not None}
        
        if not stats_validos or self.df_combinado is None:
            self.logger.error("No hay datos combinados o estadísticas para analizar")
            return
            
        self.logger.info("\n" + "=" * 50)
        self.logger.info("RESUMEN DE COBERTURA FINAL")
        self.logger.info("=" * 50)
        
        total_indicadores = len(stats_validos)
        total_dias = len(self.indice_diario)
        
        self.logger.info(f"Total indicadores procesados: {total_indicadores}")
        self.logger.info(f"Rango de fechas: {self.fecha_min_global.strftime('%Y-%m-%d')} a {self.fecha_max_global.strftime('%Y-%m-%d')}")
        self.logger.info(f"Total días en la serie: {total_dias}")
        self.logger.info("\nCobertura por indicador:")
        
        for variable, stats in stats_validos.items():
            # Calcular cobertura final después del ffill
            valores_finales = 0
            columnas_var = 0
            for col in self.df_combinado.columns:
                if col != 'fecha' and variable in col:
                    valores_finales += self.df_combinado[col].notna().sum()
                    columnas_var += 1
            
            if columnas_var > 0:
                cobertura_final = (valores_finales / (total_dias * columnas_var)) * 100
            else:
                cobertura_final = 0
            
            # Actualizar estadísticas
            self.estadisticas[variable]['cobertura_final'] = cobertura_final
            
            self.logger.info(f"- {variable}: {cobertura_final:.2f}%")

    def guardar_resultados(self, output_file='datos_economicos_other_procesados.xlsx'):
        """
        Guarda los resultados procesados en un archivo Excel
        """
        if self.df_combinado is None:
            self.logger.error("No hay datos para guardar")
            return False
            
        try:
            self.logger.info("\n" + "=" * 50)
            self.logger.info("GUARDANDO RESULTADOS")
            self.logger.info("=" * 50)
            
            self.logger.info(f"Guardando resultados en: {output_file}")
            
            # Crear un writer de Excel
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Guardar el DataFrame combinado en la primera hoja
                self.logger.info(f"Guardando DataFrame combinado en '{output_file}'...")
                self.df_combinado.to_excel(writer, sheet_name='Datos_Combinados', index=False)
                
                # Guardar estadísticas en una segunda hoja
                self.logger.info("Guardando estadísticas de los indicadores...")
                
                # Convertir diccionario de estadísticas a DataFrame
                stats_data = []
                for variable, stats in self.estadisticas.items():
                    row = {
                        'Variable': variable,
                        'Tipo_Macro': stats.get('tipo_macro', ''),
                        'Columnas_TARGET': ', '.join(stats.get('columnas_target', [])),
                        'Total_Filas_Original': stats.get('total_filas', 0),
                        'Valores_Validos_Original': stats.get('valores_validos', 0),
                        'Cobertura_Final_%': stats.get('cobertura_final', 0),
                        'Fecha_Min': stats.get('fecha_min', ''),
                        'Fecha_Max': stats.get('fecha_max', '')
                    }
                    stats_data.append(row)
                    
                df_stats = pd.DataFrame(stats_data)
                df_stats.to_excel(writer, sheet_name='Estadisticas', index=False)
                
                # Guardar metadatos
                metadata = {
                    'Proceso': ['OtherDataProcessor'],
                    'Fecha de proceso': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Total indicadores': [len(self.estadisticas)],
                    'Periodo': [f"{self.fecha_min_global.strftime('%Y-%m-%d')} a {self.fecha_max_global.strftime('%Y-%m-%d')}"],
                    'Total días': [len(self.indice_diario)]
                }
                pd.DataFrame(metadata).to_excel(writer, sheet_name='Metadatos')
                
                # Guardar la configuración utilizada
                if self.config_data is not None:
                    self.config_data.to_excel(writer, sheet_name='Configuración', index=False)
            
            self.logger.info(f"Archivo guardado exitosamente: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al guardar resultados: {str(e)}")
            return False

    def ejecutar_proceso_completo(self, output_file='datos_economicos_other_procesados.xlsx'):
        """
        Ejecuta el proceso completo de preprocesamiento de forma paramétrica basada en Data Engineering.xlsx
        """
        inicio = time.time()
        self.logger.info("Iniciando proceso completo OtherDataProcessor...")
        
        # 1. Leer configuración
        self.leer_configuracion()
        if self.config_data is None or len(self.config_data) == 0:
            return False
        
        # Mostrar resumen de configuraciones encontradas
        self.logger.info("\nResumen de configuraciones a procesar:")
        for idx, row in self.config_data.iterrows():
            self.logger.info(f"- {row['Variable']} (Tipo Macro: {row['Tipo Macro']}, TARGET: {row['TARGET']})")
        
        # 2. Procesar cada archivo
        for _, config_row in self.config_data.iterrows():
            variable, df_procesado = self.procesar_archivo(config_row)
            self.datos_procesados[variable] = df_procesado
        
        # Verificar si se procesó al menos un archivo correctamente
        archivos_correctos = sum(1 for df in self.datos_procesados.values() if df is not None)
        if archivos_correctos == 0:
            self.logger.error("No se pudo procesar correctamente ningún archivo")
            return False
        
        self.logger.info(f"\nArchivos procesados correctamente: {archivos_correctos}/{len(self.datos_procesados)}")
        
        # 3. Generar índice diario
        self.generar_indice_diario()
        if self.indice_diario is None:
            return False
        
        # 4. Combinar datos
        self.combinar_datos()
        if self.df_combinado is None:
            return False
        
        # 5. Analizar cobertura final
        self.analizar_cobertura_final()
        
        # 6. Guardar resultados
        resultado = self.guardar_resultados(output_file)
        
        # 7. Mostrar resumen final
        fin = time.time()
        tiempo_ejecucion = fin - inicio
        
        self.logger.info("\n" + "=" * 50)
        self.logger.info("RESUMEN DE EJECUCIÓN")
        self.logger.info("=" * 50)
        self.logger.info(f"Proceso: OtherDataProcessor")
        self.logger.info(f"Tiempo de ejecución: {tiempo_ejecucion:.2f} segundos")
        self.logger.info(f"Archivos procesados: {len(self.datos_procesados)}")
        self.logger.info(f"Archivos con error: {sum(1 for df in self.datos_procesados.values() if df is None)}")
        self.logger.info(f"Archivos procesados correctamente: {archivos_correctos}")
        self.logger.info(f"Periodo de datos: {self.fecha_min_global.strftime('%Y-%m-%d')} a {self.fecha_max_global.strftime('%Y-%m-%d')}")
        self.logger.info(f"Datos combinados: {len(self.df_combinado)} filas, {len(self.df_combinado.columns)} columnas")
        self.logger.info(f"Archivo de salida: {output_file}")
        self.logger.info(f"Estado: {'COMPLETADO' if resultado else 'ERROR'}")
        self.logger.info("=" * 50)
        
        return resultado

    def run(self, output_file='datos_economicos_other_procesados.xlsx'):
        """
        Método principal para ejecutar el procesador (compatible con BaseProcessor)
        """
        return self.ejecutar_proceso_completo(output_file)
