import os
import re
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from src.core.utils import configure_logging, convertir_valor, detectar_formato_fecha_inteligente, convertir_fecha_adaptativo

class OtherDataProcessor:
    """
    Clase para procesar datos de la fuente "Other" usando scripts personalizados.
    """
    def __init__(self, config_file, data_root='data/raw', log_file='logs/otherdataprocessor.log'):
        self.config_file = config_file
        self.data_root = data_root
        self.logger = configurar_logging(log_file)
        self.config_data = None
        self.fecha_min_global = None
        self.fecha_max_global = None
        self.indice_diario = None
        self.datos_procesados = {}
        self.df_combinado = None
        self.estadisticas = {}
        
        # Mapeo de variables a rutas de archivos conocidas
        self.data_paths = {
            'US_Empire_State_Index': os.path.join('business_confidence', 'US_Empire_State_Index.csv'),
            'AAII_Investor_Sentiment': os.path.join('consumer_confidence', 'AAII_Investor_Sentiment.xls'),
            'Chicago_Fed_NFCI': os.path.join('leading_economic_index', 'Chicago_Fed_NFCI.csv')
        }

        self.logger.info("=" * 80)
        self.logger.info("INICIANDO PROCESO: OtherDataProcessor")
        self.logger.info(f"Archivo de configuración: {config_file}")
        self.logger.info(f"Directorio raíz de datos: {data_root}")
        self.logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)
    
    def leer_configuracion(self):
        """
        Lee y filtra la configuración del archivo Excel para la fuente "Other"
        """
        try:
            self.logger.info("Leyendo archivo de configuración...")
            df_config = pd.read_excel(self.config_file)
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
        Encuentra la ruta completa del archivo basado en la variable y tipo_macro
        """
        if variable in self.data_paths:
            ruta_conocida = os.path.join(self.data_root, self.data_paths[variable])
            if os.path.exists(ruta_conocida):
                return ruta_conocida
        ruta_base = os.path.join(self.data_root, tipo_macro)
        for ext in ['.csv', '.xlsx', '.xls']:
            nombre_archivo = f"{variable}{ext}"
            ruta_completa = os.path.join(ruta_base, nombre_archivo)
            if os.path.exists(ruta_completa):
                return ruta_completa
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if file.startswith(variable) and file.endswith(('.csv', '.xlsx', '.xls')):
                    return os.path.join(root, file)
        return None
    
    def ejecutar_script(self, script_path, variable, output_path=None):
        """
        Ejecuta un script de Python a través del sistema
        """
        try:
            self.logger.info(f"Ejecutando script: {script_path}")
            spec = importlib.util.spec_from_file_location("custom_script", script_path)
            module = importlib.util.module_from_spec(spec)
            current_dir = os.getcwd()
            script_dir = os.path.dirname(os.path.abspath(script_path))
            os.chdir(script_dir)
            sys.argv = ['']
            try:
                spec.loader.exec_module(module)
                resultado = True
            except Exception as e:
                self.logger.error(f"Error al ejecutar el script {script_path}: {str(e)}")
                resultado = False
            os.chdir(current_dir)
            return resultado
        except Exception as e:
            self.logger.error(f"Error al cargar el script {script_path}: {str(e)}")
            return False
    
    def procesar_empire_state_manualmente(self, variable, tipo_macro, target_col):
        """
        Procesa los datos de Empire State Index manualmente cuando el script no está disponible
        """
        self.logger.info(f"Procesando {variable} manualmente")
        ruta_directa = os.path.join(self.data_root, 'business_confidence', f"{variable}.csv")
        if not os.path.exists(ruta_directa):
            input_file = self.encontrar_ruta_archivo(variable, tipo_macro)
            if input_file is None:
                self.logger.error(f"Archivo de entrada no encontrado para {variable}")
                return variable, None
        else:
            input_file = ruta_directa
        self.logger.info(f"- Archivo de entrada: {input_file}")
        try:
            df = pd.read_csv(input_file)
            self.logger.info(f"- Archivo cargado: {len(df)} filas, {len(df.columns)} columnas")
            nuevo_df = pd.DataFrame()
            fecha_col = None
            for col in df.columns:
                if any(term in col.lower() for term in ['date', 'survey', 'week']):
                    fecha_col = col
                    break
            if fecha_col is None and len(df.columns) > 0:
                fecha_col = df.columns[0]
            if fecha_col is None:
                self.logger.error(f"No se pudo identificar columna de fecha en {input_file}")
                return variable, None
            if 'surveyDate' in df.columns:
                fecha_col = 'surveyDate'
            nuevo_df['fecha'] = pd.to_datetime(df[fecha_col])
            fecha_min = nuevo_df['fecha'].min()
            fecha_max = nuevo_df['fecha'].max()
            self.logger.info(f"- Rango de fechas: {fecha_min.strftime('%Y-%m-%d')} a {fecha_max.strftime('%Y-%m-%d')}")
            fecha_2014 = pd.Timestamp('2014-01-01')
            tiene_desde_2014 = fecha_min <= fecha_2014
            self.logger.info(f"- ¿Tiene datos desde 2014 o antes?: {'Sí' if tiene_desde_2014 else 'No'}")
            if fecha_min > fecha_2014:
                self.logger.warning(f"- ATENCIÓN: Los datos comienzan en {fecha_min.strftime('%Y-%m-%d')}, después de 2014-01-01")
            if tipo_macro == 'business_confidence' and target_col == 'macro_business_confidence_fe_1month':
                if 'GACDISA' in df.columns:
                    nuevo_df[f'{target_col}_GACDISA_{variable}_{tipo_macro}'] = df['GACDISA']
                    self.logger.info(f"- Columna GACDISA encontrada y procesada")
                if 'AWCDISA' in df.columns:
                    nuevo_df[f'{target_col}_AWCDISA_{variable}_{tipo_macro}'] = df['AWCDISA']
                    self.logger.info(f"- Columna AWCDISA encontrada y procesada")
                self.logger.info(f"Procesamiento específico para {variable} con TARGET={target_col}")
            else:
                columnas_indicador = ['GACDISA', 'AWCDISA', 'headline', 'main_index']
                prefix = target_col if pd.notna(target_col) else 'ESI'
                columnas_encontradas = []
                for col in columnas_indicador:
                    if col in df.columns:
                        nuevo_df[f'{prefix}_{col}_{variable}_{tipo_macro}'] = df[col]
                        columnas_encontradas.append(col)
                if columnas_encontradas:
                    self.logger.info(f"- Columnas encontradas y procesadas: {', '.join(columnas_encontradas)}")
                if len(nuevo_df.columns) == 1:
                    columnas_usadas = []
                    for col in df.columns:
                        if col != fecha_col and pd.api.types.is_numeric_dtype(df[col]):
                            nuevo_df[f'{prefix}_{col}_{variable}_{tipo_macro}'] = df[col]
                            columnas_usadas.append(col)
                    if columnas_usadas:
                        self.logger.info(f"- Se usaron columnas numéricas genéricas: {', '.join(columnas_usadas)}")
            nuevo_df = nuevo_df.sort_values('fecha')
            for col in nuevo_df.columns:
                if col != 'fecha':
                    col_no_nulos = nuevo_df[col].notna().sum()
                    pct_no_nulos = (col_no_nulos / len(nuevo_df)) * 100
                    fecha_min_col = nuevo_df.loc[nuevo_df[col].notna(), 'fecha'].min()
                    fecha_max_col = nuevo_df.loc[nuevo_df[col].notna(), 'fecha'].max()
                    self.logger.info(f"- Columna {col}: {col_no_nulos} valores no nulos ({pct_no_nulos:.2f}%), rango: {fecha_min_col.strftime('%Y-%m-%d')} a {fecha_max_col.strftime('%Y-%m-%d')}")
            fecha_min = nuevo_df['fecha'].min()
            fecha_max = nuevo_df['fecha'].max()
            self.estadisticas[variable] = {
                'tipo_macro': tipo_macro,
                'columna_target': target_col or 'ESI',
                'total_filas': len(nuevo_df),
                'valores_validos': len(nuevo_df),
                'fecha_min': fecha_min,
                'fecha_max': fecha_max
            }
            self.logger.info(f"- {variable}: {len(nuevo_df)} filas procesadas manualmente, periodo: {fecha_min.strftime('%Y-%m-%d')} a {fecha_max.strftime('%Y-%m-%d')}")
            return variable, nuevo_df
        except Exception as e:
            self.logger.error(f"Error al procesar {variable} manualmente: {str(e)}")
            return variable, None
    
    def procesar_aaii_investor_sentiment(self, variable, tipo_macro, target_col):
        """
        Procesa los datos de AAII Investor Sentiment manualmente
        """
        self.logger.info(f"Procesando {variable} manualmente")
        ruta_directa = os.path.join(self.data_root, 'consumer_confidence', f"{variable}.xls")
        if not os.path.exists(ruta_directa):
            input_file = self.encontrar_ruta_archivo(variable, tipo_macro)
            if input_file is None:
                self.logger.error(f"Archivo de entrada no encontrado para {variable}")
                return variable, None
        else:
            input_file = ruta_directa
        self.logger.info(f"- Archivo de entrada: {input_file}")
        try:
            df = None
            engines = ['openpyxl', 'xlrd']
            for engine in engines:
                try:
                    self.logger.info(f"Intentando leer Excel con engine: {engine}")
                    df = pd.read_excel(input_file, engine=engine)
                    break
                except Exception as e:
                    self.logger.warning(f"Error al leer con engine {engine}: {str(e)}")
            if df is None:
                try:
                    self.logger.info("Intentando leer como CSV...")
                    df = pd.read_csv(input_file)
                except Exception as e:
                    self.logger.error(f"Error al leer como CSV: {str(e)}")
                    return variable, None
            if df is None or len(df) == 0:
                self.logger.error(f"No se pudo leer el archivo {input_file} con ningún método")
                return variable, None
            self.logger.info(f"- Archivo cargado: {len(df)} filas, {len(df.columns)} columnas")
            self.logger.info(f"- Columnas disponibles: {', '.join(df.columns.tolist())}")
            columna_fecha = None
            for col in df.columns:
                if any(term in col.lower() for term in ['date', 'week', 'period']):
                    columna_fecha = col
                    self.logger.info(f"- Usando columna '{col}' como fecha")
                    break
            if columna_fecha is None:
                columna_fecha = df.columns[0]
                self.logger.info(f"- Usando primera columna '{columna_fecha}' como fecha (fallback)")
            columnas_objetivo = ['Bearish', 'Bull-Bear Spread', 'Bullish']
            columnas_encontradas = [col for col in columnas_objetivo if col in df.columns]
            columnas_faltantes = [col for col in columnas_objetivo if col not in df.columns]
            if columnas_encontradas:
                self.logger.info(f"- Columnas encontradas: {', '.join(columnas_encontradas)}")
            if columnas_faltantes:
                self.logger.warning(f"- Columnas no encontradas: {', '.join(columnas_faltantes)}")
            resultado = pd.DataFrame()
            resultado['fecha'] = pd.to_datetime(df[columna_fecha])
            fecha_min_total = resultado['fecha'].min()
            fecha_max_total = resultado['fecha'].max()
            self.logger.info(f"- Rango de fechas total: {fecha_min_total.strftime('%Y-%m-%d')} a {fecha_max_total.strftime('%Y-%m-%d')}")
            fecha_2014 = pd.to_datetime('2014-01-01')
            tiene_desde_2014 = fecha_min_total <= fecha_2014
            self.logger.info(f"- ¿Tiene datos desde 2014 o antes?: {'Sí' if tiene_desde_2014 else 'No'}")
            if fecha_min_total > fecha_2014:
                self.logger.warning(f"- ATENCIÓN: Los datos comienzan en {fecha_min_total.strftime('%Y-%m-%d')}, después de 2014-01-01")
            prefix = target_col if pd.notna(target_col) else 'AAII'
            for columna in columnas_encontradas:
                nombre_col = f'{prefix}_{columna}_{variable}_{tipo_macro}'
                resultado[nombre_col] = df[columna]
                fechas_validas = resultado[resultado[nombre_col].notna()]
                if len(fechas_validas) > 0:
                    fecha_min_col = fechas_validas['fecha'].min()
                    fecha_max_col = fechas_validas['fecha'].max()
                    self.logger.info(f"- Columna {nombre_col}: Rango de fechas {fecha_min_col.strftime('%Y-%m-%d')} a {fecha_max_col.strftime('%Y-%m-%d')}")
                    self.logger.info(f"- Valores disponibles: {len(fechas_validas)}/{len(resultado)} ({len(fechas_validas)/len(resultado)*100:.2f}%)")
            if not columnas_encontradas:
                self.logger.warning(f"- No se encontraron columnas específicas. Usando columnas numéricas disponibles.")
                columnas_usadas = []
                for col in df.columns:
                    if col != columna_fecha and pd.api.types.is_numeric_dtype(df[col]):
                        nombre_col = f'{prefix}_{col}_{variable}_{tipo_macro}'
                        resultado[nombre_col] = df[col]
                        columnas_usadas.append(col)
                        fechas_validas = resultado[resultado[nombre_col].notna()]
                        if len(fechas_validas) > 0:
                            fecha_min_col = fechas_validas['fecha'].min()
                            fecha_max_col = fechas_validas['fecha'].max()
                            self.logger.info(f"- Columna {nombre_col}: Rango de fechas {fecha_min_col.strftime('%Y-%m-%d')} a {fecha_max_col.strftime('%Y-%m-%d')}")
                            self.logger.info(f"- Valores disponibles: {len(fechas_validas)}/{len(resultado)} ({len(fechas_validas)/len(resultado)*100:.2f}%)")
                if columnas_usadas:
                    self.logger.info(f"- Se utilizaron columnas numéricas alternativas: {', '.join(columnas_usadas)}")
            resultado = resultado.sort_values('fecha')
            fecha_min = resultado['fecha'].min()
            fecha_max = resultado['fecha'].max()
            self.estadisticas[variable] = {
                'tipo_macro': tipo_macro,
                'columna_target': target_col or prefix,
                'total_filas': len(resultado),
                'valores_validos': len(resultado),
                'fecha_min': fecha_min,
                'fecha_max': fecha_max
            }
            self.logger.info(f"- {variable}: {len(resultado)} filas procesadas manualmente, periodo: {fecha_min.strftime('%Y-%m-%d')} a {fecha_max.strftime('%Y-%m-%d')}")
            return variable, resultado
        except Exception as e:
            self.logger.error(f"Error al procesar {variable} manualmente: {str(e)}")
            return variable, None

    def procesar_put_call_ratio(self, variable, tipo_macro, target_col):
        """
        Procesa los datos de Put/Call Ratio manualmente, agregando intradía a un solo valor diario.
        Si se encuentran columnas para 'put_volume' y 'call_volume', se agrupan por día sumándolas
        y se calcula el ratio. En caso contrario se utiliza un fallback agrupando las columnas numéricas
        (mediante la media).
        """
        self.logger.info(f"Procesando {variable} manualmente")
        ruta_directa = os.path.join(self.data_root, 'consumer_confidence', f"{variable}.csv")
        if not os.path.exists(ruta_directa):
            input_file = self.encontrar_ruta_archivo(variable, tipo_macro)
            if input_file is None:
                self.logger.error(f"Archivo de entrada no encontrado para {variable}")
                return variable, None
        else:
            input_file = ruta_directa
        self.logger.info(f"- Archivo de entrada: {input_file}")
        
        try:
            df = pd.read_csv(input_file)
            self.logger.info(f"- Archivo cargado: {len(df)} filas, {len(df.columns)} columnas")
            self.logger.info(f"- Columnas disponibles: {', '.join(df.columns.tolist())}")
            
            result_df = pd.DataFrame()
            fecha_col = None
            for col in df.columns:
                if any(term in col.lower() for term in ['date', 'time', 'periodo']):
                    fecha_col = col
                    self.logger.info(f"- Usando columna '{col}' como fecha/hora")
                    break
            if fecha_col is None and len(df.columns) > 0:
                fecha_col = df.columns[0]
                self.logger.info(f"- Usando primera columna '{fecha_col}' como fecha (fallback)")
            if fecha_col is None:
                self.logger.error(f"No se pudo identificar columna de fecha en {input_file}")
                return variable, None
                
            df['fecha'] = pd.to_datetime(df[fecha_col], errors='coerce')
            df = df.dropna(subset=['fecha'])
            df['fecha_dia'] = df['fecha'].dt.normalize()
            
            # Buscar columnas de volúmenes
            col_put = None
            col_call = None
            posibles_put = ['put_volume', 'Put_Volume', 'volume_puts']
            posibles_call = ['call_volume', 'Call_Volume', 'volume_calls']
            for c in df.columns:
                if c in posibles_put:
                    col_put = c
                if c in posibles_call:
                    col_call = c
            
            if not col_put or not col_call:
                self.logger.warning("- No se encontraron columnas 'put_volume' y 'call_volume' estándar. "
                                    "Se aplicará fallback usando columnas numéricas.")
                result_df['fecha'] = df['fecha_dia']
                for col in df.select_dtypes(include=[np.number]).columns:
                    if col != 'fecha_dia':
                        prefix = target_col if pd.notna(target_col) else 'PutCall'
                        nuevo_nombre = f'{prefix}_{col}_{variable}_{tipo_macro}'
                        result_df[nuevo_nombre] = df[col]
                result_df = result_df.groupby('fecha', as_index=False).mean()
                result_df = result_df.sort_values('fecha')
                fecha_min = result_df['fecha'].min()
                fecha_max = result_df['fecha'].max()
                self.logger.info(f"- {variable}: {len(result_df)} filas tras agrupar fallback, periodo: {fecha_min.date()} a {fecha_max.date()}")
                return variable, result_df
            else:
                df_diario = df.groupby('fecha_dia', as_index=False).agg({
                    col_put: 'sum',
                    col_call: 'sum'
                })
                df_diario['put_call_ratio'] = df_diario[col_put] / df_diario[col_call].replace(0, np.nan)
                df_diario.rename(columns={'fecha_dia': 'fecha'}, inplace=True)
                col_name = f'consumer_confidence_PutCall_{variable}_{tipo_macro}'
                df_diario[col_name] = df_diario['put_call_ratio']
                df_diario = df_diario.sort_values('fecha')
                fecha_min = df_diario['fecha'].min()
                fecha_max = df_diario['fecha'].max()
                self.logger.info(f"- {variable}: {len(df_diario)} filas diarias tras agrupar, periodo: {fecha_min.date()} a {fecha_max.date()}")
                result_df = df_diario[['fecha', col_put, col_call, col_name]]
                self.estadisticas[variable] = {
                    'tipo_macro': tipo_macro,
                    'columna_target': target_col or 'PutCall',
                    'total_filas': len(result_df),
                    'valores_validos': len(result_df),
                    'fecha_min': fecha_min,
                    'fecha_max': fecha_max
                }
                return variable, result_df

        except Exception as e:
            self.logger.error(f"Error al procesar {variable} manualmente: {str(e)}")
            return variable, None
            
    def procesar_archivo_generico(self, variable, tipo_macro, target_col):
        """
        Procesamiento genérico para cualquier archivo de datos
        """
        self.logger.info(f"Procesando {variable} de forma genérica")
        input_file = self.encontrar_ruta_archivo(variable, tipo_macro)
        if input_file is None:
            self.logger.error(f"Archivo de entrada no encontrado para {variable}")
            return variable, None
        self.logger.info(f"- Archivo de entrada: {input_file}")
        try:
            if input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
            elif input_file.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(input_file)
            else:
                self.logger.error(f"Formato de archivo no soportado: {input_file}")
                return variable, None
            fecha_col = None
            for col in df.columns:
                if any(term in col.lower() for term in ['date', 'time', 'fecha', 'week', 'period']):
                    fecha_col = col
                    break
            if fecha_col is None and len(df.columns) > 0:
                try:
                    pd.to_datetime(df.iloc[:, 0])
                    fecha_col = df.columns[0]
                except:
                    self.logger.error(f"No se pudo identificar columna de fecha en {input_file}")
                    return variable, None
            result_df = pd.DataFrame()
            result_df['fecha'] = pd.to_datetime(df[fecha_col])
            result_df = result_df.dropna(subset=['fecha'])
            prefix = target_col if pd.notna(target_col) else variable.split('_')[0]
            for col in df.columns:
                if col != fecha_col and pd.api.types.is_numeric_dtype(df[col]):
                    result_df[f'{prefix}_{col}_{variable}_{tipo_macro}'] = df[col]
            result_df = result_df.sort_values('fecha')
            fecha_min = result_df['fecha'].min()
            fecha_max = result_df['fecha'].max()
            self.estadisticas[variable] = {
                'tipo_macro': tipo_macro,
                'columna_target': target_col or prefix,
                'total_filas': len(result_df),
                'valores_validos': len(result_df),
                'fecha_min': fecha_min,
                'fecha_max': fecha_max
            }
            self.logger.info(f"- {variable}: {len(result_df)} filas procesadas genéricamente, periodo: {fecha_min.strftime('%Y-%m-%d')} a {fecha_max.strftime('%Y-%m-%d')}")
            return variable, result_df
        except Exception as e:
            self.logger.error(f"Error al procesar {variable} genéricamente: {str(e)}")
            return variable, None
            
    def procesar_chicago_fed_manualmente(self, variable, tipo_macro, target_col):
        """
        Procesa los datos de Chicago Fed NFCI manualmente cuando el script no está disponible
        """
        self.logger.info(f"Procesando {variable} manualmente")
        ruta_directa = os.path.join(self.data_root, 'leading_economic_index', f"{variable}.csv")
        if not os.path.exists(ruta_directa):
            input_file = self.encontrar_ruta_archivo(variable, tipo_macro)
            if input_file is None:
                self.logger.error(f"Archivo de entrada no encontrado para {variable}")
                return variable, None
        else:
            input_file = ruta_directa
        self.logger.info(f"- Archivo de entrada: {input_file}")
        try:
            df = pd.read_csv(input_file)
            self.logger.info(f"- Archivo cargado: {len(df)} filas, {len(df.columns)} columnas")
            self.logger.info(f"- Columnas disponibles: {', '.join(df.columns.tolist())}")
            if 'Friday_of_Week' in df.columns:
                df['fecha'] = pd.to_datetime(df['Friday_of_Week'], format='%m/%d/%Y')
                self.logger.info(f"- Usando columna 'Friday_of_Week' como fecha")
            else:
                for col in df.columns:
                    if any(term in col.lower() for term in ['date', 'week', 'period']):
                        df['fecha'] = pd.to_datetime(df[col])
                        self.logger.info(f"- Usando columna '{col}' como fecha")
                        break
            if 'fecha' not in df.columns:
                self.logger.error(f"No se pudo identificar columna de fecha en {input_file}")
                return variable, None
            fecha_min_total = df['fecha'].min()
            fecha_max_total = df['fecha'].max()
            self.logger.info(f"- Rango de fechas total: {fecha_min_total.strftime('%Y-%m-%d')} a {fecha_max_total.strftime('%Y-%m-%d')}")
            fecha_2014 = pd.to_datetime('2014-01-01')
            tiene_desde_2014 = fecha_min_total <= fecha_2014
            self.logger.info(f"- ¿Tiene datos desde 2014 o antes?: {'Sí' if tiene_desde_2014 else 'No'}")
            if fecha_min_total > fecha_2014:
                self.logger.warning(f"- ATENCIÓN: Los datos comienzan en {fecha_min_total.strftime('%Y-%m-%d')}, después de 2014-01-01")
            df_filtrado = df[df['fecha'] >= fecha_2014]
            self.logger.info(f"- Filtrando desde 2014-01-01: {len(df_filtrado)}/{len(df)} filas ({(len(df_filtrado)/len(df)*100):.2f}%)")
            resultado = pd.DataFrame()
            resultado['fecha'] = df_filtrado['fecha']
            columnas_procesadas = []
            if 'NFCI' in df_filtrado.columns:
                resultado[f'NFCI_{variable}_{tipo_macro}'] = df_filtrado['NFCI']
                columnas_procesadas.append('NFCI')
                fecha_min_nfci = df_filtrado.loc[df_filtrado['NFCI'].notna(), 'fecha'].min()
                fecha_max_nfci = df_filtrado.loc[df_filtrado['NFCI'].notna(), 'fecha'].max()
                self.logger.info(f"- Columna NFCI: Rango de fechas {fecha_min_nfci.strftime('%Y-%m-%d')} a {fecha_max_nfci.strftime('%Y-%m-%d')}")
            if 'ANFCI' in df_filtrado.columns:
                resultado[f'ANFCI_{variable}_{tipo_macro}'] = df_filtrado['ANFCI']
                columnas_procesadas.append('ANFCI')
                fecha_min_anfci = df_filtrado.loc[df_filtrado['ANFCI'].notna(), 'fecha'].min()
                fecha_max_anfci = df_filtrado.loc[df_filtrado['ANFCI'].notna(), 'fecha'].max()
                self.logger.info(f"- Columna ANFCI: Rango de fechas {fecha_min_anfci.strftime('%Y-%m-%d')} a {fecha_max_anfci.strftime('%Y-%m-%d')}")
            if len(columnas_procesadas) == 0:
                self.logger.warning(f"- No se encontraron columnas NFCI o ANFCI. Utilizando columnas numéricas disponibles")
                for col in df_filtrado.columns:
                    if col != 'fecha' and pd.api.types.is_numeric_dtype(df_filtrado[col]):
                        prefix = target_col if pd.notna(target_col) else 'NFCI'
                        nuevo_nombre = f'{prefix}_{col}_{variable}_{tipo_macro}'
                        resultado[nuevo_nombre] = df_filtrado[col]
                        columnas_procesadas.append(col)
                        fecha_min_col = df_filtrado.loc[df_filtrado[col].notna(), 'fecha'].min()
                        fecha_max_col = df_filtrado.loc[df_filtrado[col].notna(), 'fecha'].max()
                        self.logger.info(f"- Columna {col}: Rango de fechas {fecha_min_col.strftime('%Y-%m-%d')} a {fecha_max_col.strftime('%Y-%m-%d')}")
            self.logger.info(f"- Columnas procesadas: {', '.join(columnas_procesadas)}")
            resultado = resultado.sort_values('fecha')
            fecha_min = resultado['fecha'].min()
            fecha_max = resultado['fecha'].max()
            self.estadisticas[variable] = {
                'tipo_macro': tipo_macro,
                'columna_target': target_col or 'NFCI',
                'total_filas': len(resultado),
                'valores_validos': len(resultado),
                'fecha_min': fecha_min,
                'fecha_max': fecha_max
            }
            self.logger.info(f"- {variable}: {len(resultado)} filas procesadas manualmente, periodo: {fecha_min.strftime('%Y-%m-%d')} a {fecha_max.strftime('%Y-%m-%d')}")
            return variable, resultado
        except Exception as e:
            self.logger.error(f"Error al procesar {variable} manualmente: {str(e)}")
            return variable, None
    
    def cargar_resultado_script(self, output_file, variable, target_col):
        """
        Carga el archivo de resultado generado por el script y lo estandariza
        """
        try:
            self.logger.info(f"Cargando resultado desde {output_file}")
            if not os.path.exists(output_file):
                self.logger.error(f"El archivo de salida {output_file} no existe")
                return None
            if output_file.endswith('.csv'):
                df = pd.read_csv(output_file)
            elif output_file.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(output_file)
            else:
                self.logger.error(f"Formato de archivo no soportado: {output_file}")
                return None
            fecha_col = None
            for col in df.columns:
                if any(palabra in col.lower() for palabra in ['date', 'fecha', 'time', 'día']):
                    fecha_col = col
                    break
            if fecha_col is None:
                self.logger.error(f"No se pudo identificar la columna de fecha en {output_file}")
                return None
            result_df = pd.DataFrame()
            result_df['fecha'] = df[fecha_col].apply(convertir_fecha)
            result_df = result_df.dropna(subset=['fecha'])
            for col in df.columns:
                if col != fecha_col:
                    new_name = f"{target_col}_{variable}_{col}"
                    result_df[new_name] = df[col]
            result_df = result_df.sort_values('fecha')
            return result_df
        except Exception as e:
            self.logger.error(f"Error al cargar el resultado del script: {str(e)}")
            return None
    
    def procesar_archivo(self, config_row):
        """
        Procesa un archivo para una configuración dada según los parámetros del archivo Excel
        """
        variable = config_row['Variable']
        tipo_macro = config_row['Tipo Macro']
        target_col = config_row['TARGET']
        tipo_preprocesamiento = config_row.get('Tipo de Preprocesamiento Según la Fuente', 'Normal')
        self.logger.info(f"\nProcesando: {variable} ({tipo_macro})")
        self.logger.info(f"- Columna TARGET: {target_col}")
        self.logger.info(f"- Tipo de preprocesamiento: {tipo_preprocesamiento}")
        if variable == 'US_Empire_State_Index':
            return self.procesar_empire_state_manualmente(variable, tipo_macro, target_col)
        elif variable == 'AAII_Investor_Sentiment':
            return self.procesar_aaii_investor_sentiment(variable, tipo_macro, target_col)
        elif variable == 'Chicago_Fed_NFCI':
            return self.procesar_chicago_fed_manualmente(variable, tipo_macro, target_col)
        else:
            self.logger.warning(f"No existe procesador específico para {variable}, intentando procesamiento genérico")
            return self.procesar_archivo_generico(variable, tipo_macro, target_col)
    
    def generar_indice_diario(self):
        """
        Genera un DataFrame con índice diario entre fechas mínima y máxima globales
        """
        dfs_validos = {var: df for var, df in self.datos_procesados.items() if df is not None}
        if not dfs_validos:
            self.logger.error("No hay datos procesados válidos para generar el índice diario")
            return None
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
        todas_fechas = pd.date_range(start=self.fecha_min_global, end=self.fecha_max_global, freq='D')
        self.indice_diario = pd.DataFrame({'fecha': todas_fechas})
        self.logger.info(f"- Total de fechas diarias generadas: {len(self.indice_diario)}")
        return self.indice_diario
    
    def combinar_datos(self):
        """
        Combina todos los indicadores procesados con el índice diario
        """
        dfs_validos = {var: df for var, df in self.datos_procesados.items() if df is not None}
        if not dfs_validos:
            self.logger.error("No hay datos procesados válidos para combinar")
            return None
        if self.indice_diario is None:
            self.logger.error("No se ha generado el índice diario")
            return None
        self.logger.info("\nCombinando datos con índice diario...")
        df_combinado = self.indice_diario.copy()
        # Para cada indicador, se normaliza la fecha, se agrupa para asegurar un único registro diario y se hace merge
        for variable, df in dfs_validos.items():
            self.logger.info(f"- Combinando: {variable}")
            df['fecha'] = pd.to_datetime(df['fecha']).dt.normalize()
            df_unique = df.groupby('fecha', as_index=False).last()
            df_combinado = pd.merge(df_combinado, df_unique, on='fecha', how='left')
            for col in df_unique.columns:
                if col != 'fecha':
                    df_combinado[col] = df_combinado[col].ffill()
        self.df_combinado = df_combinado
        self.logger.info(f"- DataFrame combinado: {len(df_combinado)} filas, {len(df_combinado.columns)} columnas")
        return self.df_combinado
    
    def analizar_cobertura_final(self):
        """
        Genera un informe detallado de cobertura final
        """
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
            self.estadisticas[variable]['cobertura_final'] = cobertura_final
            self.logger.info(f"- {variable}: {cobertura_final:.2f}%")
    
    def generar_estadisticas_valores(self):
        """
        Genera estadísticas descriptivas de los valores para cada indicador
        """
        if self.df_combinado is None:
            self.logger.error("No hay datos combinados para analizar")
            return
        self.logger.info("\n" + "=" * 50)
        self.logger.info("ESTADÍSTICAS DE VALORES")
        self.logger.info("=" * 50)
        for col in self.df_combinado.columns:
            if col == 'fecha':
                continue
            serie = self.df_combinado[col].dropna()
            if len(serie) == 0:
                self.logger.warning(f"La columna {col} no tiene valores")
                continue
            stats = {
                'min': serie.min(),
                'max': serie.max(),
                'mean': serie.mean(),
                'median': serie.median(),
                'std': serie.std()
            }
            self.logger.info(f"\nEstadísticas para {col}:")
            self.logger.info(f"- Min: {stats['min']}")
            self.logger.info(f"- Max: {stats['max']}")
            self.logger.info(f"- Media: {stats['mean']}")
            self.logger.info(f"- Mediana: {stats['median']}")
            self.logger.info(f"- Desv. Estándar: {stats['std']}")
            variable = next((var for var in self.estadisticas.keys() if var in col), None)
            if variable:
                col_stats = {f"{col}_min": stats['min'],
                             f"{col}_max": stats['max'],
                             f"{col}_mean": stats['mean'],
                             f"{col}_median": stats['median'],
                             f"{col}_std": stats['std']}
                self.estadisticas[variable].update(col_stats)
    
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
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                self.logger.info(f"Guardando DataFrame combinado en '{output_file}'...")
                self.df_combinado.to_excel(writer, sheet_name='Datos_Combinados', index=False)
                self.logger.info("Guardando estadísticas de los indicadores...")
                stats_data = []
                for variable, stats in self.estadisticas.items():
                    row = {
                        'Variable': variable,
                        'Tipo_Macro': stats.get('tipo_macro', ''),
                        'Columna_TARGET': stats.get('columna_target', ''),
                        'Total_Filas_Original': stats.get('total_filas', 0),
                        'Valores_Validos_Original': stats.get('valores_validos', 0),
                        'Cobertura_Final_%': stats.get('cobertura_final', 0),
                        'Fecha_Min': stats.get('fecha_min', ''),
                        'Fecha_Max': stats.get('fecha_max', '')
                    }
                    for k, v in stats.items():
                        if any(k.startswith(f"{col}_") for col in ['min', 'max', 'mean', 'median', 'std']):
                            row[k] = v
                    stats_data.append(row)
                df_stats = pd.DataFrame(stats_data)
                df_stats.to_excel(writer, sheet_name='Estadisticas', index=False)
                metadata = {
                    'Proceso': ['OtherDataProcessor'],
                    'Fecha de proceso': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Total indicadores': [len(self.estadisticas)],
                    'Periodo': [f"{self.fecha_min_global.strftime('%Y-%m-%d')} a {self.fecha_max_global.strftime('%Y-%m-%d')}"],
                    'Total días': [len(self.indice_diario)]
                }
                pd.DataFrame(metadata).to_excel(writer, sheet_name='Metadatos')
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
        self.leer_configuracion()
        if self.config_data is None or len(self.config_data) == 0:
            return False
        self.logger.info("\nResumen de configuraciones a procesar:")
        for idx, row in self.config_data.iterrows():
            self.logger.info(f"- {row['Variable']} (Tipo Macro: {row['Tipo Macro']}, TARGET: {row['TARGET']})")
        for _, config_row in self.config_data.iterrows():
            variable, df_procesado = self.procesar_archivo(config_row)
            self.datos_procesados[variable] = df_procesado
        archivos_correctos = sum(1 for df in self.datos_procesados.values() if df is not None)
        if archivos_correctos == 0:
            self.logger.error("No se pudo procesar correctamente ningún archivo")
            return False
        self.logger.info(f"\nArchivos procesados correctamente: {archivos_correctos}/{len(self.datos_procesados)}")
        self.generar_indice_diario()
        if self.indice_diario is None:
            return False
        self.combinar_datos()
        if self.df_combinado is None:
            return False
        self.analizar_cobertura_final()
        self.generar_estadisticas_valores()
        resultado = self.guardar_resultados(output_file)
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

# Función principal para ejecutar el proceso
def ejecutar_otherdataprocessor(
    config_file=os.path.join(PROJECT_ROOT, 'pipelines/Data Engineering.xlsx'),
    output_file=os.path.join(PROJECT_ROOT, 'data/0_raw/datos_economicos_other_procesados.xlsx'),
    data_root=os.path.join(PROJECT_ROOT, 'data/0_raw'),
    log_file=os.path.join(PROJECT_ROOT, 'logs/otherdataprocessor.log')
):
    """
    Ejecuta el proceso OtherDataProcessor
    
    Args:
        config_file (str): Ruta al archivo de configuración
        output_file (str): Ruta al archivo de salida
        data_root (str): Directorio raíz donde se encuentran los subdirectorios de datos
        log_file (str): Ruta al archivo de log
        
    Returns:
        bool: True si el proceso se completó exitosamente, False en caso contrario
    """
    procesador = OtherDataProcessor(config_file, data_root, log_file)
    return procesador.ejecutar_proceso_completo(output_file)

# Ejemplo de uso
if __name__ == "__main__":
    resultado = ejecutar_otherdataprocessor()
    print(f"Proceso {'completado exitosamente' if resultado else 'finalizado con errores'}")

# ## Categorizar Columnas

import re
import logging
import pandas as pd

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Definición de patrones regex para cada categoría (buscando al final de la cadena)
regex_patterns = {
    'bond': re.compile(r"_bond$"),
    'business_confidence': re.compile(r"_business_confidence$"),
    'car_registrations': re.compile(r"_car_registrations$"),
    'comm_loans': re.compile(r"_comm_loans$"),
    'commodities': re.compile(r"_commodities$"),
    'consumer_confidence': re.compile(r"_consumer_confidence$"),
    'economics': re.compile(r"_economics$"),
    'exchange_rate': re.compile(r"_exchange_rate$"),
    'exports': re.compile(r"_exports$"),
    'index_pricing': re.compile(r"_index_pricing$"),
    'leading_economic_index': re.compile(r"_leading_economic_index$"),
    'unemployment_rate': re.compile(r"_unemployment_rate$")
}

def categorize_column(col_name: str) -> str:
    """
    Determina la categoría de una columna basándose en el patrón que aparece al final del nombre.
    Retorna la categoría si hay coincidencia o "Sin categoría" en caso contrario.
    """
    for category, pattern in regex_patterns.items():
        if pattern.search(col_name):
            return category
    return "Sin categoría"

def main():
    # 1) Determinar la ruta del repositorio (dos niveles arriba de este archivo)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    # 2) Construir la ruta relativa al Excel
    merge_path = repo_root / "data" / "1_preprocess" / "MERGEDEXCELS.xlsx"

    # 3) Intentar leer el archivo Excel
    try:
        df = pd.read_excel(merge_path, engine="openpyxl")
        columns = df.columns.tolist()
        logging.info(f"Archivo Excel cargado exitosamente desde {merge_path}")
    except FileNotFoundError:
        logging.error(f"No se encontró el fichero: {merge_path}")
        return
    except Exception as e:
        logging.error("Error al leer el archivo Excel: %s", e)
        return

    logging.info("Iniciando la categorización de columnas...")

    # Agrupar columnas por categoría
    grouped_columns = {}
    for col in columns:
        category = categorize_column(col)
        grouped_columns.setdefault(category, []).append(col)

    # Imprimir el resultado en el formato solicitado
    for category, cols in grouped_columns.items():
        print(f"columnas encontradas para {category}:")
        for col in cols:
            print(col)
        print()  # Línea en blanco para separar grupos

    logging.info("Proceso completado exitosamente.")

# =============================================================================
# BANCO REPÚBLICA PROCESSOR - INTEGRADO
# Agregar este código al final de step_0_preprocess.py (antes del if __name__)
# =============================================================================

