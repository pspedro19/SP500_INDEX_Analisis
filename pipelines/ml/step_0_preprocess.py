#pip install openpyxl

# ## MyInvesting Copy-Paste

import pandas as pd
import numpy as np
import os
import re
import time
import logging
from datetime import datetime, timedelta
from dateutil.parser import parse
import warnings
warnings.filterwarnings('ignore')

# Configuración de logging
def configurar_logging(log_file='myinvestingreportcp.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('EconomicDataProcessor')

class EconomicDataProcessor:
    """
    Clase para procesar datos macroeconómicos con robustez en el manejo de fechas y 
    forward filling de series de frecuencia (por ejemplo, mensuales a diarios).

    Funcionalidades:
      - Conversión robusta de cadenas de fecha usando múltiples estrategias.
      - Transformación de series (generalmente mensuales) a datos diarios mediante merge_asof.
      - Renombrado de la columna de valores usando el patrón: 
            {target_col}_{variable}_{tipo_macro}
      - Validación y log detallado en cada etapa.
    """

    def __init__(self, config_file, data_root='data/0_raw', log_file='myinvestingreportcp.log'):
        self.config_file = config_file
        self.data_root = data_root
        self.logger = configurar_logging(log_file)
        self.config_data = None
        self.global_min_date = None
        self.global_max_date = None
        self.daily_index = None
        self.processed_data = {}  # Diccionario {variable: DataFrame procesado}
        self.final_df = None
        self.stats = {}
        # Cache para la preferencia de conversión de fecha
        self.date_cache = {}

        self.logger.info("=" * 80)
        self.logger.info("INICIANDO PROCESO: EconomicDataProcessor")
        self.logger.info(f"Archivo de configuración: {config_file}")
        self.logger.info(f"Directorio raíz de datos: {data_root}")
        self.logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)

    def read_config(self):
        try:
            self.logger.info("Leyendo archivo de configuración...")
            df_config = pd.read_excel(self.config_file)
            self.config_data = df_config[
                (df_config['Fuente'] == 'Investing Data') &
                (df_config['Tipo de Preprocesamiento Según la Fuente'] == 'Copiar y Pegar')
            ].copy()
            self.logger.info(f"Se encontraron {len(self.config_data)} configuraciones para procesar")
            return self.config_data
        except Exception as e:
            self.logger.error(f"Error al leer configuración: {e}")
            return None

    def robust_parse_date(self, date_str, preferred_dayfirst=None):
        """
        Intenta convertir la cadena de fecha usando múltiples estrategias.
        - Primero, busca patrones como "Apr 01, 2025 (Mar)".
        - Luego utiliza pd.to_datetime con la opción dayfirst según la preferencia.
        - Si no se especifica, prueba ambas opciones y elige la que dé una fecha razonable.

        Args:
            date_str (str): Cadena de fecha.
            preferred_dayfirst (bool, opcional): Preferencia de interpretación.

        Returns:
            pd.Timestamp o None.
        """
        if not isinstance(date_str, str):
            return None
        date_str = date_str.strip()
        if not date_str:
            return None

        m = re.search(r'([A-Za-z]+\s+\d{1,2},\s+\d{4})', date_str)
        if m:
            candidate = m.group(1)
            try:
                parsed = pd.to_datetime(candidate, errors='coerce')
                if parsed is not None:
                    return parsed
            except Exception as e:
                self.logger.warning(f"Error al parsear patrón en '{date_str}': {e}")

        if preferred_dayfirst is not None:
            try:
                parsed = pd.to_datetime(date_str, dayfirst=preferred_dayfirst, errors='coerce')
                threshold = pd.Timestamp.today() + pd.Timedelta(days=30)
                if parsed and parsed <= threshold:
                    return parsed
            except Exception as e:
                self.logger.warning(f"Error con dayfirst={preferred_dayfirst} en '{date_str}': {e}")

        try:
            parsed_true = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
            parsed_false = pd.to_datetime(date_str, dayfirst=False, errors='coerce')
            threshold = pd.Timestamp.today() + pd.Timedelta(days=30)
            valid_true = parsed_true and parsed_true <= threshold
            valid_false = parsed_false and parsed_false <= threshold
            if valid_true and not valid_false:
                return parsed_true
            elif valid_false and not valid_true:
                return parsed_false
            elif valid_true and valid_false:
                return parsed_true  # Por defecto dayfirst=True
            else:
                return parsed_true if pd.notnull(parsed_true) else parsed_false
        except Exception as e:
            self.logger.warning(f"Error en robust_parse_date para '{date_str}': {e}")
            return None

    def process_file(self, config_row):
        """
        Procesa un archivo individual:
          - Lee el archivo (se usa estrategia especial para US_Leading_EconIndex).
          - Convierte la columna 'Release Date' robustamente.
          - Detecta y convierte la columna target a numérico.
          - Renombra la columna de valor usando el patrón:
                {target_col}_{variable}_{tipo_macro}
          - Selecciona solo las columnas 'fecha' y la columna renombrada.

        Returns:
            tuple: (variable, DataFrame procesado) o (variable, None) en caso de error.
        """
        variable = config_row['Variable']
        macro_type = config_row['Tipo Macro']
        target_col = config_row['TARGET']

        # Construir la ruta (buscando también en subdirectorios)
        ruta = os.path.join(self.data_root, macro_type, f"{variable}.xlsx")
        if not os.path.exists(ruta):
            for root, dirs, files in os.walk(self.data_root):
                if f"{variable}.xlsx" in files:
                    ruta = os.path.join(root, f"{variable}.xlsx")
                    break
        if not os.path.exists(ruta):
            self.logger.error(f"Archivo no encontrado: {variable}.xlsx")
            return variable, None

        self.logger.info(f"\nProcesando: {variable} ({macro_type})")
        self.logger.info(f"- Archivo: {variable}.xlsx")
        self.logger.info(f"- Columna TARGET: {target_col}")
        self.logger.info(f"- Ruta encontrada: {ruta}")

        try:
            # Estrategia especial para US_Leading_EconIndex: ajustar header y limpiar columnas
            if variable == "US_Leading_EconIndex":
                self.logger.info("Utilizando estrategia especial para US_Leading_EconIndex (header=2)")
                df = pd.read_excel(ruta, header=2, engine='openpyxl')
                df.columns = df.columns.str.strip()
                self.logger.info(f"Columnas leídas: {df.columns.tolist()}")
            else:
                df = pd.read_excel(ruta, engine='openpyxl')
        except Exception as e:
            self.logger.error(f"Error al leer {ruta}: {e}")
            return variable, None

        self.logger.info(f"- Filas encontradas: {len(df)}")
        if 'Release Date' not in df.columns:
            self.logger.error(f"No se encontró la columna 'Release Date' en {ruta}")
            return variable, None

        # Determinar preferencia de dayfirst (cacheada)
        if ruta not in self.date_cache:
            sample = df['Release Date'].dropna().head(10)
            count_true, count_false = 0, 0
            threshold = pd.Timestamp.today() + pd.Timedelta(days=30)
            for val in sample:
                dt_true = pd.to_datetime(val, dayfirst=True, errors='coerce')
                dt_false = pd.to_datetime(val, dayfirst=False, errors='coerce')
                if pd.notnull(dt_true) and dt_true <= threshold:
                    count_true += 1
                if pd.notnull(dt_false) and dt_false <= threshold:
                    count_false += 1
            preferred = count_true >= count_false
            self.date_cache[ruta] = preferred
            self.logger.info(f"Preferencia de dayfirst para {ruta}: {preferred}")
        else:
            preferred = self.date_cache[ruta]

        df['fecha'] = df['Release Date'].apply(lambda x: self.robust_parse_date(x, preferred_dayfirst=preferred))
        df = df.dropna(subset=['fecha'])
        df = df.sort_values('fecha')

        # Si el target especificado no está, intenta buscar una alternativa
        if target_col not in df.columns:
            for col in df.columns:
                if col.strip().lower() == target_col.strip().lower():
                    target_col = col
                    self.logger.warning(f"No se encontró '{config_row['TARGET']}', se usará '{target_col}'")
                    break
        if target_col not in df.columns:
            self.logger.error(f"No se encontró columna TARGET ni alternativa en {ruta}")
            return variable, None

        # Convertir la columna target a numérico y descartar valores no válidos
        df['valor'] = pd.to_numeric(df[target_col], errors='coerce')
        df = df.dropna(subset=['valor'])
        if df.empty:
            self.logger.error(f"No se encontraron valores válidos para '{target_col}' en {ruta}")
            return variable, None

        # Actualizar rango global de fechas
        current_min = df['fecha'].min()
        current_max = df['fecha'].max()
        if self.global_min_date is None or current_min < self.global_min_date:
            self.global_min_date = current_min
        if self.global_max_date is None or current_max > self.global_max_date:
            self.global_max_date = current_max

        # Calcular cobertura (puedes ajustar la fórmula si lo deseas)
        cobertura = (len(df) / len(df)) * 100

        # RENOMBRAR LA COLUMNA: Crear un nombre único
        nuevo_nombre = f"{target_col}_{variable}_{macro_type}"
        df.rename(columns={'valor': nuevo_nombre}, inplace=True)
        self.stats[variable] = {
            'macro_type': macro_type,
            'target_column': target_col,
            'total_rows': len(df),
            'valid_values': len(df),
            'coverage': cobertura,
            'date_min': current_min,
            'date_max': current_max,
            'nuevo_nombre': nuevo_nombre
        }
        self.logger.info(f"- Valores no nulos en TARGET: {len(df)}")
        self.logger.info(f"- Periodo: {current_min.strftime('%Y-%m-%d')} a {current_max.strftime('%Y-%m-%d')}")
        self.logger.info(f"- Cobertura: {cobertura:.2f}%")
        return variable, df[['fecha', nuevo_nombre]].copy()

    def generate_daily_index(self):
        """
        Genera un DataFrame con un índice diario desde la fecha global mínima hasta la máxima.
        """
        if self.global_min_date is None or self.global_max_date is None:
            self.logger.error("No se pudieron determinar las fechas globales")
            return None
        self.daily_index = pd.DataFrame({
            'fecha': pd.date_range(start=self.global_min_date, end=self.global_max_date, freq='D')
        })
        self.logger.info(f"Índice diario generado: {len(self.daily_index)} días desde {self.global_min_date.strftime('%Y-%m-%d')} hasta {self.global_max_date.strftime('%Y-%m-%d')}")
        return self.daily_index

    def combine_data(self):
        """
        Convierte cada serie (generalmente reportada en frecuencias bajas) a datos diarios usando merge_asof.
        Para cada archivo, se asocia el dato reportado más reciente a cada día.
        """
        if self.daily_index is None:
            self.logger.error("El índice diario no ha sido generado")
            return None

        combined = self.daily_index.copy()
        for variable, df in self.processed_data.items():
            if df is None or df.empty:
                self.logger.warning(f"Omitiendo {variable} por falta de datos")
                continue
            df = df.sort_values('fecha')
            # merge_asof para asignar cada día con el valor reportado más reciente
            df_daily = pd.merge_asof(combined, df, on='fecha', direction='backward')
            col_name = self.stats[variable]['nuevo_nombre']
            # Como precaución se aplica ffill
            df_daily[col_name] = df_daily[col_name].ffill()
            combined = combined.merge(df_daily[['fecha', col_name]], on='fecha', how='left')
        self.final_df = combined
        self.logger.info(f"DataFrame final combinado: {len(self.final_df)} filas, {len(self.final_df.columns)} columnas")
        return self.final_df

    def analyze_coverage(self):
        """
        Genera un resumen de cobertura y estadísticas para cada indicador.
        """
        total_days = len(self.daily_index)
        self.logger.info("\nResumen de Cobertura:")
        for variable, stats in self.stats.items():
            self.logger.info(f"- {variable}: {stats['coverage']:.2f}% desde {stats['date_min'].strftime('%Y-%m-%d')} a {stats['date_max'].strftime('%Y-%m-%d')}")

    def save_results(self, output_file='datos_economicos_procesados.xlsx'):
        """
        Guarda el DataFrame final combinado y las estadísticas en un archivo Excel.
        """
        if self.final_df is None:
            self.logger.error("No hay datos combinados para guardar")
            return False
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                self.final_df.to_excel(writer, sheet_name='Datos Diarios', index=False)
                df_stats = pd.DataFrame(self.stats).T
                df_stats.to_excel(writer, sheet_name='Estadisticas')
                meta = {
                    'Proceso': ['EconomicDataProcessor'],
                    'Fecha de proceso': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Total indicadores': [len(self.stats)],
                    'Periodo': [f"{self.global_min_date.strftime('%Y-%m-%d')} a {self.global_max_date.strftime('%Y-%m-%d')}"],
                    'Total días': [len(self.daily_index)]
                }
                pd.DataFrame(meta).to_excel(writer, sheet_name='Metadatos', index=False)
            self.logger.info(f"Archivo guardado exitosamente: {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar resultados: {e}")
            return False

    def run(self, output_file='datos_economicos_procesados.xlsx'):
        """
        Ejecuta el proceso completo:
          1. Lee la configuración.
          2. Procesa cada archivo.
          3. Determina el rango global de fechas.
          4. Genera el índice diario.
          5. Convierte cada serie a datos diarios y los combina.
          6. Realiza un análisis de cobertura.
          7. Guarda los resultados.
        """
        start_time = time.time()
        if self.read_config() is None:
            return False

        for _, config_row in self.config_data.iterrows():
            var, df_processed = self.process_file(config_row)
            self.processed_data[var] = df_processed

        if len([df for df in self.processed_data.values() if df is not None]) == 0:
            self.logger.error("No se procesó ningún archivo correctamente")
            return False

        self.generate_daily_index()
        self.combine_data()
        self.analyze_coverage()
        result = self.save_results(output_file)
        end_time = time.time()
        self.logger.info("\nResumen de Ejecución:")
        self.logger.info(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")
        self.logger.info(f"Archivos procesados: {len(self.config_data)}")
        self.logger.info(f"Archivo de salida: {output_file}")
        self.logger.info(f"Estado: {'COMPLETADO' if result else 'ERROR'}")
        return result


# Al inicio del script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def run_economic_data_processor(config_file=os.path.join(PROJECT_ROOT, 'pipelines/Data Engineering.xlsx'),
                               output_file=os.path.join(PROJECT_ROOT, 'data/0_raw/datos_economicos_procesados_cp.xlsx'),
                               data_root=os.path.join(PROJECT_ROOT, 'data/0_raw'),
                               log_file=os.path.join(PROJECT_ROOT, 'logs/myinvestingreportcp.log')):
    processor = EconomicDataProcessor(config_file, data_root, log_file)
    return processor.run(output_file)

# Ejemplo de uso
if __name__ == "__main__":
    success = run_economic_data_processor()
    print(f"Proceso {'completado exitosamente' if success else 'finalizado con errores'}")

# ## MyInvesting Normal

import pandas as pd
import numpy as np
import os
import re
import time
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------
# Configuración de logging
# ---------------------------------------------------
def configurar_logging(log_file='myinvestingreportnormal.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('MyinvestingreportNormal')

# ---------------------------------------------------
# Función para convertir valores numéricos
# ---------------------------------------------------
def convertir_valor(valor, variable=None, formatos_conocidos=None):
    """
    Convierte cualquier representación de valor numérico a float.
    """
    if isinstance(valor, (int, float)):
        return float(valor)
    
    if not isinstance(valor, str) or valor is None:
        return None

    valor_limpio = valor.strip()
    if not valor_limpio:
        return None

    # Aplicar formato conocido si existe
    if variable and formatos_conocidos and variable in formatos_conocidos:
        formato = formatos_conocidos[variable]
        if formato == 'europeo':
            valor_limpio = valor_limpio.replace('.', '')
            valor_limpio = valor_limpio.replace(',', '.')
    
    # Multiplicadores para sufijos (K, M, etc.)
    multiplicadores = {'%': 1, 'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
    multiplicador = 1
    for sufijo, mult in multiplicadores.items():
        if valor_limpio.endswith(sufijo):
            valor_limpio = valor_limpio.replace(sufijo, '')
            multiplicador = mult
            break

    # Ajuste de separadores: si se detectan ambos, se decide según la posición
    if ',' in valor_limpio and '.' in valor_limpio:
        if valor_limpio.rfind(',') > valor_limpio.rfind('.'):
            valor_limpio = valor_limpio.replace('.', '')
            valor_limpio = valor_limpio.replace(',', '.')
        else:
            valor_limpio = valor_limpio.replace(',', '')
    elif ',' in valor_limpio:
        partes = valor_limpio.split(',')
        if len(partes) == 2 and len(partes[1]) <= 2:
            valor_limpio = valor_limpio.replace(',', '.')
        else:
            valor_limpio = valor_limpio.replace(',', '')
    
    try:
        return float(valor_limpio) * multiplicador
    except (ValueError, TypeError):
        return None

# ---------------------------------------------------
# Detección dinámica del formato de fechas
# ---------------------------------------------------
def detectar_formato_fecha_inteligente(df, col_fecha, muestra_registros=10):
    """
    Analiza una muestra de la columna de fecha para determinar si se debe forzar dayfirst=True.
    Retorna un diccionario con {'dayfirst': bool, 'confianza': float}
    """
    fecha_actual = pd.Timestamp(datetime.now().date())
    muestras = df[col_fecha].dropna().astype(str).head(muestra_registros).tolist()
    
    resultados = {
        'dayfirst': {'validas': 0, 'invalidas': 0, 'futuras': 0},
        'no_dayfirst': {'validas': 0, 'invalidas': 0, 'futuras': 0}
    }
    
    for fecha_str in muestras:
        fecha_str = fecha_str.strip()
        ambigua = False
        # Identifica fechas con formato numérico separado por /, - o .
        if re.match(r'^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4}$', fecha_str):
            separador = re.findall(r'[\/\-\.]', fecha_str)[0]
            partes = fecha_str.split(separador)
            if len(partes) == 3:
                try:
                    p1, p2 = int(partes[0]), int(partes[1])
                    # Si ambos números son menores o iguales a 12, es ambigua
                    if p1 <= 12 and p2 <= 12:
                        ambigua = True
                except:
                    pass

        for modo, dayfirst in [('dayfirst', True), ('no_dayfirst', False)]:
            try:
                fecha = pd.to_datetime(fecha_str, dayfirst=dayfirst)
                if fecha > fecha_actual + pd.Timedelta(days=30):
                    resultados[modo]['futuras'] += 1
                else:
                    resultados[modo]['validas'] += 1
            except:
                resultados[modo]['invalidas'] += 1

    score_dayfirst = resultados['dayfirst']['validas'] - (resultados['dayfirst']['invalidas'] * 0.5) - (resultados['dayfirst']['futuras'] * 2)
    score_no_dayfirst = resultados['no_dayfirst']['validas'] - (resultados['no_dayfirst']['invalidas'] * 0.5) - (resultados['no_dayfirst']['futuras'] * 2)
    usar_dayfirst = score_dayfirst > score_no_dayfirst
    confianza = abs(score_dayfirst - score_no_dayfirst) / (muestra_registros * 2)
    return {'dayfirst': usar_dayfirst, 'confianza': confianza}

# ---------------------------------------------------
# Conversión adaptativa de fechas
# ---------------------------------------------------
def convertir_fecha_adaptativo(fecha_str, configuracion_archivo=None):
    """
    Convierte una fecha a pd.Timestamp usando la configuración detectada.
    Si la cadena contiene puntos (.) y coincide con el patrón DD.MM.YYYY, se fuerza el formato.
    """
    # Si ya es Timestamp o datetime, retornarla
    if isinstance(fecha_str, (pd.Timestamp, datetime)):
        return pd.Timestamp(fecha_str)
    if pd.isna(fecha_str):
        return None
    fecha_str = str(fecha_str).strip()
    # Si se detecta un punto como separador y el patrón es DD.MM.YYYY, forzamos el formato
    if re.match(r'^\d{1,2}\.\d{1,2}\.\d{4}$', fecha_str):
        try:
            fecha = pd.to_datetime(fecha_str, format='%d.%m.%Y', dayfirst=True)
            return fecha
        except Exception:
            pass

    # Si hay configuración detectada, usarla
    if configuracion_archivo is not None:
        try:
            fecha = pd.to_datetime(fecha_str, dayfirst=configuracion_archivo['dayfirst'])
            return fecha
        except Exception:
            pass

    # Intento estándar sin forzar dayfirst
    try:
        fecha = pd.to_datetime(fecha_str)
        return fecha
    except Exception:
        return None

# ---------------------------------------------------
# Clase para gestionar y cachear formatos de fechas
# ---------------------------------------------------
class FormatosFechas:
    """Gestiona y cachea la configuración de conversión de fechas por archivo."""
    def __init__(self):
        self.formatos_cache = {}  # {variable: configuracion}
    
    def detectar_formato(self, df, col_fecha, variable=None):
        configuracion = detectar_formato_fecha_inteligente(df, col_fecha)
        if variable:
            self.formatos_cache[variable] = configuracion
        return configuracion
        
    def obtener_formato(self, variable):
        return self.formatos_cache.get(variable)

# ---------------------------------------------------
# Clase principal para procesar los datos económicos
# ---------------------------------------------------
class MyinvestingreportNormal:
    """
    Procesa datos económicos con detección dinámica de formatos de fechas y 
    validación para evitar interpretaciones erróneas (como fechas en abril cuando
    los datos crudos solo llegan hasta marzo).
    """
    def __init__(self, config_file, data_root='data/raw', log_file='myinvestingreportnormal.log'):
        self.config_file = config_file
        self.data_root = data_root
        self.logger = configurar_logging(log_file)
        self.config_data = None
        self.fecha_min_global = None
        self.fecha_max_global = None
        self.archivo_fecha_min = None
        self.archivo_fecha_max = None
        self.indice_diario = None
        self.datos_procesados = {}
        self.df_combinado = None
        self.estadisticas = {}

        self.formatos_numericos = {}  # Se usará un diccionario para formatos numéricos simples (si se desea ampliar)
        self.formatos_fechas = FormatosFechas()
        self.formatos_conocidos = {}  # Configuraciones exitosas de CSV

        self.inicializar_formatos_conocidos()

        self.logger.info("=" * 80)
        self.logger.info("INICIANDO PROCESO: MyinvestingreportNormal")
        self.logger.info(f"Archivo de configuración: {config_file}")
        self.logger.info(f"Directorio raíz de datos: {data_root}")
        self.logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)

    def inicializar_formatos_conocidos(self):
        # Para algunos índices se conoce el formato numérico; aquí se puede ampliar si es necesario.
        indices_europeos = ['Russell_2000', 'NASDAQ_Composite', 'S&P500_Index', 'Nikkei_225', 'DAX_30', 'VIX_VolatilityIndex']
        for indice in indices_europeos:
            self.formatos_conocidos[indice] = 'europeo'

    def leer_configuracion(self):
        self.logger.info("Leyendo archivo de configuración...")
        try:
            df_config = pd.read_excel(self.config_file)
            self.config_data = df_config[
                (df_config['Fuente'] == 'Investing Data') &
                (df_config['Tipo de Preprocesamiento Según la Fuente'] == 'Normal')
            ].copy()
            num_configs = len(self.config_data)
            self.logger.info(f"Se encontraron {num_configs} configuraciones para procesar")
            if num_configs == 0:
                self.logger.warning("No se encontraron configuraciones que cumplan los criterios")
                return None
            return self.config_data
        except Exception as e:
            self.logger.error(f"Error al leer configuración: {str(e)}")
            return None

    def encontrar_ruta_archivo(self, variable, tipo_macro):
        ruta_base = os.path.join(self.data_root, tipo_macro)
        nombre_archivo = f"{variable}.csv"
        ruta_completa = os.path.join(ruta_base, nombre_archivo)
        if os.path.exists(ruta_completa):
            return ruta_completa
        nombre_archivo_alt = f"{variable}.xlsx"
        ruta_completa_alt = os.path.join(ruta_base, nombre_archivo_alt)
        if os.path.exists(ruta_completa_alt):
            return ruta_completa_alt
        for root, dirs, files in os.walk(self.data_root):
            if nombre_archivo in files:
                return os.path.join(root, nombre_archivo)
            if nombre_archivo_alt in files:
                return os.path.join(root, nombre_archivo_alt)
        return None

    def leer_csv_adaptativo(self, ruta_archivo, variable):
        configuraciones = [
            {'sep': ',', 'decimal': '.', 'encoding': 'utf-8'},
            {'sep': ',', 'decimal': '.', 'encoding': 'latin1'},
            {'sep': ';', 'decimal': ',', 'thousands': '.', 'encoding': 'utf-8'},
            {'sep': ';', 'decimal': ',', 'thousands': '.', 'encoding': 'latin1'},
            {'sep': ',', 'decimal': ',', 'thousands': '.', 'encoding': 'utf-8'},
            {'sep': '\t', 'encoding': 'utf-8'},
            {'sep': ' ', 'encoding': 'utf-8'}
        ]
        if variable in self.formatos_conocidos:
            config_conocida = self.formatos_conocidos[variable]
            try:
                df = pd.read_csv(ruta_archivo, **config_conocida)
                if len(df) > 0:
                    return df
            except Exception:
                pass
        errores = []
        for idx, config in enumerate(configuraciones):
            try:
                df = pd.read_csv(ruta_archivo, **config)
                if len(df) > 0:
                    self.formatos_conocidos[variable] = config
                    return df
            except Exception as e:
                errores.append(f"Config {idx}: {str(e)}")
        self.logger.error(f"No se pudo leer {ruta_archivo} con ninguna configuración")
        for error in errores:
            self.logger.debug(f"- {error}")
        return None

    def detectar_columnas(self, df):
        candidatos_fecha = [col for col in df.columns if any(
            palabra in col.lower() for palabra in ['date', 'fecha', 'time', 'día', 'day', 'periodo']
        )]
        candidatos_valor = [col for col in df.columns if any(
            palabra in col.lower() for palabra in ['price', 'precio', 'close', 'cierre', 'último', 'ultimo', 'valor', 'value']
        )]
        columnas_datetime = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        columna_fecha = None
        if columnas_datetime:
            columna_fecha = columnas_datetime[0]
        elif candidatos_fecha:
            for col in candidatos_fecha:
                try:
                    pd.to_datetime(df[col].iloc[:5])
                    columna_fecha = col
                    break
                except Exception:
                    continue
        if columna_fecha is None and len(df.columns) > 0:
            try:
                pd.to_datetime(df[df.columns[0]].iloc[:5])
                columna_fecha = df.columns[0]
            except Exception:
                pass

        columna_valor = None
        if candidatos_valor:
            columna_valor = candidatos_valor[0]
        elif len(df.columns) > 1:
            columna_valor = df.columns[1]
        return columna_fecha, columna_valor

    def convertir_fecha(self, fecha_str, configuracion_archivo=None):
        return convertir_fecha_adaptativo(fecha_str, configuracion_archivo)

    def limpiar_valor_porcentaje(self, valor, variable=None):
        return convertir_valor(valor, variable, self.formatos_conocidos)

    def procesar_archivo(self, config_row):
        variable = config_row['Variable']
        tipo_macro = config_row['Tipo Macro']
        target_col = config_row['TARGET']

        ruta_archivo = self.encontrar_ruta_archivo(variable, tipo_macro)
        self.logger.info(f"\nProcesando: {variable} ({tipo_macro})")
        self.logger.info(f"- Archivo: {variable}")
        self.logger.info(f"- Columna TARGET: {target_col}")
        if ruta_archivo is None:
            self.logger.error(f"- ERROR: Archivo no encontrado: {variable}")
            return variable, None
        self.logger.info(f"- Ruta encontrada: {ruta_archivo}")

        try:
            _, extension = os.path.splitext(ruta_archivo)
            extension = extension.lower()
            if extension == '.csv':
                df = self.leer_csv_adaptativo(ruta_archivo, variable)
            elif extension in ['.xlsx', '.xls']:
                df = pd.read_excel(ruta_archivo, engine='openpyxl')
            else:
                self.logger.error(f"- ERROR: Formato de archivo no soportado: {extension}")
                return variable, None
            if df is None or len(df) == 0:
                self.logger.error("- ERROR: El archivo está vacío o no se pudo leer")
                return variable, None

            col_fecha, col_valor = self.detectar_columnas(df)
            if col_fecha is None or col_valor is None:
                self.logger.error("- ERROR: No se pudieron detectar columnas necesarias (fecha o valor)")
                return variable, None

            self.logger.info(f"Columna de fecha detectada: {col_fecha}")
            # Detección dinámica del formato de fechas
            config_fecha = self.formatos_fechas.detectar_formato(df, col_fecha, variable)
            self.logger.info(f"Formato de fecha detectado para {variable}: dayfirst={config_fecha['dayfirst']} (confianza: {config_fecha['confianza']:.2f})")
            
            # Convertir fechas usando la configuración detectada
            df['fecha'] = df[col_fecha].apply(lambda x: self.convertir_fecha(x, configuracion_archivo=config_fecha))
            df = df.dropna(subset=['fecha'])
            ejemplos = df['fecha'].head(5).tolist()
            self.logger.info(f"Ejemplos de fechas convertidas para {variable}: {ejemplos}")
            
            # Validar fechas excesivamente futuras (más de 30 días a partir de hoy)
            fecha_actual = pd.Timestamp(datetime.now().date())
            fechas_futuras = df[df['fecha'] > fecha_actual + pd.Timedelta(days=30)]
            if len(fechas_futuras) > 0:
                self.logger.warning(f"Se detectaron {len(fechas_futuras)} fechas futuras anómalas en {variable}")
                self.logger.warning(f"Ejemplos de fechas futuras: {fechas_futuras['fecha'].head(3).tolist()}")
                df = df[df['fecha'] <= fecha_actual + pd.Timedelta(days=30)].copy()
            
            muestra_valores = df[col_valor].astype(str).head(20).tolist()
            formato_detectado = "americano"  # Se usa como etiqueta básica
            self.logger.info(f"Formato numérico detectado para {variable}: {formato_detectado}")

            df['valor'] = df[col_valor].apply(lambda x: self.limpiar_valor_porcentaje(x, variable))
            df = df.dropna(subset=['valor'])

            total_filas = len(df)
            if total_filas == 0:
                self.logger.error(f"- ERROR: No se encontraron valores válidos en {variable}")
                return variable, None

            nuevo_nombre = f"{target_col}_{variable}_{tipo_macro}"
            df.rename(columns={'valor': nuevo_nombre}, inplace=True)
            df_procesado = df[['fecha', nuevo_nombre]].copy()
            df_procesado = df_procesado.sort_values('fecha')

            fecha_min = df_procesado['fecha'].min()
            fecha_max = df_procesado['fecha'].max()
            self.logger.info(f"Para {variable} (columna {col_fecha}), la fecha mínima es {fecha_min} y la fecha máxima es {fecha_max}")
            
            if self.fecha_min_global is None or fecha_min < self.fecha_min_global:
                self.fecha_min_global = fecha_min
                self.archivo_fecha_min = variable
            if self.fecha_max_global is None or fecha_max > self.fecha_max_global:
                self.fecha_max_global = fecha_max
                self.archivo_fecha_max = variable

            self.estadisticas[variable] = {
                'tipo_macro': tipo_macro,
                'columna_target': target_col,
                'total_filas': total_filas,
                'valores_validos': df_procesado[nuevo_nombre].count(),
                'fecha_min': fecha_min,
                'fecha_max': fecha_max,
                'nuevo_nombre': nuevo_nombre,
                'formato_fecha': f"dayfirst={config_fecha['dayfirst']}",
                'confianza_formato': config_fecha['confianza']
            }
            self.logger.info(f"- {variable}: {total_filas} filas procesadas, periodo: {fecha_min.strftime('%Y-%m-%d')} a {fecha_max.strftime('%Y-%m-%d')}")
            return variable, df_procesado
        except Exception as e:
            self.logger.error(f"- ERROR al procesar {ruta_archivo}: {str(e)}")
            return variable, None

    def generar_indice_diario(self):
        if self.fecha_min_global is None or self.fecha_max_global is None:
            self.logger.error("No se pudieron determinar fechas mínima y máxima globales")
            return None
        self.logger.info("\nGenerando índice temporal diario...")
        self.logger.info(f"Archivo con fecha mínima global: {self.archivo_fecha_min} ({self.fecha_min_global.strftime('%Y-%m-%d')})")
        self.logger.info(f"Archivo con fecha máxima global: {self.archivo_fecha_max} ({self.fecha_max_global.strftime('%Y-%m-%d')})")
        todas_fechas = pd.date_range(start=self.fecha_min_global, end=self.fecha_max_global, freq='D')
        self.indice_diario = pd.DataFrame({'fecha': todas_fechas})
        self.logger.info(f"- Total de fechas diarias generadas: {len(self.indice_diario)}")
        return self.indice_diario

    def combinar_datos(self):
        if not self.datos_procesados:
            self.logger.error("No hay datos procesados para combinar")
            return None
        if self.indice_diario is None:
            self.logger.error("No se ha generado el índice diario")
            return None
        self.logger.info("\nCombinando datos con índice diario (usando join para reducir consumo de memoria)...")
        df_combinado = self.indice_diario.copy().set_index('fecha')
        for variable, df in self.datos_procesados.items():
            if df is None:
                self.logger.warning(f"Omitiendo {variable} por errores de procesamiento")
                continue
            nombre_col = df.columns[1]
            self.logger.info(f"- Combinando: {nombre_col}")
            df_temp = df.set_index('fecha')[[nombre_col]]
            df_temp = df_temp[~df_temp.index.duplicated(keep='first')]
            df_temp = df_temp.reindex(df_combinado.index)
            df_temp = df_temp.ffill()
            df_combinado = df_combinado.join(df_temp)
        self.df_combinado = df_combinado.reset_index()
        self.logger.info(f"- DataFrame combinado: {len(self.df_combinado)} filas, {len(self.df_combinado.columns)} columnas")
        return self.df_combinado

    def analizar_cobertura_final(self):
        if self.df_combinado is None or not self.estadisticas:
            self.logger.error("No hay datos combinados o estadísticas para analizar")
            return
        self.logger.info("\n" + "=" * 50)
        self.logger.info("RESUMEN DE COBERTURA FINAL")
        self.logger.info("=" * 50)
        total_indicadores = len(self.estadisticas)
        total_dias = len(self.indice_diario)
        self.logger.info(f"Total indicadores procesados: {total_indicadores}")
        self.logger.info(f"Rango de fechas: {self.fecha_min_global.strftime('%Y-%m-%d')} a {self.fecha_max_global.strftime('%Y-%m-%d')}")
        self.logger.info(f"Total días en la serie: {total_dias}")
        for variable, stats in self.estadisticas.items():
            cobertura = (stats['valores_validos'] / total_dias) * 100
            self.logger.info(f"- {variable} ({stats['nuevo_nombre']}): Cobertura aproximada {cobertura:.2f}%")

    def generar_estadisticas_valores(self):
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
            self.logger.info(f"- Min: {stats['min']:.4f}")
            self.logger.info(f"- Max: {stats['max']:.4f}")
            self.logger.info(f"- Media: {stats['mean']:.4f}")
            self.logger.info(f"- Mediana: {stats['median']:.4f}")
            self.logger.info(f"- Desv. Estándar: {stats['std']:.4f}")
            if col in self.estadisticas:
                self.estadisticas[col] = {**self.estadisticas.get(col, {}), **stats}

    def guardar_resultados(self, output_file='datos_economicos_normales_procesados.xlsx'):
        if self.df_combinado is None:
            self.logger.error("No hay datos para guardar")
            return False
        try:
            self.logger.info("\n" + "=" * 50)
            self.logger.info("GUARDANDO RESULTADOS")
            self.logger.info("=" * 50)
            self.logger.info(f"Guardando resultados en: {output_file}")
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                self.df_combinado.to_excel(writer, sheet_name='Datos Diarios', index=False)
                df_stats = pd.DataFrame()
                for var, stats in self.estadisticas.items():
                    serie = pd.Series(stats, name=var)
                    df_temp = pd.DataFrame(serie).transpose()
                    df_stats = pd.concat([df_stats, df_temp])
                df_stats.to_excel(writer, sheet_name='Estadisticas')
                metadata = {
                    'Proceso': ['MyinvestingreportNormal'],
                    'Fecha de proceso': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Total indicadores': [len(self.estadisticas)],
                    'Periodo': [f"{self.fecha_min_global.strftime('%Y-%m-%d')} a {self.fecha_max_global.strftime('%Y-%m-%d')}"],
                    'Total días': [len(self.indice_diario)]
                }
                pd.DataFrame(metadata).to_excel(writer, sheet_name='Metadatos')
            self.logger.info(f"Archivo guardado exitosamente: {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar resultados: {str(e)}")
            return False

    def ejecutar_proceso_completo(self, output_file='datos_economicos_normales_procesados.xlsx'):
        inicio = time.time()
        self.logger.info("Iniciando proceso completo MyinvestingreportNormal...")
        self.leer_configuracion()
        if self.config_data is None or len(self.config_data) == 0:
            return False
        for _, config_row in self.config_data.iterrows():
            variable, df_procesado = self.procesar_archivo(config_row)
            self.datos_procesados[variable] = df_procesado
        archivos_correctos = sum(1 for df in self.datos_procesados.values() if df is not None)
        if archivos_correctos == 0:
            self.logger.error("No se pudo procesar correctamente ningún archivo")
            return False
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
        self.logger.info(f"Proceso: MyinvestingreportNormal")
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

def ejecutar_myinvestingreportnormal(
    config_file=os.path.join(PROJECT_ROOT, 'pipelines/Data Engineering.xlsx'),
    output_file=os.path.join(PROJECT_ROOT, 'data/0_raw/datos_economicos_normales_procesados.xlsx'),
    data_root=os.path.join(PROJECT_ROOT, 'data/0_raw'),
    log_file=os.path.join(PROJECT_ROOT, 'logs/myinvestingreportnormal.log')
):
    procesador = MyinvestingreportNormal(config_file, data_root, log_file)
    return procesador.ejecutar_proceso_completo(output_file)

if __name__ == "__main__":
    resultado = ejecutar_myinvestingreportnormal()
    print(f"Proceso {'completado exitosamente' if resultado else 'finalizado con errores'}")

# ## FRED-NORMAL

import pandas as pd
import numpy as np
import os
import re
import time
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de logging
def configurar_logging(log_file='freddataprocessor.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('FredDataProcessor')

class FredDataProcessor:
    """
    Clase para procesar datos de FRED (Fuente "FRED" y Preprocesamiento "Normal").

    Se espera que cada archivo tenga:
      - Una columna de fecha llamada "observation_date" (o "DATE" si no existe).
      - Una columna de datos cuyo nombre se especifica en el archivo de configuración (TARGET).

    Esta versión:
      - Busca el archivo usando varias extensiones: .csv, .xlsx, .xls.
      - Detecta dinámicamente el formato de fecha analizando hasta 20 registros.
         Si la mayoría siguen el formato ISO (YYYY-MM-DD), se fuerza ese formato;
         de lo contrario se evalúan ambas opciones (dayfirst True/False) usando la monotonicidad.
      - La función improved_robust_parse_date ahora maneja objetos Timestamp y valores no-string.
      - Convierte la columna de fecha y la columna target a numérico.
      - Renombra la columna de datos con el patrón: {TARGET}_{variable}_{Tipo_Macro}.
      - Genera un índice diario global y usa merge_asof para imputar los datos (forward fill).
    """

    def __init__(self, config_file, data_root='data/raw', log_file='freddataprocessor.log'):
        self.config_file = config_file
        self.data_root = data_root
        self.logger = configurar_logging(log_file)
        self.config_data = None
        self.global_min_date = None
        self.global_max_date = None
        self.daily_index = None
        self.processed_data = {}   # {variable: DataFrame procesado}
        self.final_df = None
        self.stats = {}
        self.date_cache = {}  # Guarda la preferencia de formato para cada archivo

        self.logger.info("=" * 80)
        self.logger.info("INICIANDO PROCESO: FredDataProcessor")
        self.logger.info(f"Archivo de configuración: {config_file}")
        self.logger.info(f"Directorio raíz de datos: {data_root}")
        self.logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)

    def read_config(self):
        try:
            self.logger.info("Leyendo archivo de configuración...")
            df_config = pd.read_excel(self.config_file)
            self.config_data = df_config[
                (df_config['Fuente'] == 'FRED') &
                (df_config['Tipo de Preprocesamiento Según la Fuente'] == 'Normal')
            ].copy()
            self.logger.info(f"Se encontraron {len(self.config_data)} configuraciones para procesar")
            return self.config_data
        except Exception as e:
            self.logger.error(f"Error al leer configuración: {e}")
            return None

    def detect_date_format(self, series, n=20, iso_threshold=0.6):
        """
        Analiza hasta n registros de la serie de fechas para determinar si la mayoría 
        siguen el formato ISO (YYYY-MM-DD).

        Returns:
            "ISO" si al menos iso_threshold de los registros coinciden con el patrón ISO,
            de lo contrario "AMBIGUOUS".
        """
        sample = series.dropna().head(n)
        if len(sample) == 0:
            return "AMBIGUOUS"
        iso_count = 0
        for val in sample:
            if isinstance(val, str) and re.match(r'^\d{4}-\d{2}-\d{2}$', val.strip()):
                iso_count += 1
        ratio = iso_count / len(sample)
        self.logger.info(f"Detección formato: {iso_count}/{len(sample)} registros ISO (ratio {ratio:.2f})")
        return "ISO" if ratio >= iso_threshold else "AMBIGUOUS"

    def monotonic_score(self, parsed_series):
        """
        Calcula la puntuación de monotonicidad de una serie de fechas.
        Es la proporción de diferencias no negativas respecto al total.
        """
        parsed = parsed_series.dropna()
        if len(parsed) < 2:
            return 0
        diffs = parsed.diff().dropna()
        score = (diffs >= timedelta(0)).sum() / len(diffs)
        return score

    def improved_robust_parse_date(self, date_str, preferred_dayfirst=None, use_iso=False):
        """
        Convierte una cadena o timestamp de fecha.

        Args:
            date_str: Cadena de fecha o timestamp.
            preferred_dayfirst (bool, opcional): Preferencia para la conversión.
            use_iso (bool): Si se debe forzar el formato ISO (YYYY-MM-DD).

        Returns:
            pd.Timestamp o None.
        """
        # Si ya es un timestamp, devolverlo directamente
        if isinstance(date_str, pd.Timestamp):
            return date_str

        if not isinstance(date_str, str):
            self.logger.debug(f"Valor no string en fecha: {date_str} (tipo: {type(date_str)})")
            return None

        date_str = date_str.strip()
        if not date_str:
            return None

        if use_iso:
            try:
                return pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
            except Exception as e:
                self.logger.warning(f"Error al convertir formato ISO en '{date_str}': {e}")
                return None

        # Intentar patrón "Apr 01, 2025 (Mar)"
        m = re.search(r'([A-Za-z]+\s+\d{1,2},\s+\d{4})', date_str)
        if m:
            candidate = m.group(1)
            try:
                parsed = pd.to_datetime(candidate, errors='coerce')
                if pd.notnull(parsed):
                    return parsed
            except Exception as e:
                self.logger.warning(f"Error al parsear patrón en '{date_str}': {e}")

        if preferred_dayfirst is not None:
            try:
                return pd.to_datetime(date_str, dayfirst=preferred_dayfirst, errors='coerce')
            except Exception as e:
                self.logger.warning(f"Error con dayfirst={preferred_dayfirst} en '{date_str}': {e}")

        try:
            return pd.to_datetime(date_str, dayfirst=True, errors='coerce')
        except Exception as e:
            self.logger.warning(f"Error en improved_robust_parse_date para '{date_str}': {e}")
            return None

    def process_file(self, config_row):
        """
        Procesa un archivo individual de FRED.
        
        - Busca el archivo usando extensiones: .csv, .xlsx, .xls.
        - Usa la columna de fecha "observation_date" (o "DATE").
        - Analiza hasta 20 registros para determinar el formato de fecha.
        - Convierte la columna de fecha y la columna target a numérico.
        - Renombra la columna de datos con el patrón: {TARGET}_{variable}_{Tipo_Macro}.
        - Devuelve un DataFrame con columnas ['fecha', nuevo_nombre].
        """
        variable = config_row['Variable']
        macro_type = config_row['Tipo Macro']
        target_col = config_row['TARGET']

        # Lista de extensiones a buscar
        extensions = ['.csv', '.xlsx', '.xls']
        ruta = None
        for ext in extensions:
            ruta_candidate = os.path.join(self.data_root, macro_type, f"{variable}{ext}")
            if os.path.exists(ruta_candidate):
                ruta = ruta_candidate
                break
        if ruta is None:
            for ext in extensions:
                for root, dirs, files in os.walk(self.data_root):
                    if f"{variable}{ext}" in files:
                        ruta = os.path.join(root, f"{variable}{ext}")
                        break
                if ruta is not None:
                    break
        if ruta is None:
            self.logger.error(f"Archivo no encontrado: {variable}* (se probaron extensiones: {', '.join(extensions)})")
            return variable, None

        self.logger.info(f"\nProcesando: {variable} ({macro_type})")
        self.logger.info(f"- Archivo: {os.path.basename(ruta)}")
        self.logger.info(f"- Columna TARGET: {target_col}")
        self.logger.info(f"- Ruta encontrada: {ruta}")

        try:
            if ruta.endswith('.csv'):
                df = pd.read_csv(ruta)
            elif ruta.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(ruta)
            else:
                self.logger.error(f"Extensión no soportada para {ruta}")
                return variable, None
        except Exception as e:
            self.logger.error(f"Error al leer {ruta}: {e}")
            return variable, None

        self.logger.info(f"- Filas encontradas: {len(df)}")
        # Determinar la columna de fecha: preferir "observation_date", sino "DATE"
        if 'observation_date' in df.columns:
            date_col = 'observation_date'
        elif 'DATE' in df.columns:
            date_col = 'DATE'
        else:
            self.logger.error(f"No se encontró columna de fecha ('observation_date' o 'DATE') en {ruta}")
            return variable, None

        # Detectar el formato de fecha a partir de 20 registros
        fmt = self.detect_date_format(df[date_col], n=20, iso_threshold=0.6)
        use_iso = (fmt == "ISO")
        self.logger.info(f"Formato detectado para {ruta}: {fmt}")

        # Si el formato no es ISO, determinar la preferencia de dayfirst usando la monotonicidad
        if not use_iso:
            sample = df[date_col].dropna().head(20)
            parsed_true = pd.to_datetime(sample, dayfirst=True, errors='coerce')
            parsed_false = pd.to_datetime(sample, dayfirst=False, errors='coerce')
            score_true = self.monotonic_score(parsed_true)
            score_false = self.monotonic_score(parsed_false)
            preferred = score_true >= score_false
            self.date_cache[ruta] = preferred
            self.logger.info(f"Preferencia de dayfirst para {ruta}: {preferred} (score True: {score_true:.2f}, False: {score_false:.2f})")
        else:
            preferred = None

        # Convertir la columna de fecha usando improved_robust_parse_date
        df['fecha'] = df[date_col].apply(lambda x: self.improved_robust_parse_date(x, preferred_dayfirst=preferred, use_iso=use_iso))
        df = df.dropna(subset=['fecha'])
        df = df.sort_values('fecha')
        self.logger.info(f"Primeras fechas convertidas: {df['fecha'].head(5).tolist()}")

        # Verificar la columna target usando búsqueda insensible a mayúsculas
        if target_col not in df.columns:
            for col in df.columns:
                if col.strip().lower() == target_col.strip().lower():
                    target_col = col
                    self.logger.warning(f"No se encontró '{config_row['TARGET']}', se usará '{target_col}'")
                    break
        if target_col not in df.columns:
            self.logger.error(f"No se encontró columna TARGET ni alternativa en {ruta}")
            return variable, None

        df['valor'] = pd.to_numeric(df[target_col], errors='coerce')
        df = df.dropna(subset=['valor'])
        if df.empty:
            self.logger.error(f"No se encontraron valores válidos para '{target_col}' en {ruta}")
            return variable, None

        current_min = df['fecha'].min()
        current_max = df['fecha'].max()
        if self.global_min_date is None or current_min < self.global_min_date:
            self.global_min_date = current_min
        if self.global_max_date is None or current_max > self.global_max_date:
            self.global_max_date = current_max

        # Renombrar la columna de datos usando el patrón
        nuevo_nombre = f"{target_col}_{variable}_{macro_type}"
        df.rename(columns={'valor': nuevo_nombre}, inplace=True)
        self.stats[variable] = {
            'macro_type': macro_type,
            'target_column': target_col,
            'total_rows': len(df),
            'valid_values': len(df),
            'coverage': 100.0,
            'date_min': current_min,
            'date_max': current_max,
            'nuevo_nombre': nuevo_nombre
        }
        self.logger.info(f"- Valores no nulos en TARGET: {len(df)}")
        self.logger.info(f"- Periodo: {current_min.strftime('%Y-%m-%d')} a {current_max.strftime('%Y-%m-%d')}")
        self.logger.info(f"- Cobertura: 100.00%")
        return variable, df[['fecha', nuevo_nombre]].copy()

    def generate_daily_index(self):
        """
        Genera un índice diario desde la fecha global mínima hasta la máxima.
        """
        if self.global_min_date is None or self.global_max_date is None:
            self.logger.error("No se pudieron determinar las fechas globales")
            return None
        self.daily_index = pd.DataFrame({
            'fecha': pd.date_range(start=self.global_min_date, end=self.global_max_date, freq='D')
        })
        self.logger.info(f"Índice diario generado: {len(self.daily_index)} días desde {self.global_min_date.strftime('%Y-%m-%d')} hasta {self.global_max_date.strftime('%Y-%m-%d')}")
        return self.daily_index

    def combine_data(self):
        """
        Convierte cada serie (diaria o de menor frecuencia) a datos diarios usando merge_asof.
        Cada día se asocia al valor reportado más reciente (forward fill).
        """
        if self.daily_index is None:
            self.logger.error("El índice diario no ha sido generado")
            return None

        combined = self.daily_index.copy()
        for variable, df in self.processed_data.items():
            if df is None or df.empty:
                self.logger.warning(f"Omitiendo {variable} por falta de datos")
                continue
            df = df.sort_values('fecha')
            df_daily = pd.merge_asof(combined, df, on='fecha', direction='backward')
            col_name = self.stats[variable]['nuevo_nombre']
            df_daily[col_name] = df_daily[col_name].ffill()
            combined = combined.merge(df_daily[['fecha', col_name]], on='fecha', how='left')
        self.final_df = combined
        self.logger.info(f"DataFrame final combinado: {len(self.final_df)} filas, {len(self.final_df.columns)} columnas")
        return self.final_df

    def analyze_coverage(self):
        """
        Genera un resumen de cobertura y estadísticas para cada variable.
        """
        total_days = len(self.daily_index)
        self.logger.info("\nResumen de Cobertura:")
        for variable, stats in self.stats.items():
            self.logger.info(f"- {variable}: {stats['coverage']:.2f}% desde {stats['date_min'].strftime('%Y-%m-%d')} a {stats['date_max'].strftime('%Y-%m-%d')}")

    def save_results(self, output_file='datos_economicos_procesados.xlsx'):
        """
        Guarda el DataFrame final combinado y las estadísticas en un archivo Excel.
        """
        if self.final_df is None:
            self.logger.error("No hay datos combinados para guardar")
            return False
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                self.final_df.to_excel(writer, sheet_name='Datos Diarios', index=False)
                df_stats = pd.DataFrame(self.stats).T
                df_stats.to_excel(writer, sheet_name='Estadisticas')
                meta = {
                    'Proceso': ['FredDataProcessor'],
                    'Fecha de proceso': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Total indicadores': [len(self.stats)],
                    'Periodo': [f"{self.global_min_date.strftime('%Y-%m-%d')} a {self.global_max_date.strftime('%Y-%m-%d')}"],
                    'Total días': [len(self.daily_index)]
                }
                pd.DataFrame(meta).to_excel(writer, sheet_name='Metadatos', index=False)
            self.logger.info(f"Archivo guardado exitosamente: {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar resultados: {e}")
            return False

    def run(self, output_file='datos_economicos_procesados.xlsx'):
        """
        Ejecuta el proceso completo:
          1. Lee la configuración.
          2. Procesa cada archivo de FRED (buscando las extensiones adecuadas).
          3. Genera el índice diario global.
          4. Convierte cada serie a datos diarios y las combina.
          5. Analiza la cobertura.
          6. Guarda los resultados.
        """
        start_time = time.time()
        if self.read_config() is None:
            return False

        for _, config_row in self.config_data.iterrows():
            var, df_processed = self.process_file(config_row)
            self.processed_data[var] = df_processed

        if len([df for df in self.processed_data.values() if df is not None]) == 0:
            self.logger.error("No se procesó ningún archivo correctamente")
            return False

        self.generate_daily_index()
        self.combine_data()
        self.analyze_coverage()
        result = self.save_results(output_file)
        end_time = time.time()
        self.logger.info("\nResumen de Ejecución:")
        self.logger.info(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")
        self.logger.info(f"Archivos procesados: {len(self.config_data)}")
        self.logger.info(f"Archivo de salida: {output_file}")
        self.logger.info(f"Estado: {'COMPLETADO' if result else 'ERROR'}")
        return result

# Función principal para ejecutar el proceso
def run_fred_data_processor(
    config_file=os.path.join(PROJECT_ROOT, 'pipelines/Data Engineering.xlsx'),
    output_file=os.path.join(PROJECT_ROOT, 'data/0_raw/datos_economicos_procesados_Fred.xlsx'),
    data_root=os.path.join(PROJECT_ROOT, 'data/0_raw'),
    log_file=os.path.join(PROJECT_ROOT, 'logs/freddataprocessor.log')
):
    processor = FredDataProcessor(config_file, data_root, log_file)
    return processor.run(output_file)

# Ejemplo de uso
if __name__ == "__main__":
    success = run_fred_data_processor()
    print(f"Proceso {'completado exitosamente' if success else 'finalizado con errores'}")

# ## Other

import pandas as pd
import numpy as np
import os
import re
import time
import logging
from datetime import datetime, timedelta
import importlib.util
import sys
import warnings
warnings.filterwarnings('ignore')

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
            '%d.%m.%Y', '%d/%m/%Y', '%d-%m-%Y',
            '%m/%d/%Y', '%Y-%m-%d',
            '%Y%m%d', '%d%m%Y'
        ]
        for fmt in formatos:
            try:
                return pd.to_datetime(fecha_str, format=fmt)
            except Exception:
                continue

        # Intentar detectar patrones como "Apr 01, 2025" o meses en español
        try:
            if re.search(r'([A-Za-z]+\s+\d+,\s+\d{4})', fecha_str):
                match = re.search(r'([A-Za-z]+\s+\d+,\s+\d{4})', fecha_str)
                return pd.to_datetime(match.group(1))
            # Reemplazar meses en español por inglés
            meses_es = {
                'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr',
                'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug',
                'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
            }
            texto_procesado = fecha_str.lower()
            for mes_es, mes_en in meses_es.items():
                if mes_es in texto_procesado:
                    texto_procesado = texto_procesado.replace(mes_es, mes_en)
            return pd.to_datetime(texto_procesado)
        except Exception:
            pass

    # Intento final con pandas por defecto
    try:
        return pd.to_datetime(fecha_str)
    except Exception:
        return None

class OtherDataProcessor:
    """
    Clase para procesar datos de la fuente "Other" usando scripts personalizados.
    """
    def __init__(self, config_file, data_root='data/raw', log_file='otherdataprocessor.log'):
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
            'Put_Call_Ratio_SPY': os.path.join('consumer_confidence', 'Put_Call_Ratio_SPY.csv'),
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
        Encuentra la ruta completa del archivo basado en la variable y tipo_macro
        """
        # Primero verificar si tenemos una ruta conocida para esta variable
        if variable in self.data_paths:
            ruta_conocida = os.path.join(self.data_root, self.data_paths[variable])
            if os.path.exists(ruta_conocida):
                return ruta_conocida
                
        # Construir ruta basada en la estructura de directorios
        ruta_base = os.path.join(self.data_root, tipo_macro)
        
        # Verificar diferentes extensiones
        for ext in ['.csv', '.xlsx', '.xls']:
            nombre_archivo = f"{variable}{ext}"
            ruta_completa = os.path.join(ruta_base, nombre_archivo)
            
            if os.path.exists(ruta_completa):
                return ruta_completa
        
        # Si no se encuentra, intentar buscar en todos los subdirectorios
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
            
            # Cargar el script como un módulo
            spec = importlib.util.spec_from_file_location("custom_script", script_path)
            module = importlib.util.module_from_spec(spec)
            
            # Guardar el directorio actual
            current_dir = os.getcwd()
            script_dir = os.path.dirname(os.path.abspath(script_path))
            
            # Cambiar al directorio del script para evitar problemas de rutas relativas
            os.chdir(script_dir)
            
            # Ejecutar el script
            sys.argv = ['']  # Reset argv para evitar conflictos
            try:
                spec.loader.exec_module(module)
                resultado = True
            except Exception as e:
                self.logger.error(f"Error al ejecutar el script {script_path}: {str(e)}")
                resultado = False
            
            # Restaurar el directorio original
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
            
            if fecha_min > fecha_2014:
                self.logger.warning(f"- ATENCIÓN: Los datos comienzan en {fecha_min.strftime('%Y-%m-%d')}, después de 2014-01-01")
            
            # Manejo específico para la configuración business_confidence + macro_business_confidence_fe_1month
            if tipo_macro == 'business_confidence' and target_col == 'macro_business_confidence_fe_1month':
                # Usar las columnas específicas como están en el archivo original
                if 'GACDISA' in df.columns:
                    nuevo_df[f'{target_col}_GACDISA_{variable}_{tipo_macro}'] = df['GACDISA']
                    self.logger.info(f"- Columna GACDISA encontrada y procesada")
                if 'AWCDISA' in df.columns:
                    nuevo_df[f'{target_col}_AWCDISA_{variable}_{tipo_macro}'] = df['AWCDISA']
                    self.logger.info(f"- Columna AWCDISA encontrada y procesada")
                
                self.logger.info(f"Procesamiento específico para {variable} con TARGET={target_col}")
            else:
                # Procesamiento estándar como teníamos antes
                # Determinar qué columnas usar basado en las disponibles
                columnas_indicador = ['GACDISA', 'AWCDISA', 'headline', 'main_index']
                
                prefix = target_col if pd.notna(target_col) else 'ESI'
                
                columnas_encontradas = []
                for col in columnas_indicador:
                    if col in df.columns:
                        nuevo_df[f'{prefix}_{col}_{variable}_{tipo_macro}'] = df[col]
                        columnas_encontradas.append(col)
                
                if columnas_encontradas:
                    self.logger.info(f"- Columnas encontradas y procesadas: {', '.join(columnas_encontradas)}")
                
                # Si no se encontraron columnas específicas, usar todas las numéricas excepto la fecha
                if len(nuevo_df.columns) == 1:  # Solo tiene la columna fecha
                    columnas_usadas = []
                    for col in df.columns:
                        if col != fecha_col and pd.api.types.is_numeric_dtype(df[col]):
                            nuevo_df[f'{prefix}_{col}_{variable}_{tipo_macro}'] = df[col]
                            columnas_usadas.append(col)
                    
                    if columnas_usadas:
                        self.logger.info(f"- Se usaron columnas numéricas genéricas: {', '.join(columnas_usadas)}")
            
            # Ordenar por fecha
            nuevo_df = nuevo_df.sort_values('fecha')
            
            # Mostrar información de cada columna de datos
            for col in nuevo_df.columns:
                if col != 'fecha':
                    col_no_nulos = nuevo_df[col].notna().sum()
                    pct_no_nulos = (col_no_nulos / len(nuevo_df)) * 100
                    
                    fecha_min_col = nuevo_df.loc[nuevo_df[col].notna(), 'fecha'].min()
                    fecha_max_col = nuevo_df.loc[nuevo_df[col].notna(), 'fecha'].max()
                    
                    self.logger.info(f"- Columna {col}: {col_no_nulos} valores no nulos ({pct_no_nulos:.2f}%), "
                                    f"rango: {fecha_min_col.strftime('%Y-%m-%d')} a {fecha_max_col.strftime('%Y-%m-%d')}")
            
            # Calcular fechas mínima y máxima
            fecha_min = nuevo_df['fecha'].min()
            fecha_max = nuevo_df['fecha'].max()
            
            # Registrar estadísticas
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
        
        # Construir ruta directa basada en la estructura conocida
        ruta_directa = os.path.join(self.data_root, 'consumer_confidence', f"{variable}.xls")
        
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
            # Intentar leer el archivo Excel con diferentes engines
            df = None
            engines = ['openpyxl', 'xlrd']
            
            for engine in engines:
                try:
                    self.logger.info(f"Intentando leer Excel con engine: {engine}")
                    df = pd.read_excel(input_file, engine=engine)
                    break
                except Exception as e:
                    self.logger.warning(f"Error al leer con engine {engine}: {str(e)}")
            
            # Si todos los engines fallan, intentar leer como CSV
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
            
            # Identificar la columna de fecha
            columna_fecha = None
            for col in df.columns:
                if any(term in col.lower() for term in ['date', 'week', 'period']):
                    columna_fecha = col
                    self.logger.info(f"- Usando columna '{col}' como fecha")
                    break
            
            if columna_fecha is None:
                # Si no se encuentra una columna específica, usar la primera columna como fecha
                columna_fecha = df.columns[0]
                self.logger.info(f"- Usando primera columna '{columna_fecha}' como fecha (fallback)")
            
            # Verificar que las columnas necesarias existen
            columnas_objetivo = ['Bearish', 'Bull-Bear Spread', 'Bullish']
            columnas_encontradas = [col for col in columnas_objetivo if col in df.columns]
            columnas_faltantes = [col for col in columnas_objetivo if col not in df.columns]
            
            if columnas_encontradas:
                self.logger.info(f"- Columnas encontradas: {', '.join(columnas_encontradas)}")
            if columnas_faltantes:
                self.logger.warning(f"- Columnas no encontradas: {', '.join(columnas_faltantes)}")
            
            # Crear el resultado
            resultado = pd.DataFrame()
            resultado['fecha'] = pd.to_datetime(df[columna_fecha])
            
            # Verificar rango de fechas
            fecha_min_total = resultado['fecha'].min()
            fecha_max_total = resultado['fecha'].max()
            self.logger.info(f"- Rango de fechas total: {fecha_min_total.strftime('%Y-%m-%d')} a {fecha_max_total.strftime('%Y-%m-%d')}")
            
            # Verificar si tiene datos desde 2014
            fecha_2014 = pd.to_datetime('2014-01-01')
            tiene_desde_2014 = fecha_min_total <= fecha_2014
            self.logger.info(f"- ¿Tiene datos desde 2014 o antes?: {'Sí' if tiene_desde_2014 else 'No'}")
            
            if fecha_min_total > fecha_2014:
                self.logger.warning(f"- ATENCIÓN: Los datos comienzan en {fecha_min_total.strftime('%Y-%m-%d')}, después de 2014-01-01")
            
            # Prefijo para renombrar las columnas
            prefix = target_col if pd.notna(target_col) else 'AAII'
            
            # Añadir columnas de datos con los prefijos adecuados
            for columna in columnas_encontradas:
                nombre_col = f'{prefix}_{columna}_{variable}_{tipo_macro}'
                resultado[nombre_col] = df[columna]
                
                # Verificar rango de fechas para esta columna
                fechas_validas = resultado[resultado[nombre_col].notna()]
                if len(fechas_validas) > 0:
                    fecha_min_col = fechas_validas['fecha'].min()
                    fecha_max_col = fechas_validas['fecha'].max()
                    self.logger.info(f"- Columna {nombre_col}: Rango de fechas {fecha_min_col.strftime('%Y-%m-%d')} a {fecha_max_col.strftime('%Y-%m-%d')}")
                    self.logger.info(f"- Valores disponibles: {len(fechas_validas)}/{len(resultado)} ({len(fechas_validas)/len(resultado)*100:.2f}%)")
            
            # Si no se encontraron columnas específicas, usar todas las numéricas
            if not columnas_encontradas:
                self.logger.warning(f"- No se encontraron columnas específicas. Usando columnas numéricas disponibles.")
                columnas_usadas = []
                for col in df.columns:
                    if col != columna_fecha and pd.api.types.is_numeric_dtype(df[col]):
                        nombre_col = f'{prefix}_{col}_{variable}_{tipo_macro}'
                        resultado[nombre_col] = df[col]
                        columnas_usadas.append(col)
                        
                        # Verificar rango de fechas para esta columna
                        fechas_validas = resultado[resultado[nombre_col].notna()]
                        if len(fechas_validas) > 0:
                            fecha_min_col = fechas_validas['fecha'].min()
                            fecha_max_col = fechas_validas['fecha'].max()
                            self.logger.info(f"- Columna {nombre_col}: Rango de fechas {fecha_min_col.strftime('%Y-%m-%d')} a {fecha_max_col.strftime('%Y-%m-%d')}")
                            self.logger.info(f"- Valores disponibles: {len(fechas_validas)}/{len(resultado)} ({len(fechas_validas)/len(resultado)*100:.2f}%)")
                
                if columnas_usadas:
                    self.logger.info(f"- Se utilizaron columnas numéricas alternativas: {', '.join(columnas_usadas)}")
            
            # Ordenar por fecha
            resultado = resultado.sort_values('fecha')
            
            # Calcular fechas mínima y máxima
            fecha_min = resultado['fecha'].min()
            fecha_max = resultado['fecha'].max()
            
            # Registrar estadísticas
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
        Procesa los datos de Put Call Ratio manualmente
        """
        self.logger.info(f"Procesando {variable} manualmente")
        
        # Construir ruta directa basada en la estructura conocida
        ruta_directa = os.path.join(self.data_root, 'consumer_confidence', f"{variable}.csv")
        
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
            self.logger.info(f"- Columnas disponibles: {', '.join(df.columns.tolist())}")
            
            # Crear un nuevo DataFrame para el resultado
            result_df = pd.DataFrame()
            
            # Buscar columna de fecha
            fecha_col = None
            for col in df.columns:
                if any(term in col.lower() for term in ['date', 'time', 'periodo']):
                    fecha_col = col
                    self.logger.info(f"- Usando columna '{col}' como fecha")
                    break
            
            if fecha_col is None and len(df.columns) > 0:
                fecha_col = df.columns[0]  # Usar primera columna como fallback
                self.logger.info(f"- Usando primera columna '{fecha_col}' como fecha (fallback)")
            
            if fecha_col is None:
                self.logger.error(f"No se pudo identificar columna de fecha en {input_file}")
                return variable, None
                
            # Convertir fechas y añadir al resultado
            result_df['fecha'] = pd.to_datetime(df[fecha_col])
            
            # Verificar rango de fechas
            fecha_min_total = result_df['fecha'].min()
            fecha_max_total = result_df['fecha'].max()
            self.logger.info(f"- Rango de fechas total: {fecha_min_total.strftime('%Y-%m-%d')} a {fecha_max_total.strftime('%Y-%m-%d')}")
            
            # Verificar si tiene datos desde 2014
            fecha_2014 = pd.to_datetime('2014-01-01')
            tiene_desde_2014 = fecha_min_total <= fecha_2014
            self.logger.info(f"- ¿Tiene datos desde 2014 o antes?: {'Sí' if tiene_desde_2014 else 'No'}")
            
            if fecha_min_total > fecha_2014:
                self.logger.warning(f"- ATENCIÓN: Los datos comienzan en {fecha_min_total.strftime('%Y-%m-%d')}, después de 2014-01-01")
            
            # Generar un indicador de confianza simplificado basado en el ratio put/call
            col_name = f'consumer_confidence_PutCall_{variable}_{tipo_macro}'
            
            # Si ya existe la columna de confianza (archivo pre-procesado), usarla directamente
            if 'consumer_confidence_PutCall' in df.columns:
                self.logger.info(f"- Usando columna 'consumer_confidence_PutCall' existente")
                result_df[col_name] = df['consumer_confidence_PutCall']
                
                # Verificar rango de fechas para esta columna
                fechas_validas = result_df[result_df[col_name].notna()]
                if len(fechas_validas) > 0:
                    fecha_min_col = fechas_validas['fecha'].min()
                    fecha_max_col = fechas_validas['fecha'].max()
                    self.logger.info(f"- Columna {col_name}: Rango de fechas {fecha_min_col.strftime('%Y-%m-%d')} a {fecha_max_col.strftime('%Y-%m-%d')}")
                    self.logger.info(f"- Valores disponibles: {len(fechas_validas)}/{len(result_df)} ({len(fechas_validas)/len(result_df)*100:.2f}%)")
            else:
                # Intentar calcular un indicador basado en volúmenes si están disponibles
                if 'call_volume' in df.columns and 'put_volume' in df.columns:
                    self.logger.info(f"- Calculando ratio basado en put_volume y call_volume")
                    # Calcular ratio put/call
                    df['put_call_ratio'] = df['put_volume'] / df['call_volume'].replace(0, np.nan)
                    
                    # Usar una transformación simple para convertir el ratio en un indicador de confianza
                    # (valores más bajos del ratio indican más confianza)
                    if df['put_call_ratio'].notna().sum() > 0:
                        max_ratio = df['put_call_ratio'].max()
                        result_df[col_name] = 100 * (1 - df['put_call_ratio'] / max_ratio)
                        
                        # Verificar rango de fechas para esta columna
                        fechas_validas = result_df[result_df[col_name].notna()]
                        if len(fechas_validas) > 0:
                            fecha_min_col = fechas_validas['fecha'].min()
                            fecha_max_col = fechas_validas['fecha'].max()
                            self.logger.info(f"- Columna {col_name}: Rango de fechas {fecha_min_col.strftime('%Y-%m-%d')} a {fecha_max_col.strftime('%Y-%m-%d')}")
                            self.logger.info(f"- Valores disponibles: {len(fechas_validas)}/{len(result_df)} ({len(fechas_validas)/len(result_df)*100:.2f}%)")
                    else:
                        self.logger.warning(f"- No se pudieron calcular ratios put/call (posibles divisiones por cero)")
                else:
                    self.logger.warning(f"- No se encontraron columnas call_volume y put_volume. Buscando otras columnas numéricas.")
                    # Si no hay columnas específicas, usar todas las numéricas excepto la fecha
                    columnas_usadas = []
                    for col in df.columns:
                        if col != fecha_col and pd.api.types.is_numeric_dtype(df[col]):
                            prefix = target_col if pd.notna(target_col) else 'PutCall'
                            nuevo_nombre = f'{prefix}_{col}_{variable}_{tipo_macro}'
                            result_df[nuevo_nombre] = df[col]
                            columnas_usadas.append(col)
                            
                            # Verificar rango de fechas para esta columna
                            fechas_validas = result_df[result_df[nuevo_nombre].notna()]
                            if len(fechas_validas) > 0:
                                fecha_min_col = fechas_validas['fecha'].min()
                                fecha_max_col = fechas_validas['fecha'].max()
                                self.logger.info(f"- Columna {nuevo_nombre}: Rango de fechas {fecha_min_col.strftime('%Y-%m-%d')} a {fecha_max_col.strftime('%Y-%m-%d')}")
                                self.logger.info(f"- Valores disponibles: {len(fechas_validas)}/{len(result_df)} ({len(fechas_validas)/len(result_df)*100:.2f}%)")
                    
                    if columnas_usadas:
                        self.logger.info(f"- Se utilizaron columnas numéricas alternativas: {', '.join(columnas_usadas)}")
                    else:
                        self.logger.warning(f"- No se encontraron columnas numéricas alternativas")
            
            # Ordenar por fecha
            result_df = result_df.sort_values('fecha')
            
            # Calcular fechas mínima y máxima
            fecha_min = result_df['fecha'].min()
            fecha_max = result_df['fecha'].max()
            
            # Registrar estadísticas
            self.estadisticas[variable] = {
                'tipo_macro': tipo_macro,
                'columna_target': target_col or 'PutCall',
                'total_filas': len(result_df),
                'valores_validos': len(result_df),
                'fecha_min': fecha_min,
                'fecha_max': fecha_max
            }
            
            self.logger.info(f"- {variable}: {len(result_df)} filas procesadas manualmente, periodo: {fecha_min.strftime('%Y-%m-%d')} a {fecha_max.strftime('%Y-%m-%d')}")
            
            return variable, result_df
            
        except Exception as e:
            self.logger.error(f"Error al procesar {variable} manualmente: {str(e)}")
            return variable, None
            
    def procesar_archivo_generico(self, variable, tipo_macro, target_col):
        """
        Procesamiento genérico para cualquier archivo de datos
        """
        self.logger.info(f"Procesando {variable} de forma genérica")
        
        # Buscar el archivo en su ubicación esperada o en el árbol de directorios
        input_file = self.encontrar_ruta_archivo(variable, tipo_macro)
        if input_file is None:
            self.logger.error(f"Archivo de entrada no encontrado para {variable}")
            return variable, None
            
        self.logger.info(f"- Archivo de entrada: {input_file}")
        
        try:
            # Cargar el archivo según su extensión
            if input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
            elif input_file.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(input_file)
            else:
                self.logger.error(f"Formato de archivo no soportado: {input_file}")
                return variable, None
            
            # Buscar columna de fecha
            fecha_col = None
            for col in df.columns:
                if any(term in col.lower() for term in ['date', 'time', 'fecha', 'week', 'period']):
                    fecha_col = col
                    break
            
            if fecha_col is None and len(df.columns) > 0:
                # Intentar con la primera columna como fecha
                try:
                    pd.to_datetime(df.iloc[:, 0])
                    fecha_col = df.columns[0]
                except:
                    self.logger.error(f"No se pudo identificar columna de fecha en {input_file}")
                    return variable, None
            
            # Crear un nuevo DataFrame con las columnas estandarizadas
            result_df = pd.DataFrame()
            result_df['fecha'] = pd.to_datetime(df[fecha_col])
            result_df = result_df.dropna(subset=['fecha'])
            
            # Determinar el prefijo para las columnas de datos
            prefix = target_col if pd.notna(target_col) else variable.split('_')[0]
            
            # Añadir todas las columnas numéricas con prefijo estandarizado
            for col in df.columns:
                if col != fecha_col and pd.api.types.is_numeric_dtype(df[col]):
                    result_df[f'{prefix}_{col}_{variable}_{tipo_macro}'] = df[col]
            
            # Ordenar por fecha
            result_df = result_df.sort_values('fecha')
            
            # Calcular fechas mínima y máxima
            fecha_min = result_df['fecha'].min()
            fecha_max = result_df['fecha'].max()
            
            # Registrar estadísticas
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
        
        # Construir ruta directa basada en la estructura conocida
        ruta_directa = os.path.join(self.data_root, 'leading_economic_index', f"{variable}.csv")
        
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
            # Cargar el dataset
            df = pd.read_csv(input_file)
            self.logger.info(f"- Archivo cargado: {len(df)} filas, {len(df.columns)} columnas")
            self.logger.info(f"- Columnas disponibles: {', '.join(df.columns.tolist())}")
            
            # Convertir la columna de fecha al formato correcto
            if 'Friday_of_Week' in df.columns:
                df['fecha'] = pd.to_datetime(df['Friday_of_Week'], format='%m/%d/%Y')
                self.logger.info(f"- Usando columna 'Friday_of_Week' como fecha")
            else:
                # Buscar columna de fecha
                for col in df.columns:
                    if any(term in col.lower() for term in ['date', 'week', 'period']):
                        df['fecha'] = pd.to_datetime(df[col])
                        self.logger.info(f"- Usando columna '{col}' como fecha")
                        break
            
            # Verificar que tenemos una columna de fecha
            if 'fecha' not in df.columns:
                self.logger.error(f"No se pudo identificar columna de fecha en {input_file}")
                return variable, None
            
            # Verificar rango de fechas total
            fecha_min_total = df['fecha'].min()
            fecha_max_total = df['fecha'].max()
            self.logger.info(f"- Rango de fechas total: {fecha_min_total.strftime('%Y-%m-%d')} a {fecha_max_total.strftime('%Y-%m-%d')}")
            
            # Verificar si tiene datos desde 2014
            fecha_2014 = pd.to_datetime('2014-01-01')
            tiene_desde_2014 = fecha_min_total <= fecha_2014
            self.logger.info(f"- ¿Tiene datos desde 2014 o antes?: {'Sí' if tiene_desde_2014 else 'No'}")
            
            if fecha_min_total > fecha_2014:
                self.logger.warning(f"- ATENCIÓN: Los datos comienzan en {fecha_min_total.strftime('%Y-%m-%d')}, después de 2014-01-01")
            
            # Filtrar desde 2014 si corresponde
            df_filtrado = df[df['fecha'] >= fecha_2014]
            self.logger.info(f"- Filtrando desde 2014-01-01: {len(df_filtrado)}/{len(df)} filas ({(len(df_filtrado)/len(df)*100):.2f}%)")
            
            # Crear resultado
            resultado = pd.DataFrame()
            resultado['fecha'] = df_filtrado['fecha']
            
            # Verificar y añadir columnas NFCI y ANFCI 
            columnas_procesadas = []
            if 'NFCI' in df_filtrado.columns:
                resultado[f'NFCI_{variable}_{tipo_macro}'] = df_filtrado['NFCI']
                columnas_procesadas.append('NFCI')
                
                # Verificar rango de fechas para NFCI
                fecha_min_nfci = df_filtrado.loc[df_filtrado['NFCI'].notna(), 'fecha'].min()
                fecha_max_nfci = df_filtrado.loc[df_filtrado['NFCI'].notna(), 'fecha'].max()
                self.logger.info(f"- Columna NFCI: Rango de fechas {fecha_min_nfci.strftime('%Y-%m-%d')} a {fecha_max_nfci.strftime('%Y-%m-%d')}")
            
            if 'ANFCI' in df_filtrado.columns:
                resultado[f'ANFCI_{variable}_{tipo_macro}'] = df_filtrado['ANFCI']
                columnas_procesadas.append('ANFCI')
                
                # Verificar rango de fechas para ANFCI
                fecha_min_anfci = df_filtrado.loc[df_filtrado['ANFCI'].notna(), 'fecha'].min()
                fecha_max_anfci = df_filtrado.loc[df_filtrado['ANFCI'].notna(), 'fecha'].max()
                self.logger.info(f"- Columna ANFCI: Rango de fechas {fecha_min_anfci.strftime('%Y-%m-%d')} a {fecha_max_anfci.strftime('%Y-%m-%d')}")
            
            # Si no se encontraron las columnas específicas, usar todas las numéricas excepto la fecha
            if len(columnas_procesadas) == 0:
                self.logger.warning(f"- No se encontraron columnas NFCI o ANFCI. Utilizando columnas numéricas disponibles")
                
                for col in df_filtrado.columns:
                    if col != 'fecha' and pd.api.types.is_numeric_dtype(df_filtrado[col]):
                        prefix = target_col if pd.notna(target_col) else 'NFCI'
                        nuevo_nombre = f'{prefix}_{col}_{variable}_{tipo_macro}'
                        resultado[nuevo_nombre] = df_filtrado[col]
                        columnas_procesadas.append(col)
                        
                        # Verificar rango de fechas para esta columna
                        fecha_min_col = df_filtrado.loc[df_filtrado[col].notna(), 'fecha'].min()
                        fecha_max_col = df_filtrado.loc[df_filtrado[col].notna(), 'fecha'].max()
                        self.logger.info(f"- Columna {col}: Rango de fechas {fecha_min_col.strftime('%Y-%m-%d')} a {fecha_max_col.strftime('%Y-%m-%d')}")
            
            self.logger.info(f"- Columnas procesadas: {', '.join(columnas_procesadas)}")
            
            # Ordenar por fecha
            resultado = resultado.sort_values('fecha')
            
            # Calcular fechas mínima y máxima
            fecha_min = resultado['fecha'].min()
            fecha_max = resultado['fecha'].max()
            
            # Registrar estadísticas
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
            
            # Comprobar si el archivo existe
            if not os.path.exists(output_file):
                self.logger.error(f"El archivo de salida {output_file} no existe")
                return None
            
            # Cargar el archivo según su extensión
            if output_file.endswith('.csv'):
                df = pd.read_csv(output_file)
            elif output_file.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(output_file)
            else:
                self.logger.error(f"Formato de archivo no soportado: {output_file}")
                return None
            
            # Identificar columna de fecha
            fecha_col = self.fecha_column_map.get(variable, None)
            if fecha_col not in df.columns:
                for col in df.columns:
                    if any(palabra in col.lower() for palabra in ['date', 'fecha', 'time', 'día']):
                        fecha_col = col
                        break
            
            if fecha_col is None:
                self.logger.error(f"No se pudo identificar la columna de fecha en {output_file}")
                return None
            
            # Crear un nuevo DataFrame con las columnas estandarizadas
            result_df = pd.DataFrame()
            
            # Convertir y estandarizar la columna de fecha
            result_df['fecha'] = df[fecha_col].apply(convertir_fecha)
            result_df = result_df.dropna(subset=['fecha'])
            
            # Renombrar las columnas de datos con el patrón estándar
            for col in df.columns:
                if col != fecha_col:
                    new_name = f"{target_col}_{variable}_{col}"
                    result_df[new_name] = df[col]
            
            # Ordenar por fecha
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
        
        # Determinar el procesador específico basado en la variable
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
            
            # Calcular estadísticas básicas
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
            
            # Guardar estadísticas en el diccionario
            variable = next((var for var in self.estadisticas.keys() if var in col), None)
            if variable:
                # Añadir estadísticas para esta columna
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
                        'Columna_TARGET': stats.get('columna_target', ''),
                        'Total_Filas_Original': stats.get('total_filas', 0),
                        'Valores_Validos_Original': stats.get('valores_validos', 0),
                        'Cobertura_Final_%': stats.get('cobertura_final', 0),
                        'Fecha_Min': stats.get('fecha_min', ''),
                        'Fecha_Max': stats.get('fecha_max', '')
                    }
                    # Añadir estadísticas específicas por columna si existen
                    for k, v in stats.items():
                        if any(k.startswith(f"{col}_") for col in ['min', 'max', 'mean', 'median', 'std']):
                            row[k] = v
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
        
        # 6. Generar estadísticas de valores
        self.generar_estadisticas_valores()
        
        # 7. Guardar resultados
        resultado = self.guardar_resultados(output_file)
        
        # 8. Mostrar resumen final
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

import pandas as pd
import numpy as np
import os
import re
import time
import logging
from datetime import datetime, timedelta
import importlib.util
import sys
import warnings
warnings.filterwarnings('ignore')

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
            '%d.%m.%Y', '%d/%m/%Y', '%d-%m-%Y',
            '%m/%d/%Y', '%Y-%m-%d',
            '%Y%m%d', '%d%m%Y'
        ]
        for fmt in formatos:
            try:
                return pd.to_datetime(fecha_str, format=fmt)
            except Exception:
                continue

        try:
            if re.search(r'([A-Za-z]+\s+\d+,\s+\d{4})', fecha_str):
                match = re.search(r'([A-Za-z]+\s+\d+,\s+\d{4})', fecha_str)
                return pd.to_datetime(match.group(1))
            meses_es = {
                'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr',
                'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug',
                'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
            }
            texto_procesado = fecha_str.lower()
            for mes_es, mes_en in meses_es.items():
                if mes_es in texto_procesado:
                    texto_procesado = texto_procesado.replace(mes_es, mes_en)
            return pd.to_datetime(texto_procesado)
        except Exception:
            pass

    try:
        return pd.to_datetime(fecha_str)
    except Exception:
        return None

class OtherDataProcessor:
    """
    Clase para procesar datos de la fuente "Other" usando scripts personalizados.
    """
    def __init__(self, config_file, data_root='data/raw', log_file='otherdataprocessor.log'):
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
            'Put_Call_Ratio_SPY': os.path.join('consumer_confidence', 'Put_Call_Ratio_SPY.csv'),
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
        elif variable == 'Put_Call_Ratio_SPY':
            return self.procesar_put_call_ratio(variable, tipo_macro, target_col) 
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
    # Intentar leer el archivo Excel
    try:
        df = pd.read_excel("MERGEDEXCELS.xlsx")
        columns = df.columns.tolist()
        logging.info("Archivo Excel cargado exitosamente.")
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

if __name__ == "__main__":
    main()

