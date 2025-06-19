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


class FredDataProcessor(BaseProcessor):
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

    def __init__(self, config_file: str, data_root: str | None = "data/0_raw", log_file: str = "fred_data.log"):
        super().__init__(data_root=data_root, log_file=log_file)
        self.config_file = config_file
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

    def _validate_input(self, data: Any) -> bool:
        return True

    def _transform(self, data: Any) -> Any:
        self.logger.info("Running FRED Data processor")
        return data

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
        self.logger.info(f"Formato detectado para {ruta}: {fmt}")
        
        use_iso = (fmt == "ISO")
        
        if use_iso:
            # Usar formato ISO directamente
            df['fecha'] = df[date_col].apply(lambda x: self.improved_robust_parse_date(x, use_iso=True))
        else:
            # Evaluar ambas opciones usando la monotonicidad
            sample = df[date_col].dropna().head(100)
            test_true = sample.apply(lambda x: self.improved_robust_parse_date(x, preferred_dayfirst=True))
            test_false = sample.apply(lambda x: self.improved_robust_parse_date(x, preferred_dayfirst=False))
            
            score_true = self.monotonic_score(test_true)
            score_false = self.monotonic_score(test_false)
            
            preferred = score_true >= score_false
            self.logger.info(f"Preferencia de dayfirst para {ruta}: {preferred} (score True: {score_true:.3f}, False: {score_false:.3f})")
            
            # Aplicar la preferencia a toda la serie
            df['fecha'] = df[date_col].apply(lambda x: self.improved_robust_parse_date(x, preferred_dayfirst=preferred))

        df = df.dropna(subset=['fecha'])

        # Verificar la columna target usando búsqueda insensible a mayúsculas
        if target_col not in df.columns:
            found_col = None
            for col in df.columns:
                if col.strip().lower() == target_col.strip().lower():
                    found_col = col
                    break
            if found_col:
                target_col = found_col
                self.logger.warning(f"No se encontró '{config_row['TARGET']}', se usará '{target_col}'")
            else:
                self.logger.error(f"No se encontró columna TARGET ni alternativa en {ruta}")
                return variable, None

        # Convertir la columna target a numérico
        df['valor'] = pd.to_numeric(df[target_col], errors='coerce')
        df = df.dropna(subset=['valor'])
        
        if df.empty:
            self.logger.error(f"No se encontraron valores válidos para '{target_col}' en {ruta}")
            return variable, None

        # Actualizar fechas globales
        current_min = df['fecha'].min()
        current_max = df['fecha'].max()
        if self.global_min_date is None or current_min < self.global_min_date:
            self.global_min_date = current_min
        if self.global_max_date is None or current_max > self.global_max_date:
            self.global_max_date = current_max

        # Renombrar la columna de datos usando el patrón
        nuevo_nombre = f"{target_col}_{variable}_{macro_type}"
        df = df.rename(columns={'valor': nuevo_nombre})

        # Guardar estadísticas
        valores_validos = df[nuevo_nombre].notna().sum()
        self.stats[variable] = {
            'target_col': target_col,
            'macro_type': macro_type,
            'total_rows': len(df),
            'valid_values': valores_validos,
            'coverage': (valores_validos / len(df)) * 100,
            'date_min': current_min,
            'date_max': current_max
        }
        
        self.logger.info(f"- Valores no nulos en TARGET: {valores_validos}")
        self.logger.info(f"- Periodo: {current_min.strftime('%Y-%m-%d')} a {current_max.strftime('%Y-%m-%d')}")

        return variable, df[['fecha', nuevo_nombre]].copy()

    def generate_daily_index(self):
        """
        Genera un índice diario entre las fechas mínima y máxima globales.
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

        self.final_df = self.daily_index.copy()
        
        for variable, df_processed in self.processed_data.items():
            if df_processed is not None and not df_processed.empty:
                # Usar merge_asof para imputar forward fill
                self.final_df = pd.merge_asof(
                    self.final_df.sort_values('fecha'),
                    df_processed.sort_values('fecha'),
                    on='fecha',
                    direction='backward'
                )
        
        self.logger.info(f"Datos combinados: {len(self.final_df)} filas, {len(self.final_df.columns)} columnas")
        return self.final_df

    def analyze_coverage(self):
        """
        Analiza y registra estadísticas de cobertura y estadísticas para cada variable.
        """
        total_days = len(self.daily_index)
        self.logger.info("\nResumen de Cobertura:")
        for variable, stats in self.stats.items():
            self.logger.info(
                f"- {variable}: {stats['coverage']:.2f}% desde {stats['date_min'].strftime('%Y-%m-%d')} a {stats['date_max'].strftime('%Y-%m-%d')}"
            )

    def save_results(self, output_file='datos_economicos_procesados_Fred.xlsx'):
        """
        Guarda el DataFrame final combinado y las estadísticas en un archivo Excel.
        """
        if self.final_df is None:
            self.logger.error("No hay datos combinados para guardar")
            return False
        
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                self.final_df.to_excel(writer, sheet_name='Datos Diarios', index=False)
                
                # Guardar estadísticas
                stats_df = pd.DataFrame.from_dict(self.stats, orient='index')
                stats_df.to_excel(writer, sheet_name='Estadisticas')
                
                # Metadatos
                meta = {
                    'Total archivos': [len(self.stats)],
                    'Fecha de proceso': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Periodo': [f"{self.global_min_date.strftime('%Y-%m-%d')} a {self.global_max_date.strftime('%Y-%m-%d')}"],
                    'Total días': [len(self.daily_index)]
                }
                pd.DataFrame(meta).to_excel(writer, sheet_name='Metadatos', index=False)
                
            self.logger.info(f"Archivo guardado exitosamente: {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar resultados: {e}")
            return False

    def run(self, output_file='datos_economicos_procesados_Fred.xlsx'):
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
