import os
import re
import time
import pandas as pd
from datetime import datetime, timedelta
from sp500_analysis.shared.logging.logger import configurar_logging


class InvestingProcessor:
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
        self.logger.info("INICIANDO PROCESO: InvestingProcessor")
        self.logger.info(f"Archivo de configuración: {config_file}")
        self.logger.info(f"Directorio raíz de datos: {data_root}")
        self.logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)

    def read_config(self):
        try:
            self.logger.info("Leyendo archivo de configuración...")
            df_config = pd.read_excel(self.config_file)
            self.config_data = df_config[
                (df_config['Fuente'] == 'Investing Data')
                & (df_config['Tipo de Preprocesamiento Según la Fuente'] == 'Copiar y Pegar')
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
        Procesa un archivo individual con manejo robusto de formatos numéricos.
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

        # FUNCIÓN MEJORADA: Convertir la columna target a numérico con manejo de formatos especiales
        def convertir_valor_robusto(val):
            """Maneja múltiples formatos numéricos incluyendo sufijos y símbolos"""
            if pd.isna(val) or val == '':
                return None

            # Convertir a string y limpiar
            val_str = str(val).strip()

            # Manejar sufijos multiplicadores
            multiplicador = 1
            if val_str.endswith('B') or val_str.endswith('b'):  # Billones
                multiplicador = 1e9
                val_str = val_str[:-1].strip()
            elif val_str.endswith('M') or val_str.endswith('m'):  # Millones
                multiplicador = 1e6
                val_str = val_str[:-1].strip()
            elif val_str.endswith('K') or val_str.endswith('k'):  # Miles
                multiplicador = 1e3
                val_str = val_str[:-1].strip()

            # Eliminar porcentaje
            if val_str.endswith('%'):
                val_str = val_str[:-1].strip()
                multiplicador *= 0.01  # Para convertir 5% a 0.05

            # Reemplazar comas por puntos (formato europeo)
            val_str = val_str.replace(',', '.')

            try:
                return float(val_str) * multiplicador
            except ValueError:
                return None

        # Aplicar la función de conversión robusta
        df['valor'] = df[target_col].apply(convertir_valor_robusto)
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
            'nuevo_nombre': nuevo_nombre,
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
        self.daily_index = pd.DataFrame(
            {'fecha': pd.date_range(start=self.global_min_date, end=self.global_max_date, freq='D')}
        )
        self.logger.info(
            f"Índice diario generado: {len(self.daily_index)} días desde {self.global_min_date.strftime('%Y-%m-%d')} hasta {self.global_max_date.strftime('%Y-%m-%d')}"
        )
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
        self.logger.info(
            f"DataFrame final combinado: {len(self.final_df)} filas, {len(self.final_df.columns)} columnas"
        )
        return self.final_df

    def analyze_coverage(self):
        """
        Genera un resumen de cobertura y estadísticas para cada indicador.
        """
        total_days = len(self.daily_index)
        self.logger.info("\nResumen de Cobertura:")
        for variable, stats in self.stats.items():
            self.logger.info(
                f"- {variable}: {stats['coverage']:.2f}% desde {stats['date_min'].strftime('%Y-%m-%d')} a {stats['date_max'].strftime('%Y-%m-%d')}"
            )

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
                    'Proceso': ['InvestingProcessor'],
                    'Fecha de proceso': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Total indicadores': [len(self.stats)],
                    'Periodo': [
                        f"{self.global_min_date.strftime('%Y-%m-%d')} a {self.global_max_date.strftime('%Y-%m-%d')}"
                    ],
                    'Total días': [len(self.daily_index)],
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
