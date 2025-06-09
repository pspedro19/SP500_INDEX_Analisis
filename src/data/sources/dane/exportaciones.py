import os
import re
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from src.core.utils import configure_logging, convertir_valor, detectar_formato_fecha_inteligente, convertir_fecha_adaptativo

class DANEExportacionesProcessor:


        import re
        import time
        import pandas as pd
        from datetime import datetime
        from pathlib import Path

        def __init__(self, data_root='data/0_raw', log_file='dane_exportaciones.log'):
            self.data_root = data_root
            self.logger = configurar_logging(log_file)
            self.global_min_date = None
            self.global_max_date = None
            self.daily_index = None
            self.processed_data = {}
            self.final_df = None
            self.stats = {}
            self.discovered_files = []

            self.logger.info("=" * 80)
            self.logger.info("DANE EXPORTACIONES PROCESSOR")
            self.logger.info(f"Directorio de datos: {data_root}")
            self.logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("=" * 80)
        
        def configurar_logging(log_file):
            import logging
            logging.basicConfig(
                filename=log_file,
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            return logging.getLogger(__name__)
    
        def detect_dane_exportaciones_file(self, file_path):
            try:
                df = pd.read_excel(file_path, sheet_name=0, header=None, nrows=15)
                indicators = 0
                for i in range(5):
                    for j in range(df.shape[1]):
                        cell_value = str(df.iloc[i, j]).upper()
                        if 'DANE' in cell_value:
                            indicators += 3
                            break
                for i in range(10):
                    for j in range(df.shape[1]):
                        cell_value = str(df.iloc[i, j]).lower()
                        if 'exportaciones' in cell_value:
                            indicators += 2
                            break
                for i in range(10, 20):
                    if i < len(df):
                        cell_value = str(df.iloc[i, 0]).lower().strip()
                        if re.match(r'^[a-z]{3}-\d{2}$', cell_value):
                            indicators += 3
                            break
                        if cell_value == 'mes':
                            indicators += 3
                            break
                for i in range(15):
                    for j in range(df.shape[1]):
                        if i < len(df) and j < df.shape[1]:
                            cell_value = str(df.iloc[i, j]).lower()
                            if any(word in cell_value for word in ['café', 'carbón', 'petróleo', 'tradicionales']):
                                indicators += 1
                                break
                return indicators >= 6
            except Exception as e:
                self.logger.warning(f"Error detectando archivo DANE: {e}")
                return False

        def auto_discover_files(self):
            self.logger.info("Buscando archivos de exportaciones DANE...")
            discovered = []
            for root, dirs, files in os.walk(self.data_root):
                for file in files:
                    if file.endswith(('.xlsx', '.xls')) and not file.startswith('~'):
                        file_path = os.path.join(root, file)
                        if self.detect_dane_exportaciones_file(file_path):
                            variable_name = Path(file).stem
                            discovered.append({
                                'Variable': variable_name,
                                'Archivo': file_path,
                                'TARGET': 'Total_Exportaciones_Tradicionales',
                                'Tipo_Macro': 'exports',
                                'Carpeta': os.path.basename(os.path.dirname(file_path))
                            })
            self.discovered_files = discovered
            self.logger.info(f"Encontrados {len(discovered)} archivos DANE Exportaciones:")
            for file_info in discovered:
                self.logger.info(f"   Carpeta: {file_info['Carpeta']}/{file_info['Variable']}")
            return discovered

        def convert_colombian_number(self, value):
            if pd.isna(value):
                return None
            if isinstance(value, (int, float)):
                return float(value)
            value = str(value).strip()
            if value in ['-', '', 'N/A', 'n/a', '0']:
                return None
            try:
                if '.' in value and ',' in value:
                    value = value.replace('.', '').replace(',', '.')
                elif ',' in value:
                    value = value.replace(',', '.')
                elif '.' in value:
                    if value.count('.') == 1 and len(value.split('.')[1]) <= 2:
                        pass
                    else:
                        value = value.replace('.', '')
                return float(value)
            except:
                return None

        def process_file_automatically(self, file_info):
            variable = file_info['Variable']
            target_col = file_info['TARGET']
            macro_type = file_info['Tipo_Macro']
            file_path = file_info['Archivo']

            self.logger.info(f"\nProcesando: {variable}")
            self.logger.info(f"   Archivo: {file_path}")
            self.logger.info(f"   TARGET: {target_col}")

            try:
                df = pd.read_excel(file_path, sheet_name=0, header=None)

                # Buscar la fila donde comienzan los datos
                start_row = None
                for i in range(len(df)):
                    cell_value = str(df.iloc[i, 0]).lower().strip()
                    if cell_value == 'mes':
                        start_row = i + 1
                        self.logger.info(f"   Encontrado header 'MES' en fila {i+1}, datos inician en fila {start_row+1}")
                        break

                if start_row is None:
                    self.logger.error(f"No se encontró inicio de datos en {variable}")
                    return variable, None

                data_section = df.iloc[start_row:].copy()
                data_section = data_section.reset_index(drop=True)

                # Seleccionar solo las columnas relevantes
                if 18 >= data_section.shape[1]:
                    self.logger.error(f"El archivo {variable} no tiene columna 18 para total exportaciones.")
                    return variable, None

                df_filtered = data_section[[0, 18]].copy()
                df_filtered.columns = ['fecha', 'total_exportaciones']

                # Limpiar: eliminar totales, notas, fuentes, NaN
                df_filtered = df_filtered[df_filtered['fecha'].notna()]
                df_filtered = df_filtered[~df_filtered['fecha'].astype(str).str.lower().str.contains('total|nan|fuente|nota|actualizado')]
                df_filtered['fecha'] = pd.to_datetime(df_filtered['fecha'], format='%b-%y', errors='coerce')
                df_filtered = df_filtered[df_filtered['fecha'].notna()]
                df_filtered['total_exportaciones'] = df_filtered['total_exportaciones'].apply(self.convert_colombian_number)
                df_filtered = df_filtered.dropna(subset=['total_exportaciones'])

                if df_filtered.empty:
                    self.logger.error(f"No se encontraron datos válidos en {variable}")
                    return variable, None

                df_filtered = df_filtered.sort_values('fecha').reset_index(drop=True)
                nuevo_nombre = f"{target_col}_{variable}_{macro_type}"
                df_filtered.rename(columns={'total_exportaciones': nuevo_nombre}, inplace=True)

                current_min = df_filtered['fecha'].min()
                current_max = df_filtered['fecha'].max()
                if self.global_min_date is None or current_min < self.global_min_date:
                    self.global_min_date = current_min
                if self.global_max_date is None or current_max > self.global_max_date:
                    self.global_max_date = current_max

                self.stats[variable] = {
                    'macro_type': macro_type,
                    'target_column': target_col,
                    'total_rows': len(df_filtered),
                    'valid_values': df_filtered[nuevo_nombre].notna().sum(),
                    'coverage': 100.0,
                    'date_min': current_min,
                    'date_max': current_max,
                    'nuevo_nombre': nuevo_nombre
                }

                self.logger.info(f"   EXITO: {len(df_filtered)} registros válidos procesados")
                self.logger.info(f"   Periodo: {current_min.strftime('%Y-%m-%d')} a {current_max.strftime('%Y-%m-%d')}")

                return variable, df_filtered

            except Exception as e:
                self.logger.error(f"Error procesando {variable}: {str(e)}")
                return variable, None

        def generate_daily_index(self):
            if self.global_min_date is None or self.global_max_date is None:
                self.logger.error("No se pudieron determinar fechas globales")
                return None
            self.daily_index = pd.DataFrame({
                'fecha': pd.date_range(start=self.global_min_date, end=self.global_max_date, freq='D')
            })
            self.logger.info(f"Índice diario: {len(self.daily_index)} días")
            return self.daily_index

        def combine_data(self):
            if self.daily_index is None:
                return None
            combined = self.daily_index.copy()
            for variable, df in self.processed_data.items():
                if df is None or df.empty:
                    continue
                df = df.sort_values('fecha')
                df_daily = pd.merge_asof(combined, df, on='fecha', direction='backward')
                col_name = self.stats[variable]['nuevo_nombre']
                df_daily[col_name] = df_daily[col_name].ffill()
                combined = combined.merge(df_daily[['fecha', col_name]], on='fecha', how='left')
            self.final_df = combined
            self.logger.info(f"Datos combinados: {len(self.final_df)} filas, {len(self.final_df.columns)} columnas")
            return self.final_df

        def save_results(self, output_file):
            if self.final_df is None:
                self.logger.error("No hay datos para guardar")
                return False
            try:
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    self.final_df.to_excel(writer, sheet_name='Datos Diarios', index=False)
                    df_stats = pd.DataFrame(self.stats).T
                    df_stats.to_excel(writer, sheet_name='Estadisticas')
                    meta = {
                        'Proceso': ['DANEExportacionesProcessor'],
                        'Fecha de proceso': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                        'Total indicadores': [len(self.stats)],
                        'Periodo': [f"{self.global_min_date.strftime('%Y-%m-%d')} a {self.global_max_date.strftime('%Y-%m-%d')}"],
                        'Total días': [len(self.daily_index)]
                    }
                    pd.DataFrame(meta).to_excel(writer, sheet_name='Metadatos', index=False)
                self.logger.info(f"Archivo guardado: {output_file}")
                return True
            except Exception as e:
                self.logger.error(f"Error guardando: {str(e)}")
                return False

        def run(self, output_file):
            start_time = time.time()

            if not self.auto_discover_files():
                self.logger.error("No se encontraron archivos DANE Exportaciones")
                return False

            for file_info in self.discovered_files:
                var, df_processed = self.process_file_automatically(file_info)
                self.processed_data[var] = df_processed

            successful = len([df for df in self.processed_data.values() if df is not None])
            if successful == 0:
                self.logger.error("No se procesó ningún archivo correctamente")
                return False

            self.generate_daily_index()
            self.combine_data()
            result = self.save_results(output_file)

            end_time = time.time()
            self.logger.info(f"\\nTiempo: {end_time - start_time:.2f} segundos")
            self.logger.info(f"Archivos procesados: {successful}")
            self.logger.info(f"Estado: {'COMPLETADO' if result else 'ERROR'}")

            return result

            

