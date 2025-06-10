import os
import re
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from src.core.utils import configure_logging, convertir_valor, detectar_formato_fecha_inteligente, convertir_fecha_adaptativo
from .formats import FormatosFechas

class MyinvestingreportNormal:
    """
    Procesa datos económicos con detección dinámica de formatos de fechas y 
    validación para evitar interpretaciones erróneas (como fechas en abril cuando
    los datos crudos solo llegan hasta marzo).
    """
    def __init__(self, config_file, data_root='data/raw', log_file='logs/myinvestingreportnormal.log'):
        self.config_file = config_file
        self.data_root = data_root
        self.logger = configure_logging(log_file)
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
