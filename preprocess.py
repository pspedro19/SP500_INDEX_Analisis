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

# Configurar logging
def configurar_logging(log_file='myinvestingreportcp.log'):
    """Configura el sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('MyinvestingreportCP')

class MyinvestingreportCP:
    """
    Clase para implementar el procesamiento de datos económicos de Investing
    con método Copiar y Pegar
    """
    
    def __init__(self, config_file, log_file='myinvestingreportcp.log'):
        """
        Inicializa el procesador
        
        Args:
            config_file (str): Ruta al archivo de configuración (Data Engineering.xlsx)
            log_file (str): Ruta al archivo de log
        """
        self.config_file = config_file
        self.logger = configurar_logging(log_file)
        self.config_data = None
        self.fecha_min_global = None
        self.fecha_max_global = None
        self.indice_diario = None
        self.datos_procesados = {}
        self.df_combinado = None
        self.estadisticas = {}
        
        self.logger.info("=" * 80)
        self.logger.info("INICIANDO PROCESO: MyinvestingreportCP")
        self.logger.info(f"Archivo de configuración: {config_file}")
        self.logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)
    
    def leer_configuracion(self):
        """
        Lee y filtra la configuración del archivo Excel
        
        Returns:
            pd.DataFrame: Configuraciones filtradas
        """
        try:
            self.logger.info("Leyendo archivo de configuración...")
            # Leer archivo de configuración
            df_config = pd.read_excel(self.config_file)
            
            # Filtrar por tipo de preprocesamiento y fuente
            self.config_data = df_config[
                (df_config['Fuente'] == 'Investing Data') & 
                (df_config['Tipo de Preprocesamiento Según la Fuente'] == 'Copiar y Pegar')
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
    
    def extraer_fecha(self, texto_fecha):
        """
        Extrae una fecha del formato "Apr 01, 2025 (Mar)" o similar
        
        Args:
            texto_fecha: Texto con la fecha a extraer
            
        Returns:
            pd.Timestamp: Fecha extraída o None si no se pudo procesar
        """
        if not isinstance(texto_fecha, str):
            return None
        
        # Buscar formato "Apr 01, 2025 (Mar)"
        match = re.search(r'([A-Za-z]+\s+\d+,\s+\d{4})', texto_fecha)
        if match:
            try:
                return pd.to_datetime(match.group(1))
            except:
                pass
        
        # Intentar parsear directamente
        try:
            return pd.to_datetime(texto_fecha)
        except:
            return None
    
    def procesar_archivo(self, config_row):
        """
        Procesa un archivo individual según la configuración
        
        Args:
            config_row (pd.Series): Fila de configuración del archivo
            
        Returns:
            tuple: (nombre_indicador, pd.DataFrame procesado) o (nombre_indicador, None) si hay error
        """
        variable = config_row['Variable']
        tipo_macro = config_row['Tipo Macro']
        target_col = config_row['TARGET']
        
        # Determinar ruta del archivo
        archivo = f"{variable}.xlsx"
        
        self.logger.info(f"\nProcesando: {variable} ({tipo_macro})")
        self.logger.info(f"- Archivo: {archivo}")
        self.logger.info(f"- Columna TARGET: {target_col}")
        
        try:
            # Leer archivo Excel
            df = pd.read_excel(archivo)
            total_filas = len(df)
            self.logger.info(f"- Filas encontradas: {total_filas}")
            
            # Verificar si existe columna de fecha
            if 'Release Date' not in df.columns:
                self.logger.error(f"- ERROR: No se encontró columna 'Release Date' en {archivo}")
                return variable, None
                
            # Verificar si existe columna TARGET
            columna_encontrada = None
            if target_col in df.columns:
                columna_encontrada = target_col
            else:
                # Buscar columnas alternativas
                columnas_numericas = [col for col in df.columns 
                             if col not in ['Release Date', 'Time', 'Previous', 'Forecast']
                             and pd.to_numeric(df[col], errors='coerce').notna().any()]
                
                if columnas_numericas:
                    columna_encontrada = columnas_numericas[0]
                    self.logger.warning(f"- AVISO: No se encontró columna '{target_col}', usando '{columna_encontrada}'")
                else:
                    self.logger.error(f"- ERROR: No se encontró columna TARGET ni alternativas en {archivo}")
                    return variable, None
            
            # Procesar fechas
            df['fecha'] = df['Release Date'].apply(self.extraer_fecha)
            df = df.dropna(subset=['fecha'])
            
            # Contar fechas procesadas correctamente
            fechas_procesadas = len(df)
            if fechas_procesadas < total_filas:
                self.logger.warning(f"- AVISO: {total_filas - fechas_procesadas} fechas no pudieron ser procesadas")
            
            # Extraer valores de TARGET
            df['valor'] = pd.to_numeric(df[columna_encontrada], errors='coerce')
            df = df.dropna(subset=['valor'])
            
            valores_validos = len(df)
            cobertura = (valores_validos / total_filas) * 100
            
            # Renombrar columna según el patrón
            nuevo_nombre = f"{target_col}_{variable}_{tipo_macro}"
            df.rename(columns={'valor': nuevo_nombre}, inplace=True)
            
            # Seleccionar solo las columnas relevantes
            df_procesado = df[['fecha', nuevo_nombre]].copy()
            
            # Ordenar por fecha
            df_procesado = df_procesado.sort_values('fecha')
            
            # Calcular fechas mínima y máxima
            fecha_min = df_procesado['fecha'].min()
            fecha_max = df_procesado['fecha'].max()
            
            # Actualizar fechas mínima y máxima globales
            if self.fecha_min_global is None or fecha_min < self.fecha_min_global:
                self.fecha_min_global = fecha_min
                
            if self.fecha_max_global is None or fecha_max > self.fecha_max_global:
                self.fecha_max_global = fecha_max
            
            # Registrar estadísticas
            self.estadisticas[variable] = {
                'tipo_macro': tipo_macro,
                'columna_target': columna_encontrada,
                'total_filas': total_filas,
                'valores_validos': valores_validos,
                'cobertura': cobertura,
                'fecha_min': fecha_min,
                'fecha_max': fecha_max,
                'nuevo_nombre': nuevo_nombre
            }
            
            estado = "OK" if cobertura >= 75 else "ALERTA"
            
            self.logger.info(f"- Valores no nulos en TARGET: {valores_validos}")
            self.logger.info(f"- Cobertura: {cobertura:.2f}%")
            self.logger.info(f"- Periodo: {fecha_min.strftime('%Y-%m-%d')} a {fecha_max.strftime('%Y-%m-%d')}")
            self.logger.info(f"- Estado: {estado}")
            
            return variable, df_procesado
            
        except FileNotFoundError:
            self.logger.error(f"- ERROR: Archivo no encontrado: {archivo}")
            return variable, None
        except Exception as e:
            self.logger.error(f"- ERROR al procesar {archivo}: {str(e)}")
            return variable, None
    
    def generar_indice_diario(self):
        """
        Genera un DataFrame con índice diario entre fechas mínima y máxima globales
        
        Returns:
            pd.DataFrame: DataFrame con índice diario
        """
        if self.fecha_min_global is None or self.fecha_max_global is None:
            self.logger.error("No se pudieron determinar fechas mínima y máxima globales")
            return None
            
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
        
        Returns:
            pd.DataFrame: DataFrame combinado con todos los indicadores
        """
        if not self.datos_procesados:
            self.logger.error("No hay datos procesados para combinar")
            return None
            
        if self.indice_diario is None:
            self.logger.error("No se ha generado el índice diario")
            return None
            
        self.logger.info("\nCombinando datos con índice diario...")
        
        # Comenzar con el índice diario
        df_combinado = self.indice_diario.copy()
        
        # Para cada indicador, realizar el merge y aplicar ffill
        for variable, df in self.datos_procesados.items():
            if df is None:
                self.logger.warning(f"Omitiendo {variable} por errores de procesamiento")
                continue
                
            nombre_col = df.columns[1]  # La columna de valores (después de 'fecha')
            
            self.logger.info(f"- Combinando: {nombre_col}")
            
            # Realizar merge
            df_combinado = pd.merge(df_combinado, df, on='fecha', how='left')
            
            # Aplicar forward fill (ffill)
            df_combinado[nombre_col] = df_combinado[nombre_col].ffill()
            
            # Calcular métricas después de ffill
            valores_antes = self.estadisticas[variable]['valores_validos']
            valores_despues = df_combinado[nombre_col].notna().sum()
            valores_imputados = valores_despues - valores_antes
            cobertura_final = (valores_despues / len(df_combinado)) * 100
            
            # Actualizar estadísticas
            self.estadisticas[variable].update({
                'valores_despues_ffill': valores_despues,
                'valores_imputados': valores_imputados,
                'cobertura_final': cobertura_final
            })
        
        self.df_combinado = df_combinado
        self.logger.info(f"- DataFrame combinado: {len(df_combinado)} filas, {len(df_combinado.columns)} columnas")
        
        return self.df_combinado
    
    def analizar_cobertura_final(self):
        """
        Genera un informe detallado de cobertura final
        """
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
        self.logger.info("\nCobertura por indicador:")
        
        # Contadores por nivel de cobertura
        coberturas = {
            "Excelente (>90%)": 0,
            "Buena (75-90%)": 0,
            "Regular (50-75%)": 0,
            "Baja (25-50%)": 0,
            "Crítica (<25%)": 0
        }
        
        # Ordenar por cobertura final descendente
        indicadores_ordenados = sorted(
            self.estadisticas.items(), 
            key=lambda x: x[1].get('cobertura_final', 0), 
            reverse=True
        )
        
        for variable, stats in indicadores_ordenados:
            cobertura = stats.get('cobertura_final', 0)
            estado = self._obtener_estado_cobertura(cobertura)
            
            self.logger.info(f"- {variable} ({stats['nuevo_nombre']}): {cobertura:.2f}% [{estado}]")
            
            # Incrementar contador correspondiente
            if cobertura > 90:
                coberturas["Excelente (>90%)"] += 1
            elif cobertura > 75:
                coberturas["Buena (75-90%)"] += 1
            elif cobertura > 50:
                coberturas["Regular (50-75%)"] += 1
            elif cobertura > 25:
                coberturas["Baja (25-50%)"] += 1
            else:
                coberturas["Crítica (<25%)"] += 1
        
        self.logger.info("\nDistribución de cobertura:")
        for rango, num in coberturas.items():
            self.logger.info(f"- {rango}: {num} indicadores")
        
        # Añadir información sobre valores imputados
        self.logger.info("\nImputación de datos:")
        total_valores = total_dias * total_indicadores
        total_originales = sum(s['valores_validos'] for s in self.estadisticas.values())
        total_imputados = sum(s.get('valores_imputados', 0) for s in self.estadisticas.values())
        
        self.logger.info(f"- Valores originales: {total_originales} ({total_originales/total_valores*100:.2f}%)")
        self.logger.info(f"- Valores imputados: {total_imputados} ({total_imputados/total_valores*100:.2f}%)")
    
    def _obtener_estado_cobertura(self, cobertura):
        """Determina el estado según el porcentaje de cobertura"""
        if cobertura > 90:
            return "EXCELENTE"
        elif cobertura > 75:
            return "BUENO"
        elif cobertura > 50:
            return "REGULAR"
        elif cobertura > 25:
            return "BAJO"
        else:
            return "CRÍTICO"
    
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
            
            # Identificar valores atípicos (más de 3 desviaciones estándar)
            umbral_superior = stats['mean'] + 3 * stats['std']
            umbral_inferior = stats['mean'] - 3 * stats['std']
            valores_atipicos = serie[(serie > umbral_superior) | (serie < umbral_inferior)]
            
            self.logger.info(f"\nEstadísticas para {col}:")
            self.logger.info(f"- Min: {stats['min']:.4f}")
            self.logger.info(f"- Max: {stats['max']:.4f}")
            self.logger.info(f"- Media: {stats['mean']:.4f}")
            self.logger.info(f"- Mediana: {stats['median']:.4f}")
            self.logger.info(f"- Desv. Estándar: {stats['std']:.4f}")
            
            if len(valores_atipicos) > 0:
                pct_atipicos = (len(valores_atipicos) / len(serie)) * 100
                self.logger.info(f"- ALERTA: Se encontraron {len(valores_atipicos)} valores atípicos ({pct_atipicos:.2f}%)")
                self.logger.info(f"  Umbral inferior: {umbral_inferior:.4f}, Umbral superior: {umbral_superior:.4f}")
            
            # Guardar estadísticas en el diccionario
            for var, stats_dict in self.estadisticas.items():
                if stats_dict.get('nuevo_nombre') == col:
                    stats_dict.update(stats)
                    stats_dict['valores_atipicos'] = len(valores_atipicos)
    
    def guardar_resultados(self, output_file='datos_economicos_procesados.xlsx'):
        """
        Guarda los resultados procesados en un archivo Excel
        
        Args:
            output_file (str): Ruta del archivo de salida
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
                # Guardar datos diarios
                self.df_combinado.to_excel(writer, sheet_name='Datos Diarios', index=False)
                
                # Generar hoja de estadísticas
                df_stats = pd.DataFrame()
                for var, stats in self.estadisticas.items():
                    # Convertir diccionario a series y transponer para obtener una fila
                    serie = pd.Series(stats, name=var)
                    df_temp = pd.DataFrame(serie).transpose()
                    df_stats = pd.concat([df_stats, df_temp])
                
                # Guardar estadísticas
                df_stats.to_excel(writer, sheet_name='Estadisticas')
                
                # Guardar metadatos
                metadata = {
                    'Proceso': ['MyinvestingreportCP'],
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
    
    def ejecutar_proceso_completo(self, output_file='datos_economicos_procesados.xlsx'):
        """
        Ejecuta el proceso completo de preprocesamiento
        
        Args:
            output_file (str): Ruta del archivo de salida
            
        Returns:
            bool: True si el proceso se completó exitosamente, False en caso contrario
        """
        inicio = time.time()
        self.logger.info("Iniciando proceso completo MyinvestingreportCP...")
        
        # 1. Leer configuración
        self.leer_configuracion()
        if self.config_data is None or len(self.config_data) == 0:
            return False
        
        # 2. Procesar cada archivo
        for _, config_row in self.config_data.iterrows():
            variable, df_procesado = self.procesar_archivo(config_row)
            self.datos_procesados[variable] = df_procesado
        
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
        self.logger.info(f"Proceso: MyinvestingreportCP")
        self.logger.info(f"Tiempo de ejecución: {tiempo_ejecucion:.2f} segundos")
        self.logger.info(f"Archivos procesados: {len(self.datos_procesados)}")
        self.logger.info(f"Archivos con error: {sum(1 for df in self.datos_procesados.values() if df is None)}")
        self.logger.info(f"Periodo de datos: {self.fecha_min_global.strftime('%Y-%m-%d')} a {self.fecha_max_global.strftime('%Y-%m-%d')}")
        self.logger.info(f"Datos combinados: {len(self.df_combinado)} filas x {len(self.df_combinado.columns)} columnas")
        self.logger.info(f"Archivo de salida: {output_file}")
        self.logger.info(f"Estado: {'COMPLETADO' if resultado else 'ERROR'}")
        self.logger.info("=" * 50)
        
        return resultado


# Función principal para ejecutar el proceso
def ejecutar_myinvestingreportcp(config_file='Data Engineering.xlsx',
                                 output_file='datos_economicos_procesados.xlsx',
                                 log_file='myinvestingreportcp.log'):
    """
    Ejecuta el proceso MyinvestingreportCP
    
    Args:
        config_file (str): Ruta al archivo de configuración
        output_file (str): Ruta al archivo de salida
        log_file (str): Ruta al archivo de log
        
    Returns:
        bool: True si el proceso se completó exitosamente, False en caso contrario
    """
    procesador = MyinvestingreportCP(config_file, log_file)
    return procesador.ejecutar_proceso_completo(output_file)


# Ejemplo de uso
if __name__ == "__main__":
    resultado = ejecutar_myinvestingreportcp()
    print(f"Proceso {'completado exitosamente' if resultado else 'finalizado con errores'}")