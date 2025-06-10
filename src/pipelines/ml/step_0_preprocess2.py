"""
🌟 PROCESADOR INTEGRADO: TES + EOE UNIVERSAL
Combina el procesamiento de Tasas Cero Cupón TES y todos los Índices de Confianza EOE
Mantiene la funcionalidad independiente de cada procesador
"""
import os, fnmatch
import pandas as pd
import numpy as np
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
import re
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SECCIÓN 1: TES PROCESSOR - Tasas Cero Cupón
# =============================================================================
import os
import logging
from datetime import datetime

# 🧠 Definir logging UTF-8 compatible (antes de usarlo)
def configurar_logging(log_file='logs/eoe_universal_processor.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Limpiar handlers previos si existen
    if logger.hasHandlers():
        logger.handlers.clear()

    # Archivo con UTF-8
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Consola (no redirige stdout para evitar errores de cierre)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# =============================================================================
# REEMPLAZA LA CLASE PIBProcessor EXISTENTE CON ESTA VERSIÓN MEJORADA
# =============================================================================

class EconomiaProcessor:
    """
    🏗️ ECONOMIA PROCESSOR UNIFICADO - PIB + ISE
    
    ✅ PROCESA AMBOS ARCHIVOS:
    - PIB: anexProduccionConstantesItrim2025.xlsx (Trimestral)
    - ISE: anexISE12actividadesmar2025.xlsx (Mensual)
    
    ✅ GENERA SALIDA UNIFICADA:
    - Ambas series convertidas a frecuencia diaria
    - Forward-fill para completar días faltantes
    - Excel con múltiples hojas y estadísticas
    
    🎯 ESTRUCTURA IDENTIFICADA:
    PIB: Fila 11=Años, Fila 12=Trimestres, Fila 28="Producto Interno Bruto"
    ISE: Fila 10=Años, Fila 11=Meses, Fila 28="Indicador de Seguimiento a la Economía"
    """
    
    # Mapeos de períodos a fechas
    quarter_map = {
        'I': '01-01', 'II': '04-01', 'III': '07-01', 'IV': '10-01'
    }
    
    month_map = {
        'enero': '01-01', 'febrero': '02-01', 'marzo': '03-01',
        'abril': '04-01', 'mayo': '05-01', 'junio': '06-01',
        'julio': '07-01', 'agosto': '08-01', 'septiembre': '09-01',
        'octubre': '10-01', 'noviembre': '11-01', 'diciembre': '12-01'
    }

    def __init__(self, search_paths=None, log_file='logs/economia_processor.log'):
        """Inicializar procesador unificado - RUTAS CORREGIDAS"""
        # Usar exactamente las mismas rutas que PIBProcessor
        self.search_paths = search_paths or [
            '.', 'data', 'data/0_raw', 'downloads', 'raw_data',
            'Data', 'Data/0_raw', 'Data/0_raw/bond',  # ← AGREGAR ESTAS RUTAS
            './Data', './Data/0_raw', './Data/0_raw/bond'
        ]
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('EconomiaProcessor')
        
        # Datos internos
        self.pib_df = None
        self.ise_df = None
        self.daily_index = None
        self.final_df = None
        self.stats = {}
        
        self.logger.info("=" * 80)
        self.logger.info("🏗️ ECONOMIA PROCESSOR UNIFICADO - PIB + ISE")
        self.logger.info(f"📅 Fecha de procesamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"🔍 Rutas de búsqueda: {self.search_paths}")
        self.logger.info("=" * 80)
        
    def find_files(self):
        """🔍 Buscar ambos archivos automáticamente - VERSIÓN CORREGIDA"""
        files_found = {}
        
        # Patrones de búsqueda CORREGIDOS - usando los mismos que funcionan individualmente
        patterns = {
            'pib': [
                '*anexProduccionConstantes*trim*.xlsx',  # Patrón más amplio primero
                '*anex-ProduccionConstantes*trim*.xlsx', 
                'anexProduccionConstantesItrim2025.xlsx',  # Específico después
                '*PIB*trim*.xlsx',
                '*Produccion*Constantes*.xlsx'
            ],
            'ise': [
                '*anexISE*actividades*.xlsx',  # Patrón más amplio
                '*anex-ISE*actividades*.xlsx',  # Con guión
                'anexISE12actividadesmar2025.xlsx',  # Específico
                '*ISE*.xlsx'
            ]
        }
        
        self.logger.info("🔍 Buscando archivos de economía...")
        
        for file_type, patterns_list in patterns.items():
            self.logger.info(f"   🔍 Buscando {file_type.upper()}...")
            
            for root in self.search_paths:
                if not os.path.exists(root):
                    self.logger.debug(f"      ⚠️ Ruta no existe: {root}")
                    continue
                    
                self.logger.info(f"      📁 Explorando: {root}")
                
                for dp, _, files in os.walk(root):
                    for f in files:
                        for pat in patterns_list:
                            if fnmatch.fnmatch(f, pat):
                                full_path = os.path.join(dp, f)
                                files_found[file_type] = full_path
                                self.logger.info(f"✅ {file_type.upper()} encontrado: {full_path}")
                                break
                        if file_type in files_found:
                            break
                    if file_type in files_found:
                        break
                if file_type in files_found:
                    break
            
            if file_type not in files_found:
                self.logger.warning(f"⚠️ {file_type.upper()} no encontrado en:")
                for root in self.search_paths:
                    self.logger.warning(f"   - {root}")
                self.logger.warning(f"   Patrones buscados: {patterns_list}")
        
        # Verificar que encontramos ambos
        missing = []
        if 'pib' not in files_found:
            missing.append('PIB')
        if 'ise' not in files_found:
            missing.append('ISE')
        
        if missing:
            self.logger.error(f"❌ No se encontraron: {', '.join(missing)}")
            
            # Mostrar archivos disponibles para debug
            self.logger.info("🔍 Archivos disponibles en las rutas:")
            for root in self.search_paths:
                if os.path.exists(root):
                    for dp, _, files in os.walk(root):
                        excel_files = [f for f in files if f.endswith(('.xlsx', '.xls'))]
                        if excel_files:
                            self.logger.info(f"   📁 {dp}:")
                            for f in excel_files[:5]:  # Solo mostrar primeros 5
                                self.logger.info(f"      - {f}")
                            if len(excel_files) > 5:
                                self.logger.info(f"      ... y {len(excel_files)-5} más")
            
            return None, None
        
        return files_found.get('pib'), files_found.get('ise')
        
        # Verificar que encontramos ambos
        missing = []
        if 'pib' not in files_found:
            missing.append('PIB (anexProduccionConstantesItrim2025.xlsx)')
        if 'ise' not in files_found:
            missing.append('ISE (anexISE12actividadesmar2025.xlsx)')
        
        if missing:
            self.logger.error(f"❌ No se encontraron: {', '.join(missing)}")
            return None, None
        
        return files_found.get('pib'), files_found.get('ise')

    def process_pib_file(self, file_path):
        """📊 Procesar archivo PIB (trimestral)"""
        self.logger.info(f"📊 Procesando PIB: {file_path}")
        
        try:
            # Leer archivo
            raw_df = pd.read_excel(file_path, sheet_name='Cuadro 1', header=None, dtype=object)
            
            # Extraer filas específicas para PIB
            años_row = raw_df.iloc[11]      # Fila 11: años
            trimestres_row = raw_df.iloc[12] # Fila 12: trimestres
            pib_row = raw_df.iloc[28]       # Fila 28: PIB
            
            # Verificar concepto
            concepto = str(pib_row.iloc[2]).strip()
            if 'Producto Interno Bruto' not in concepto:
                self.logger.warning(f"⚠️ PIB - Concepto inesperado: '{concepto}'")
            
            # Extraer datos
            fechas_pib = []
            valores_pib = []
            año_actual = None
            
            for col_idx in range(3, len(pib_row)):
                # Obtener año
                año_cell = años_row.iloc[col_idx] if col_idx < len(años_row) else None
                if pd.notna(año_cell) and str(año_cell).strip():
                    import re
                    año_match = re.search(r'(\d{4})', str(año_cell))
                    if año_match:
                        año_actual = int(año_match.group(1))
                
                # Obtener trimestre
                trim_cell = trimestres_row.iloc[col_idx] if col_idx < len(trimestres_row) else None
                trim_str = str(trim_cell).strip() if pd.notna(trim_cell) else None
                
                # Obtener valor
                valor_cell = pib_row.iloc[col_idx] if col_idx < len(pib_row) else None
                
                if (año_actual and trim_str in self.quarter_map and pd.notna(valor_cell)):
                    try:
                        valor_num = float(valor_cell)
                        fecha_str = f"{año_actual}-{self.quarter_map[trim_str]}"
                        fecha_dt = pd.to_datetime(fecha_str)
                        
                        fechas_pib.append(fecha_dt)
                        valores_pib.append(valor_num)
                    except (ValueError, TypeError):
                        continue
            
            if len(fechas_pib) == 0:
                self.logger.error("❌ No se pudieron extraer datos del PIB")
                return False
            
            self.pib_df = pd.DataFrame({
                'fecha': fechas_pib,
                'pib': valores_pib
            }).sort_values('fecha').reset_index(drop=True)
            
            self.logger.info(f"✅ PIB procesado: {len(self.pib_df)} trimestres")
            self.logger.info(f"   📅 Período PIB: {self.pib_df['fecha'].min().strftime('%Y-%m-%d')} "
                           f"a {self.pib_df['fecha'].max().strftime('%Y-%m-%d')}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error procesando PIB: {str(e)}")
            return False

    def process_ise_file(self, file_path):
        """📊 Procesar archivo ISE (mensual)"""
        self.logger.info(f"📊 Procesando ISE: {file_path}")
        
        try:
            # Leer archivo
            raw_df = pd.read_excel(file_path, sheet_name='Cuadro 1', header=None, dtype=object)
            
            # Extraer filas específicas para ISE (estructura ligeramente diferente)
            años_row = raw_df.iloc[10]      # Fila 10: años (no 11 como PIB)
            meses_row = raw_df.iloc[11]     # Fila 11: meses
            ise_row = raw_df.iloc[28]       # Fila 28: ISE
            
            # Verificar concepto
            concepto = str(ise_row.iloc[2]).strip()
            if 'Indicador de Seguimiento' not in concepto:
                self.logger.warning(f"⚠️ ISE - Concepto inesperado: '{concepto}'")
            
            # Extraer datos
            fechas_ise = []
            valores_ise = []
            año_actual = None
            
            for col_idx in range(3, len(ise_row)):
                # Obtener año
                año_cell = años_row.iloc[col_idx] if col_idx < len(años_row) else None
                if pd.notna(año_cell) and str(año_cell).strip():
                    import re
                    año_match = re.search(r'(\d{4})', str(año_cell))
                    if año_match:
                        año_actual = int(año_match.group(1))
                
                # Obtener mes
                mes_cell = meses_row.iloc[col_idx] if col_idx < len(meses_row) else None
                mes_str = str(mes_cell).lower().strip() if pd.notna(mes_cell) else None
                
                # Obtener valor
                valor_cell = ise_row.iloc[col_idx] if col_idx < len(ise_row) else None
                
                if (año_actual and mes_str in self.month_map and pd.notna(valor_cell)):
                    try:
                        valor_num = float(valor_cell)
                        fecha_str = f"{año_actual}-{self.month_map[mes_str]}"
                        fecha_dt = pd.to_datetime(fecha_str)
                        
                        fechas_ise.append(fecha_dt)
                        valores_ise.append(valor_num)
                    except (ValueError, TypeError):
                        continue
            
            if len(fechas_ise) == 0:
                self.logger.error("❌ No se pudieron extraer datos del ISE")
                return False
            
            self.ise_df = pd.DataFrame({
                'fecha': fechas_ise,
                'ise': valores_ise
            }).sort_values('fecha').reset_index(drop=True)
            
            self.logger.info(f"✅ ISE procesado: {len(self.ise_df)} meses")
            self.logger.info(f"   📅 Período ISE: {self.ise_df['fecha'].min().strftime('%Y-%m-%d')} "
                           f"a {self.ise_df['fecha'].max().strftime('%Y-%m-%d')}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error procesando ISE: {str(e)}")
            return False

    def generate_unified_daily_series(self):
        """🔗 Combinar PIB + ISE en serie diaria unificada"""
        if self.pib_df is None or self.ise_df is None:
            self.logger.error("❌ Faltan datos PIB o ISE para combinar")
            return False
        
        self.logger.info("🔗 Generando serie diaria unificada PIB + ISE...")
        
        # Determinar rango global de fechas
        min_date = min(self.pib_df['fecha'].min(), self.ise_df['fecha'].min())
        max_date = max(self.pib_df['fecha'].max(), self.ise_df['fecha'].max())
        
        # Generar índice diario
        self.daily_index = pd.DataFrame({
            'fecha': pd.date_range(start=min_date, end=max_date, freq='D')
        })
        
        self.logger.info(f"📅 Rango unificado: {min_date.strftime('%Y-%m-%d')} a {max_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"📊 Días totales: {len(self.daily_index)}")
        
        # Combinar con merge_asof + forward fill
        combined = self.daily_index.copy()
        
        # Agregar PIB
        pib_daily = pd.merge_asof(combined, self.pib_df, on='fecha', direction='backward')
        combined['pib'] = pib_daily['pib'].ffill()
        
        # Agregar ISE
        ise_daily = pd.merge_asof(combined, self.ise_df, on='fecha', direction='backward')
        combined['ise'] = ise_daily['ise'].ffill()
        
        self.final_df = combined
        
        # Estadísticas de cobertura
        pib_coverage = self.final_df['pib'].notna().sum()
        ise_coverage = self.final_df['ise'].notna().sum()
        total_days = len(self.final_df)
        
        self.logger.info(f"✅ Serie unificada creada:")
        self.logger.info(f"   📊 PIB cobertura: {pib_coverage}/{total_days} días ({pib_coverage/total_days*100:.1f}%)")
        self.logger.info(f"   📊 ISE cobertura: {ise_coverage}/{total_days} días ({ise_coverage/total_days*100:.1f}%)")
        
        return True

    def save_unified_results(self, output_file='economia_integrada.xlsx'):
        """💾 Guardar resultados unificados"""
        if self.final_df is None:
            self.logger.error("❌ No hay datos para guardar")
            return False
        
        try:
            self.logger.info(f"💾 Guardando economía unificada en: {output_file}")
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Hoja 1: Serie diaria unificada
                self.final_df.to_excel(writer, sheet_name='Economia_Diaria', index=False)
                
                # Hoja 2: PIB trimestral original
                if self.pib_df is not None:
                    self.pib_df.to_excel(writer, sheet_name='PIB_Trimestral', index=False)
                
                # Hoja 3: ISE mensual original
                if self.ise_df is not None:
                    self.ise_df.to_excel(writer, sheet_name='ISE_Mensual', index=False)
                
                # Hoja 4: Estadísticas comparativas
                stats_data = [
                    ['Métrica', 'PIB', 'ISE'],
                    ['Frecuencia original', 'Trimestral', 'Mensual'],
                    ['Total períodos', len(self.pib_df) if self.pib_df is not None else 0, 
                     len(self.ise_df) if self.ise_df is not None else 0],
                    ['Fecha inicio', 
                     self.pib_df['fecha'].min().strftime('%Y-%m-%d') if self.pib_df is not None else 'N/A',
                     self.ise_df['fecha'].min().strftime('%Y-%m-%d') if self.ise_df is not None else 'N/A'],
                    ['Fecha fin',
                     self.pib_df['fecha'].max().strftime('%Y-%m-%d') if self.pib_df is not None else 'N/A',
                     self.ise_df['fecha'].max().strftime('%Y-%m-%d') if self.ise_df is not None else 'N/A'],
                    ['Valor promedio',
                     f"{self.pib_df['pib'].mean():,.0f}" if self.pib_df is not None else 'N/A',
                     f"{self.ise_df['ise'].mean():.1f}" if self.ise_df is not None else 'N/A'],
                    ['Días diarios generados', len(self.final_df), len(self.final_df)],
                    ['Procesado en', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                     datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                ]
                
                stats_df = pd.DataFrame(stats_data[1:], columns=stats_data[0])
                stats_df.to_excel(writer, sheet_name='Estadisticas_Comparativas', index=False)
            
            self.logger.info(f"✅ Archivo guardado exitosamente: {output_file}")
            self.logger.info(f"📊 Hojas: Economia_Diaria, PIB_Trimestral, ISE_Mensual, Estadisticas_Comparativas")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error guardando resultados: {str(e)}")
            return False

    def run(self, pib_file=None, ise_file=None, output_file='economia_integrada.xlsx'):
        """🚀 Ejecutar procesamiento unificado completo"""
        start_time = datetime.now()
        
        self.logger.info("🚀 Iniciando procesamiento unificado PIB + ISE...")
        
        # PASO 1: Encontrar archivos
        if pib_file is None or ise_file is None:
            pib_found, ise_found = self.find_files()
            pib_file = pib_file or pib_found
            ise_file = ise_file or ise_found
            
            if not pib_file or not ise_file:
                return False
        
        # PASO 2: Procesar PIB
        if not self.process_pib_file(pib_file):
            return False
        
        # PASO 3: Procesar ISE
        if not self.process_ise_file(ise_file):
            return False
        
        # PASO 4: Generar serie unificada
        if not self.generate_unified_daily_series():
            return False
        
        # PASO 5: Guardar resultados
        if not self.save_unified_results(output_file):
            return False
        
        # Resumen final
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🎯 PROCESAMIENTO UNIFICADO COMPLETADO EXITOSAMENTE")
        self.logger.info("=" * 80)
        self.logger.info(f"⏱️  Tiempo total: {duration:.2f} segundos")
        self.logger.info(f"📁 PIB procesado: {pib_file}")
        self.logger.info(f"📁 ISE procesado: {ise_file}")
        self.logger.info(f"📊 Datos PIB: {len(self.pib_df)} trimestres")
        self.logger.info(f"📊 Datos ISE: {len(self.ise_df)} meses")
        self.logger.info(f"📅 Serie diaria: {len(self.final_df)} días")
        self.logger.info(f"💾 Archivo guardado: {output_file}")
        self.logger.info("=" * 80)
        
        return True


# =============================================================================
# FUNCIÓN DE UTILIDAD PARA EL SISTEMA PRINCIPAL
# =============================================================================

def procesar_economia_integrada(pib_file=None, ise_file=None, output_file='economia_integrada.xlsx',
                               search_paths=None, log_file='logs/economia_processor.log'):
    """
    🚀 Función principal para procesar PIB + ISE de forma unificada
    
    Args:
        pib_file (str, optional): Ruta al archivo PIB
        ise_file (str, optional): Ruta al archivo ISE  
        output_file (str): Archivo Excel de salida unificado
        search_paths (list, optional): Rutas de búsqueda
        log_file (str): Archivo de log
    
    Returns:
        bool: True si el procesamiento fue exitoso
        
    Ejemplo de uso:
        # Búsqueda automática de ambos archivos
        success = procesar_economia_integrada()
        
        # Archivos específicos
        success = procesar_economia_integrada(
            pib_file='anexProduccionConstantesItrim2025.xlsx',
            ise_file='anexISE12actividadesmar2025.xlsx'
        )
    """
    processor = EconomiaProcessor(search_paths, log_file)
    return processor.run(pib_file, ise_file, output_file)
# =============================================================================
# FUNCIÓN DE UTILIDAD PARA EL SISTEMA PRINCIPAL
# =============================================================================

def procesar_pib_trimestral(file_path=None, output_file='pib_diario_procesado.xlsx',
                           search_paths=None, log_file='logs/pib_processor.log'):
    """
    🚀 Función principal para procesar el PIB trimestral
    
    Args:
        file_path (str, optional): Ruta específica al archivo PIB
        output_file (str): Archivo Excel de salida
        search_paths (list, optional): Rutas de búsqueda personalizadas
        log_file (str): Archivo de log
    
    Returns:
        bool: True si el procesamiento fue exitoso
        
    Ejemplo de uso:
        # Búsqueda automática
        success = procesar_pib_trimestral()
        
        # Archivo específico
        success = procesar_pib_trimestral('data/anexProduccionConstantesItrim2025.xlsx')
        
        # Personalizado
        success = procesar_pib_trimestral(
            output_file='mi_pib_data.xlsx',
            search_paths=['downloads', 'data/raw']
        )
    """
    processor = PIBProcessor(search_paths, log_file)
    return processor.run(file_path, output_file)


# =============================================================================
# EJEMPLO DE USO
# =============================================================================



    # Opción 2: Archivo específico (descomenta para probar)
    # print("\n2️⃣ Procesamiento de archivo específico:")
    # success2 = procesar_pib_trimestral(
    #     file_path='anexProduccionConstantesItrim2025.xlsx',
    #     output_file='pib_custom.xlsx'
    # )

class RemesasProcessor:
    """
    Procesador de Excel de remesas: lee un archivo con columnas Fecha, Descripción y VALOR_REMESA,
    convierte la fecha, genera un índice diario, hace forward-fill y guarda resultados.
    """
    def __init__(self, log_file='logs/remesas_processor.log'):
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('RemesasProcessor')
        self.df = None
        self.daily_index = None
        self.final_df = None

    def read_file(self, file_path):
        """
        Lee el Excel de remesas y normaliza nombres de columnas de forma dinámica,
        soportando tanto archivos perfectamente tabulados como aquellos que vienen
        en una sola columna con tabuladores internos.
        """
        self.logger.info(f"📥 Leyendo archivo: {file_path}")

        # 1) Leer sin header para inspeccionar
        raw = pd.read_excel(file_path, header=None, sheet_name=0, dtype=str)

        # 2) Si solo hay UNA columna pero contiene tabuladores, explódela
        if raw.shape[1] == 1 and raw.iloc[:,0].str.contains('\t').any():
            exploded = raw[0].str.split('\t', expand=True)
            self.logger.info("🔍 Detected single-column with tabs → expanding")
        else:
            exploded = raw

        # 3) Encontrar la fila real de encabezado (donde aparezca 'Fecha')
        header_idx = None
        for i in range(min(10, len(exploded))):
            row = exploded.iloc[i].astype(str).str.lower()
            if 'fecha' in row.values:
                header_idx = i
                self.logger.info(f"✅ Header encontrado en fila {i}")
                break
        if header_idx is None:
            header_idx = 0
            self.logger.warning("⚠️ No encontré fila con 'Fecha', usando fila 0 como header")

        # 4) Extraer encabezado y datos
        header = exploded.iloc[header_idx].astype(str).str.strip()
        df = exploded.iloc[header_idx+1 : ].copy().reset_index(drop=True)
        df.columns = header

        # 5) Mapeo dinámico de columnas a nuestro estándar
        col_map = {}
        for col in df.columns:
            low = col.lower()
            if 'fecha' in low:
                col_map[col] = 'fecha'
            elif 'descrip' in low:
                col_map[col] = 'descripcion'
            elif 'valor' in low or 'remesa' in low:
                col_map[col] = 'valor_remesa'

        df = df.rename(columns=col_map)

        # 6) Verificar que ahora tengamos las 3 columnas
        expected = ['fecha', 'descripcion', 'valor_remesa']
        missing = [c for c in expected if c not in df.columns]
        if missing:
            self.logger.error(f"❌ Faltan columnas tras mapeo: {missing}")
            raise KeyError(f"Faltan columnas: {missing}")

        # 7) Dejar solo las que nos interesan y asignar a self.df
        self.df = df[expected]
        self.logger.info(f"✅ Columnas normalizadas: {self.df.columns.tolist()}")

    def convert_dates(self):
        """
        Convierte la columna 'fecha' de cadenas como 'MM/DD/YYYY hh:mm:ss.fff AM/PM' a datetime.date.
        """
        self.logger.info("🔄 Convirtiendo fechas...")
        # Ejemplo de formato: '04/30/2025 12:00:00.000 AM'
        self.df['fecha'] = pd.to_datetime(
            self.df['fecha'],
            format='%m/%d/%Y %I:%M:%S.%f %p',
            errors='coerce'
        )
        nulldates = self.df['fecha'].isna().sum()
        if nulldates:
            self.logger.warning(f"⚠️ {nulldates} fechas no pudieron convertirse y serán descartadas")
        # Eliminar filas sin fecha válida
        self.df = self.df.dropna(subset=['fecha'])
        # Normalizar a fecha (sin hora)
        self.df['fecha'] = self.df['fecha'].dt.normalize()
        self.logger.info(f"✅ Fechas convertidas. Rango: {self.df['fecha'].min().date()} a {self.df['fecha'].max().date()}")

    def process(self, file_path):
        """
        Ejecuta lectura y conversión de datos.
        """
        self.read_file(file_path)
        self.convert_dates()
        # Renombrar columnas a nombres más amigables
        self.df = self.df.rename(columns={'descripción': 'descripcion', 'valor_remesa': 'valor_remesa'})
        # Ordenar por fecha
        self.df = self.df.sort_values('fecha').reset_index(drop=True)
        return self.df

    def generate_daily_index(self):
        """
        Genera un índice diario desde la fecha mínima a la máxima.
        """
        if self.df is None:
            self.logger.error("❌ DataFrame no procesado aún")
            return
        start = self.df['fecha'].min()
        end = self.df['fecha'].max()
        self.daily_index = pd.DataFrame({
            'fecha': pd.date_range(start=start, end=end, freq='D')
        })
        self.logger.info(f"📅 Índice diario generado: {len(self.daily_index)} días desde {start.date()} hasta {end.date()}")
        return self.daily_index

    def create_daily_series(self):
        """
        Fusiona los datos originales con el índice diario y hace forward-fill de valores.
        """
        if self.daily_index is None:
            self.generate_daily_index()
        self.logger.info("🔗 Combinando datos y aplicando forward-fill...")
        combined = pd.merge_asof(
            self.daily_index,
            self.df[['fecha', 'valor_remesa']],
            on='fecha',
            direction='backward'
        )
        combined['valor_remesa'] = combined['valor_remesa'].ffill()
        self.final_df = combined
        self.logger.info(f"✅ Series diarias creadas: {len(self.final_df)} registros")
        return self.final_df

    def save_results(self, output_file='remesas_diarias.xlsx'):
        """
        Guarda el DataFrame final en Excel.
        """
        if self.final_df is None:
            self.logger.error("❌ No hay datos diarios generados para guardar")
            return False
        self.logger.info(f"💾 Guardando resultados en: {output_file}")
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            self.final_df.to_excel(writer, sheet_name='DatosDiarios', index=False)
        self.logger.info("✅ Archivo guardado exitosamente")
        return True

    def find_remesas_file(self, search_paths=None):
        """
        Busca automáticamente un archivo de remesas en los directorios indicados.
        """
        if search_paths is None:
            search_paths = ['.', 'data', 'data/0_raw']
        patterns = ['*remesas*.xlsx', '*remesas*.xls']
        for root in search_paths:
            for dirpath, _, files in os.walk(root):
                for f in files:
                    for pat in patterns:
                        if fnmatch.fnmatch(f.lower(), pat.lower()):
                            full = os.path.join(dirpath, f)
                            self.logger.info(f"✅ Archivo de remesas hallado: {full}")
                            return full
        self.logger.error("❌ No se encontró ningún archivo de remesas automáticamente")
        return None


    def run(self, file_path, output_file='remesas_diarias.xlsx'):
        """
        Ejecuta todo el proceso: lectura, conversión, fusión diaria y guardado.
        """
        self.logger.info("🚀 Iniciando procesamiento de remesas...")

        if file_path is None:
            file_path = self.find_remesas_file()
            if file_path is None:
                 return False
        try:
            self.process(file_path)
            self.generate_daily_index()
            self.create_daily_series()
            success = self.save_results(output_file)
            self.logger.info("🎯 Procesamiento completado" if success else "❌ Error en guardado final")
            return success
        except Exception as e:
            self.logger.error(f"💥 Error en procesamiento: {e}")
            return False


import fnmatch

class IEDProcessor:
    """
    Procesa el Excel de IED trimestral y luego genera una serie diaria,
    rellenando hacia adelante (forward-fill) los valores faltantes.
    """
    quarter_map = {
        'Q1': '-01-01',
        'Q2': '-04-01',
        'Q3': '-07-01',
        'Q4': '-10-01',
    }

    def __init__(self, search_paths=None, log_file='logs/ied_processor.log'):
        self.search_paths = search_paths or ['data/0_raw', 'data/0_raw/bond', '.']
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('IEDProcessor')
        self.quarterly_df = None
        self.daily_index = None
        self.final_df = None

    def find_ied_file(self):
        patterns = ['*IED*.xls*', '*Inversion_Extranjera*.xls*']
        for root in self.search_paths:
            for dp, _, files in os.walk(root):
                for f in files:
                    for pat in patterns:
                        if fnmatch.fnmatch(f, pat):
                            full = os.path.join(dp, f)
                            self.logger.info(f"✅ IED hallado: {full}")
                            return full
        self.logger.error("❌ No se encontró archivo IED automáticamente")
        return None

    def read_and_normalize(self, file_path):
        """Lee el Excel, extrae fechas trimestrales y el valor de la última columna."""
        self.logger.info(f"📥 Leyendo IED: {file_path}")

        # Leemos con doble header para capturar país y rótulos
        df2 = pd.read_excel(file_path, header=[2,3], dtype=object)

        # Año en la primera columna (puede venir NaN en filas de trimestre)
        years = pd.to_datetime(df2.iloc[:, 0], errors='coerce').dt.year

        # Trimestre en la segunda columna
        quarters = df2.iloc[:, 1].astype(str).str.strip().str.upper()

        # Valor en la última columna
        values = pd.to_numeric(df2.iloc[:, -1], errors='coerce')

        temp = pd.DataFrame({
            'year': years,
            'quarter': quarters,
            'valor_ied': values
        })

        # Filtrar solo Q1–Q4 y rellenar años faltantes
        temp = temp[temp['quarter'].isin(['Q1','Q2','Q3','Q4'])].copy()
        temp['year'] = temp['year'].ffill().astype(int)

        # Construir fecha para el primer día de cada trimestre
        temp['fecha'] = pd.to_datetime(
            temp['year'].astype(str) +
            temp['quarter'].map(self.quarter_map)
        )

        self.quarterly_df = temp[['fecha', 'valor_ied']].sort_values('fecha').reset_index(drop=True)
        self.logger.info(f"✅ IED trimestral normalizado: {len(self.quarterly_df)} registros")

    def generate_daily_index(self):
        """Genera un índice diario entre la primera y última fecha trimestral."""
        if self.quarterly_df is None:
            self.logger.error("❌ No hay datos trimestrales para generar índice")
            return
        start = self.quarterly_df['fecha'].min()
        end   = self.quarterly_df['fecha'].max()
        self.daily_index = pd.DataFrame({'fecha': pd.date_range(start, end, freq='D')})
        self.logger.info(f"📅 Índice diario generado: {len(self.daily_index)} días ({start.date()} → {end.date()})")

    def create_daily_series(self):
        """Combina índice diario con trimestral usando merge_asof + forward-fill."""
        if self.daily_index is None:
            self.generate_daily_index()
        self.logger.info("🔗 Combinando trimestral con diario y forward-fill...")
        df_daily = pd.merge_asof(
            self.daily_index,
            self.quarterly_df,
            on='fecha',
            direction='backward'
        )
        df_daily['valor_ied'] = df_daily['valor_ied'].ffill()
        self.final_df = df_daily
        self.logger.info(f"✅ Serie diaria creada: {len(self.final_df)} registros")

    def save(self, output_file='ied_diario.xlsx'):
        """Guarda la serie diaria en Excel."""
        if self.final_df is None:
            self.logger.error("❌ No hay serie diaria para guardar")
            return False
        self.final_df.to_excel(output_file, index=False)
        self.logger.info(f"💾 Guardado IED diario en: {output_file}")
        return True

    def run(self, file_path=None, output_file='ied_diario.xlsx'):
        """Ejecuta todo el flujo: detecta archivo, normaliza, crea serie diaria y guarda."""
        self.logger.info("🚀 Iniciando procesamiento de IED (diario)...")
        fp = file_path or self.find_ied_file()
        if not fp:
            return False
        try:
            self.read_and_normalize(fp)
            self.generate_daily_index()
            self.create_daily_series()
            return self.save(output_file)
        except Exception as e:
            self.logger.error(f"❌ Error en IEDProcessor: {e}")
            return False
# processor.run('IngresosEgresos de remesas de trabajadores en Colombia.xlsx')

# 🧩 Clase con logger incorporado
class EOEUniversalProcessor:
    """
    🌟 PROCESADOR UNIVERSAL EOE - Todos los Índices de Confianza
    
    ✅ DETECTA AUTOMÁTICAMENTE: ICI, ICCO, ICC, IEC, ICE y cualquier índice similar
    ✅ MANEJA MÚLTIPLES VARIABLES: Archivos con 2, 3, 4+ columnas de datos
    ✅ TÉRMINOS UNIVERSALES: "confianza", "expectativa", "opinión empresarial"  
    ✅ PATRONES FLEXIBLES: IC*, I*CO, cualquier sigla de 3-4 letras
    ✅ MÚLTIPLES FORMATOS: Maneja diferentes layouts y estructuras
    ✅ COMPATIBLE TOTAL: Funciona con toda la familia EOE
    """
    
        # Redefinir stdout en UTF-8 para consola


        # Configurar logging manualmente para usar UTF-8


    def __init__(self, data_root='data/0_raw', log_file='logs/eoe_universal_processor.log'):
        self.data_root = data_root
        self.logger = configurar_logging(log_file)
        self.global_min_date = None
        self.global_max_date = None
        self.daily_index = None
        self.processed_data = {}
        self.final_df = None
        self.stats = {}
        self.discovered_files = []
        
        # Mapeo de meses en español a números
        self.meses_map = {
            'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12
        }
        
        # Patrones de detección universales
        self.terminos_confianza = [
            'confianza', 'expectativa', 'opinión empresarial', 'encuesta',
            'ici', 'icco', 'icc', 'iec', 'ice', 'eoe', 'consumidor', 'industrial', 'comercial'
        ]
        
        self.patrones_siglas = [
            r'\bici\b', r'\bicco\b', r'\bicc\b', r'\biec\b', r'\bice\b',
            r'\bi[a-z]{1,3}[co]?\b'  # Patrón flexible para índices
        ]
        
        # Mapeo específico de variables conocidas
        self.variable_mapping = {
            'icc': {'name': 'Indice_Confianza_Consumidor', 'sigla': 'ICC', 'type': 'consumer_confidence'},
            'iec': {'name': 'Indice_Expectativas_Consumidores', 'sigla': 'IEC', 'type': 'consumer_confidence'},
            'ice': {'name': 'Indice_Condiciones_Economicas', 'sigla': 'ICE', 'type': 'consumer_confidence'},
            'ici': {'name': 'Indice_Confianza_Industrial', 'sigla': 'ICI', 'type': 'business_confidence'},  
            'icco': {'name': 'Indice_Confianza_Comercial', 'sigla': 'ICCO', 'type': 'business_confidence'}
        }
        
        self.logger.info("=" * 80)
        self.logger.info("🌟 EOE UNIVERSAL PROCESSOR - Todos los Índices de Confianza")
        self.logger.info(f"Directorio de datos: {data_root}")
        self.logger.info(f"Términos detectados: {', '.join(self.terminos_confianza)}")
        self.logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)

    def detect_eoe_universal_file(self, file_path):
        """🔍 DETECCIÓN UNIVERSAL: ¿Es cualquier archivo EOE?"""
        try:
            df = pd.read_excel(file_path, sheet_name=0, header=None, nrows=20)
            indicators = 0
            
            # 1. Buscar términos de confianza UNIVERSALES en las primeras 5 filas
            for i in range(min(5, len(df))):
                for j in range(min(df.shape[1], 5)):  # Revisar más columnas
                    if i < len(df) and j < df.shape[1]:
                        cell_value = str(df.iloc[i, j]).lower()
                        
                        # Buscar cualquier término de confianza
                        for termino in self.terminos_confianza:
                            if termino in cell_value:
                                indicators += 2
                                self.logger.debug(f"Encontrado '{termino}' en celda [{i},{j}]")
                                break
            
            # 2. Buscar siglas de índices (ICI, ICCO, ICC, etc.)
            siglas_encontradas = []
            for i in range(min(5, len(df))):
                for j in range(min(df.shape[1], 5)):
                    if i < len(df) and j < df.shape[1]:
                        cell_value = str(df.iloc[i, j]).lower()
                        
                        for patron in self.patrones_siglas:
                            matches = re.findall(patron, cell_value)
                            if matches:
                                siglas_encontradas.extend(matches)
                                indicators += 1
            
            if siglas_encontradas:
                indicators += 3  # Bonus por encontrar siglas
                self.logger.debug(f"Siglas encontradas: {set(siglas_encontradas)}")
            
            # 3. Verificar patrón de fechas mes-año (CRÍTICO)
            fechas_validas = 0
            for i in range(2, min(25, len(df))):  # Revisar más filas
                if i < len(df):
                    cell_value = str(df.iloc[i, 0]).lower().strip()
                    if re.match(r'^[a-z]{3}-\d{2}$', cell_value):
                        fechas_validas += 1
            
            if fechas_validas >= 3:
                indicators += 5  # Patrón de fechas es CRÍTICO
                self.logger.debug(f"Patrón de fechas válido: {fechas_validas} fechas encontradas")
            
            # 4. Verificar valores numéricos en columnas de datos
            valores_numericos = 0
            for i in range(2, min(15, len(df))):
                if i < len(df):
                    for j in range(1, min(df.shape[1], 5)):  # Revisar múltiples columnas
                        try:
                            val = df.iloc[i, j]
                            if pd.notna(val) and isinstance(val, (int, float)):
                                valores_numericos += 1
                        except:
                            pass
            
            if valores_numericos >= 5:
                indicators += 3
                self.logger.debug(f"Valores numéricos válidos: {valores_numericos}")
            
            # 5. Bonus por estructura típica EOE
            for i in range(min(3, len(df))):
                cell_content = str(df.iloc[i, 0]).lower()
                if 'eoe' in cell_content or 'encuesta' in cell_content or 'confianza' in cell_content:
                    indicators += 3
                    self.logger.debug("Estructura típica EOE detectada")
                    break
            
            self.logger.debug(f"Archivo {Path(file_path).name}: {indicators} indicadores EOE Universal")
            return indicators >= 10  # Umbral ajustado para mayor precisión
            
        except Exception as e:
            self.logger.warning(f"Error detectando archivo EOE Universal {file_path}: {e}")
            return False

    def extract_multiple_variables_info(self, file_path):
        """📊 EXTRACCIÓN INTELIGENTE de múltiples variables de un archivo"""
        try:
            df = pd.read_excel(file_path, sheet_name=0, header=None, nrows=10)
            variables_found = []
            
            # Buscar fila con headers de variables (típicamente fila 1 o 2)
            header_row = None
            for i in range(min(5, len(df))):
                row_content = [str(cell).lower() for cell in df.iloc[i].values if pd.notna(cell)]
                row_text = ' '.join(row_content)
                
                # Contar cuántas siglas conocidas hay en esta fila
                siglas_count = 0
                for sigla in ['icc', 'iec', 'ice', 'ici', 'icco']:
                    if sigla in row_text:
                        siglas_count += 1
                
                if siglas_count >= 2:  # Si hay 2+ siglas, probablemente es el header
                    header_row = i
                    break
            
            if header_row is not None:
                self.logger.info(f"Header de variables encontrado en fila {header_row}")
                
                # Extraer variables de la fila header
                for j in range(1, min(df.shape[1], 6)):  # Columnas 1-5 (después de fechas)
                    if j < df.shape[1]:
                        cell_value = str(df.iloc[header_row, j]).lower()
                        
                        # Identificar variable específica
                        variable_info = None
                        for sigla_key, info in self.variable_mapping.items():
                            if sigla_key in cell_value or info['sigla'].lower() in cell_value:
                                variable_info = {
                                    'column_index': j,
                                    'name': info['name'],
                                    'sigla': info['sigla'],
                                    'type': info['type'],
                                    'original_header': str(df.iloc[header_row, j])
                                }
                                variables_found.append(variable_info)
                                break
                        
                        # Si no se encontró específicamente, crear variable genérica
                        if variable_info is None and cell_value.strip() and 'índice' in cell_value:
                            variable_info = {
                                'column_index': j,
                                'name': f'Indice_Confianza_Col{j}',
                                'sigla': f'IC{j}',
                                'type': 'business_confidence',
                                'original_header': str(df.iloc[header_row, j])
                            }
                            variables_found.append(variable_info)
            
            if not variables_found:
                # Fallback: asumir estructura estándar
                self.logger.warning("No se encontraron headers específicos, usando estructura por defecto")
                for j in range(1, min(df.shape[1], 4)):
                    variables_found.append({
                        'column_index': j,
                        'name': f'Indice_Confianza_Col{j}',
                        'sigla': f'IC{j}',
                        'type': 'business_confidence',
                        'original_header': f'Columna {j}'
                    })
            
            self.logger.info(f"Variables encontradas: {len(variables_found)}")
            for var in variables_found:
                self.logger.info(f"  - {var['sigla']}: {var['name']} (col {var['column_index']})")
            
            return variables_found, header_row
            
        except Exception as e:
            self.logger.warning(f"Error extrayendo info de variables múltiples: {e}")
            return [], None

    def auto_discover_files(self):
        """🔍 DESCUBRIMIENTO AUTOMÁTICO de TODOS los archivos EOE"""
        self.logger.info("🔍 Buscando TODOS los archivos EOE (Universal)...")
        
        discovered = []
        
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if file.endswith(('.xlsx', '.xls')) and not file.startswith('~'):
                    file_path = os.path.join(root, file)
                    
                    if self.detect_eoe_universal_file(file_path):
                        variable_name = Path(file).stem
                        variables_info, header_row = self.extract_multiple_variables_info(file_path)
                        
                        # Si el archivo tiene múltiples variables, crear entrada para cada una
                        if len(variables_info) > 1:
                            for var_info in variables_info:
                                discovered.append({
                                    'Variable': f"{variable_name}_{var_info['sigla']}",
                                    'Archivo': file_path,
                                    'TARGET': var_info['name'],
                                    'Sigla': var_info['sigla'],
                                    'Tipo_Macro': var_info['type'],
                                    'Carpeta': os.path.basename(os.path.dirname(file_path)),
                                    'Column_Index': var_info['column_index'],
                                    'Header_Row': header_row,
                                    'Original_Header': var_info['original_header'],
                                    'Multi_Variable': True
                                })
                        else:
                            # Archivo con una sola variable (comportamiento original)
                            if variables_info:
                                var_info = variables_info[0]
                                discovered.append({
                                    'Variable': variable_name,
                                    'Archivo': file_path,
                                    'TARGET': var_info['name'],
                                    'Sigla': var_info['sigla'],
                                    'Tipo_Macro': var_info['type'],
                                    'Carpeta': os.path.basename(os.path.dirname(file_path)),
                                    'Column_Index': var_info['column_index'],
                                    'Header_Row': header_row,
                                    'Original_Header': var_info['original_header'],
                                    'Multi_Variable': False
                                })
        
        self.discovered_files = discovered
        self.logger.info(f"✅ Encontrados {len(discovered)} variables en archivos EOE:")
        
        # Agrupar por archivo para mostrar mejor
        files_summary = {}
        for file_info in discovered:
            archivo = file_info['Archivo']
            if archivo not in files_summary:
                files_summary[archivo] = []
            files_summary[archivo].append(file_info)
        
        for archivo, variables in files_summary.items():
            file_name = Path(archivo).name
            self.logger.info(f"   📁 {file_name}:")
            for var in variables:
                self.logger.info(f"      → {var['Sigla']}: {var['TARGET']} ({var['Tipo_Macro']})")
        
        return discovered

    def convert_eoe_date(self, date_str):
        """📅 Conversión de fechas EOE: "ene-80" → 1980-01-01"""
        if pd.isna(date_str) or not isinstance(date_str, str):
            return None
        
        date_str = date_str.lower().strip()
        
        # Patrón: mes-año (eje: ene-80, feb-25)
        match = re.match(r'^([a-z]{3})-(\d{2})$', date_str)
        if not match:
            return None
        
        mes_str, año_str = match.groups()
        
        if mes_str not in self.meses_map:
            return None
        
        mes = self.meses_map[mes_str]
        año_2d = int(año_str)
        
        # Convertir año de 2 dígitos a 4 dígitos
        # Regla: 80-99 → 1980-1999, 00-79 → 2000-2079
        if año_2d >= 80:
            año = 1900 + año_2d
        else:
            año = 2000 + año_2d
        
        try:
            return pd.Timestamp(year=año, month=mes, day=1)
        except Exception:
            return None

    def convert_eoe_number(self, value):
        """💰 Conversión de números EOE (maneja decimales y negativos)"""
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            # Validar rango lógico para índices de confianza
            if -200 <= float(value) <= 200:  # Rango razonable para índices
                return float(value)
            else:
                self.logger.warning(f"Valor fuera de rango esperado: {value}")
                return float(value)  # Incluir de todas formas pero advertir
        if not isinstance(value, str):
            return None
            
        value = str(value).strip()
        if value in ['-', '', 'N/A', 'n/a', 'nan']:
            return None
            
        try:
            num_value = float(value)
            if -200 <= num_value <= 200:
                return num_value
            else:
                self.logger.warning(f"Valor fuera de rango esperado: {num_value}")
                return num_value
        except (ValueError, TypeError):
            return None

    def process_file_automatically(self, file_info):
        """⚡ Procesamiento automático de archivo EOE Universal"""
        variable = file_info['Variable']
        target_col = file_info['TARGET']
        sigla = file_info['Sigla']
        macro_type = file_info['Tipo_Macro']
        file_path = file_info['Archivo']
        column_index = file_info['Column_Index']
        header_row = file_info.get('Header_Row', 1)
        
        self.logger.info(f"\n🌟 Procesando EOE Universal: {variable}")
        self.logger.info(f"   📁 Carpeta: {file_info['Carpeta']}")
        self.logger.info(f"   🎯 TARGET: {target_col}")
        self.logger.info(f"   🏷️  Sigla: {sigla}")
        self.logger.info(f"   📂 Tipo: {macro_type}")
        self.logger.info(f"   📊 Columna: {column_index}")
        
        try:
            # Leer Excel completo
            df = pd.read_excel(file_path, sheet_name=0, header=None)
            self.logger.info(f"   📊 Archivo cargado: {len(df)} filas total")
            
            # Determinar fila donde empiezan los datos (después del header)
            data_start_row = header_row + 1 if header_row is not None else 2
            
            # Filtrar solo filas con datos válidos
            valid_rows = []
            
            for idx in range(data_start_row, len(df)):
                row = df.iloc[idx]
                
                # Verificar si la primera columna tiene formato fecha mes-año
                fecha_cell = str(row.iloc[0]).lower().strip()
                if re.match(r'^[a-z]{3}-\d{2}$', fecha_cell):
                    # Verificar que hay un valor numérico en la columna específica
                    if len(row) > column_index and pd.notna(row.iloc[column_index]):
                        valid_rows.append({
                            'fecha_str': fecha_cell,
                            'valor': row.iloc[column_index]
                        })
            
            if len(valid_rows) == 0:
                self.logger.error(f"❌ No se encontraron datos válidos en {variable}")
                return variable, None
            
            self.logger.info(f"   ✅ {len(valid_rows)} registros válidos encontrados")
            
            # Crear DataFrame resultado
            result_df = pd.DataFrame()
            
            # Convertir fechas y valores
            fechas_convertidas = []
            valores_convertidos = []
            
            for row_data in valid_rows:
                fecha_conv = self.convert_eoe_date(row_data['fecha_str'])
                valor_conv = self.convert_eoe_number(row_data['valor'])
                
                if fecha_conv is not None and valor_conv is not None:
                    fechas_convertidas.append(fecha_conv)
                    valores_convertidos.append(valor_conv)
            
            if len(fechas_convertidas) == 0:
                self.logger.error(f"❌ No se pudieron convertir datos válidos en {variable}")
                return variable, None
            
            result_df['fecha'] = fechas_convertidas
            
            # Usar nombre original de la columna del Excel
            nombre_original = file_info.get('Original_Header', sigla)
            # Limpiar el nombre pero mantener la esencia original
            nuevo_nombre = str(nombre_original).strip()
            # Solo limpiar caracteres problemáticos pero mantener el texto original
            nuevo_nombre = re.sub(r'[\n\r\t]', ' ', nuevo_nombre)  # Reemplazar saltos de línea
            nuevo_nombre = re.sub(r'\s+', ' ', nuevo_nombre)  # Normalizar espacios
            nuevo_nombre = nuevo_nombre.strip()
            
            # Si el nombre está vacío o es muy genérico, usar la sigla
            if not nuevo_nombre or nuevo_nombre.lower() in ['', 'nan', 'none']:
                nuevo_nombre = sigla
                
            result_df[nuevo_nombre] = valores_convertidos
            
            # Ordenar por fecha
            result_df = result_df.sort_values('fecha').reset_index(drop=True)
            
            # Estadísticas básicas
            valores_stats = result_df[nuevo_nombre]
            self.logger.info(f"   📈 Estadísticas: Min={valores_stats.min():.1f}, "
                          f"Max={valores_stats.max():.1f}, "
                          f"Media={valores_stats.mean():.1f}")
            
            # Actualizar fechas globales
            current_min = result_df['fecha'].min()
            current_max = result_df['fecha'].max()
            
            if self.global_min_date is None or current_min < self.global_min_date:
                self.global_min_date = current_min
            if self.global_max_date is None or current_max > self.global_max_date:
                self.global_max_date = current_max
            
            # Estadísticas (formato compatible pero con nombre original)
            self.stats[variable] = {
                'macro_type': macro_type,
                'target_column': target_col,
                'sigla': sigla,
                'total_rows': len(result_df),
                'valid_values': result_df[nuevo_nombre].notna().sum(),
                'coverage': 100.0,
                'date_min': current_min,
                'date_max': current_max,
                'nombre_columna': nuevo_nombre,  # Nombre original para referencia
                'column_index': column_index,
                'original_header': file_info.get('Original_Header', ''),
                'min_value': valores_stats.min(),
                'max_value': valores_stats.max(),
                'mean_value': valores_stats.mean()
            }
            
            self.logger.info(f"   ✅ {len(result_df)} valores procesados correctamente")
            self.logger.info(f"   📅 Período: {current_min.strftime('%Y-%m-%d')} a {current_max.strftime('%Y-%m-%d')}")
            
            return variable, result_df
            
        except Exception as e:
            self.logger.error(f"❌ Error procesando {variable}: {str(e)}")
            import traceback
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            return variable, None

    def generate_daily_index(self):
        """📅 Generar índice diario (compatible con sistema principal)"""
        if self.global_min_date is None or self.global_max_date is None:
            self.logger.error("❌ No se pudieron determinar fechas globales")
            return None
            
        self.daily_index = pd.DataFrame({
            'fecha': pd.date_range(start=self.global_min_date, end=self.global_max_date, freq='D')
        })
        
        self.logger.info(f"📅 Índice diario generado: {len(self.daily_index)} días")
        self.logger.info(f"   📅 Desde: {self.global_min_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"   📅 Hasta: {self.global_max_date.strftime('%Y-%m-%d')}")
        return self.daily_index

    def combine_data(self):
        """🔗 Combinar datos con merge_asof (compatible con sistema principal)"""
        if self.daily_index is None:
            self.logger.error("❌ El índice diario no ha sido generado")
            return None
            
        combined = self.daily_index.copy()
        
        for variable, df in self.processed_data.items():
            if df is None or df.empty:
                self.logger.warning(f"⚠️ Omitiendo {variable} por falta de datos")
                continue
                
            df = df.sort_values('fecha')
            col_name = self.stats[variable]['nombre_columna']  # Usar nombre original
            
            # merge_asof para asociar cada día con el valor reportado más reciente
            df_daily = pd.merge_asof(combined[['fecha']], df, on='fecha', direction='backward')
            
            # Forward fill para llenar los días sin datos
            df_daily[col_name] = df_daily[col_name].ffill()
            
            # Agregar la columna al DataFrame combinado
            combined[col_name] = df_daily[col_name]
                                   
        self.final_df = combined
        self.logger.info(f"🔗 Datos combinados: {len(self.final_df)} filas, {len(self.final_df.columns)} columnas")
        
        # Mostrar resumen de columnas
        columnas = [col for col in self.final_df.columns if col != 'fecha']
        self.logger.info(f"📊 Variables en dataset final:")
        for col in columnas:
            # Mostrar nombre original y sigla asociada
            sigla_asociada = "N/A"
            for var, stats in self.stats.items():
                if stats['nombre_columna'] == col:
                    sigla_asociada = stats['sigla']
                    break
            self.logger.info(f"   - {col} ({sigla_asociada})")
        
        return self.final_df

    def analyze_coverage(self):
        """📊 Análisis de cobertura (compatible con sistema principal)"""
        if not self.stats or self.daily_index is None:
            return
            
        total_days = len(self.daily_index)
        self.logger.info("\n📊 Resumen de Cobertura EOE Universal:")
        
        for variable, stats in self.stats.items():
            self.logger.info(f"- {variable} ({stats['sigla']}): {stats['coverage']:.1f}% "
                           f"desde {stats['date_min'].strftime('%Y-%m-%d')} "
                           f"a {stats['date_max'].strftime('%Y-%m-%d')} "
                           f"[{stats['min_value']:.1f} a {stats['max_value']:.1f}]")

    def save_results(self, output_file):
        """💾 Guardar resultados (compatible con sistema principal)"""
        if self.final_df is None:
            self.logger.error("❌ No hay datos combinados para guardar")
            return False
            
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Hoja 1: Datos Diarios
                self.final_df.to_excel(writer, sheet_name='Datos_Diarios', index=False)
                
                # Hoja 2: Estadísticas detalladas
                df_stats = pd.DataFrame(self.stats).T
                df_stats.to_excel(writer, sheet_name='Estadisticas')
                
                # Hoja 3: Resumen por archivo
                archivos_summary = {}
                for file_info in self.discovered_files:
                    archivo = Path(file_info['Archivo']).name
                    if archivo not in archivos_summary:
                        archivos_summary[archivo] = []
                    archivos_summary[archivo].append(file_info)
                
                summary_data = []
                for archivo, variables in archivos_summary.items():
                    for var in variables:
                        summary_data.append({
                            'Archivo': archivo,
                            'Variable': var['Variable'],
                            'Sigla': var['Sigla'],
                            'Tipo': var['Tipo_Macro'],
                            'Columna': var['Column_Index'],
                            'Header_Original': var.get('Original_Header', ''),
                            'Multi_Variable': var.get('Multi_Variable', False)
                        })
                
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Resumen_Archivos', index=False)
                
                # Hoja 4: Metadatos del proceso
                meta = {
                    'Proceso': ['EOEUniversalProcessor_v2'],
                    'Fecha_Proceso': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Total_Variables': [len(self.stats)],
                    'Indices_Detectados': [', '.join([stats['sigla'] for stats in self.stats.values()])],
                    'Periodo_Global': [f"{self.global_min_date.strftime('%Y-%m-%d')} a {self.global_max_date.strftime('%Y-%m-%d')}"],
                    'Total_Dias': [len(self.daily_index)],
                    'Archivos_Procesados': [len(set([info['Archivo'] for info in self.discovered_files]))],
                    'Variables_Multi_Columna': [sum(1 for info in self.discovered_files if info.get('Multi_Variable', False))]
                }
                pd.DataFrame(meta).to_excel(writer, sheet_name='Metadatos', index=False)
                
            self.logger.info(f"💾 Archivo guardado exitosamente: {output_file}")
            self.logger.info(f"📊 Hojas creadas: Datos_Diarios, Estadisticas, Resumen_Archivos, Metadatos")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error guardando resultados: {str(e)}")
            import traceback
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            return False

    def run(self, output_file):
        """🚀 Ejecutar proceso completo EOE Universal"""
        start_time = time.time()
        
        self.logger.info("🚀 Iniciando proceso EOE Universal...")
        
        # 1. Descubrir archivos EOE
        discovered = self.auto_discover_files()
        if not discovered:
            self.logger.error("❌ No se encontraron archivos EOE Universal")
            return False
        
        # 2. Procesar cada variable identificada
        self.logger.info(f"\n⚡ Procesando {len(discovered)} variables...")
        successful_count = 0
        
        for file_info in discovered:
            var, df_processed = self.process_file_automatically(file_info)
            self.processed_data[var] = df_processed
            if df_processed is not None:
                successful_count += 1
        
        # 3. Verificar que se procesó al menos una variable
        if successful_count == 0:
            self.logger.error("❌ No se procesó ninguna variable correctamente")
            return False
        
        self.logger.info(f"✅ {successful_count}/{len(discovered)} variables procesadas exitosamente")
        
        # 4. Generar índice diario
        self.logger.info("\n📅 Generando índice diario...")
        self.generate_daily_index()
        
        # 5. Combinar datos
        self.logger.info("\n🔗 Combinando datos...")
        self.combine_data()
        
        # 6. Analizar cobertura
        self.analyze_coverage()
        
        # 7. Guardar resultados
        self.logger.info(f"\n💾 Guardando resultados en {output_file}...")
        result = self.save_results(output_file)
        
        end_time = time.time()
        
        # Resumen final
        self.logger.info(f"\n" + "="*80)
        self.logger.info(f"⏱️ RESUMEN DE EJECUCIÓN EOE UNIVERSAL")
        self.logger.info(f"="*80)
        self.logger.info(f"⏰ Tiempo total: {end_time - start_time:.2f} segundos")
        self.logger.info(f"📁 Archivos únicos procesados: {len(set([info['Archivo'] for info in self.discovered_files]))}")
        self.logger.info(f"📊 Variables procesadas: {successful_count}")
        self.logger.info(f"📅 Período de datos: {self.global_min_date.strftime('%Y-%m-%d')} a {self.global_max_date.strftime('%Y-%m-%d')}")
        
        if self.stats:
            self.logger.info(f"🏷️  Índices detectados:")
            for var, stats in self.stats.items():
                self.logger.info(f"   - {stats['sigla']}: '{stats['nombre_columna']}' ({stats['total_rows']} registros)")
        
        self.logger.info(f"💾 Archivo de salida: {output_file}")
        self.logger.info(f"🎯 Estado: {'✅ COMPLETADO EXITOSAMENTE' if result else '❌ ERROR EN GUARDADO'}")
        self.logger.info(f"="*80)
        
        return result
        

class TESProcessor:
    """
    🏦 PROCESADOR ESPECÍFICO para Tasas Cero Cupón TES
    
    Diseñado específicamente para el archivo: "Tasas Cero Cupón TES.csv"
    Extrae solo las 3 variables solicitadas:
    - TES pesos 1 año
    - TES pesos 5 años  
    - TES pesos 10 años
    """
    
    def __init__(self, log_file='logs/tes_processor.log'):
        self.logger = self.configurar_logging(log_file)
        self.global_min_date = None
        self.global_max_date = None
        self.daily_index = None
        self.processed_data = None
        self.final_df = None
        self.stats = {}
        
        self.logger.info("=" * 80)
        self.logger.info("🏦 TES PROCESSOR - Tasas Cero Cupón Específico")
        self.logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)

    def configurar_logging(self, log_file):
        """Configurar sistema de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('TESProcessor')

    def find_tes_file(self, search_paths=None):
        """
        🔍 Buscar el archivo de TES en múltiples ubicaciones
        """
        if search_paths is None:
            search_paths = [
                'data/0_raw',
                'data/0_raw/bond',
                'data/raw',
                'raw_data',
                '.',
                'downloads'
            ]
        
        file_patterns = [
            'Tasas Cero Cupón TES.csv',
            'tasas_cero_cupon_tes.csv',
            'TES*.csv',
            '*TES*.csv'
        ]
        
        self.logger.info("🔍 Buscando archivo TES...")
        
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
                
            self.logger.info(f"   Buscando en: {search_path}")
            
            # Búsqueda exacta primero
            for pattern in file_patterns:
                files = list(Path(search_path).rglob(pattern))
                if files:
                    found_file = str(files[0])
                    self.logger.info(f"✅ Archivo encontrado: {found_file}")
                    return found_file
        
        self.logger.error("❌ Archivo TES no encontrado")
        self.logger.info("💡 Ubicaciones buscadas:")
        for path in search_paths:
            self.logger.info(f"   - {path}")
        self.logger.info("📝 Patrones buscados:")
        for pattern in file_patterns:
            self.logger.info(f"   - {pattern}")
        
        return None

    def convert_date_tes(self, date_str):
        """
        📅 Convertir fechas del formato TES: 'MMM DD, AAAA'
        Ejemplos: 'Jan 15, 2023', 'Dec 31, 2024'
        """
        if pd.isna(date_str):
            return None
            
        if isinstance(date_str, (pd.Timestamp, datetime)):
            return pd.Timestamp(date_str)
        
        if not isinstance(date_str, str):
            return None
            
        date_str = str(date_str).strip()
        
        try:
            # Formato directo MMM DD, YYYY
            return pd.to_datetime(date_str, format='%b %d, %Y')
        except:
            try:
                # Formato alternativo con mes completo
                return pd.to_datetime(date_str, format='%B %d, %Y')
            except:
                try:
                    # Conversión genérica
                    return pd.to_datetime(date_str)
                except:
                    self.logger.warning(f"⚠️ No se pudo convertir fecha: '{date_str}'")
                    return None

    def convert_number_tes(self, value):
        """
        💰 Convertir números del formato TES
        """
        if pd.isna(value):
            return None
            
        if isinstance(value, (int, float)):
            return float(value)
        
        if not isinstance(value, str):
            return None
            
        value = str(value).strip()
        
        # Valores especiales
        if value.lower() in ['', 'n/a', 'na', '-', 'null']:
            return None
            
        try:
            # Remover caracteres especiales pero mantener punto y coma
            cleaned = re.sub(r'[^\d.,\-]', '', value)
            
            # Convertir formato colombiano si tiene comas
            if ',' in cleaned and '.' in cleaned:
                # Formato: 1.234,56 → 1234.56
                cleaned = cleaned.replace('.', '').replace(',', '.')
            elif ',' in cleaned:
                # Solo comas: 1234,56 → 1234.56
                cleaned = cleaned.replace(',', '.')
            
            return float(cleaned)
            
        except (ValueError, TypeError):
            self.logger.warning(f"⚠️ No se pudo convertir número: '{value}'")
            return None

    def process_tes_file(self, file_path):
        """
        ⚙️ Procesar el archivo TES específico
        """
        self.logger.info(f"\n📊 Procesando archivo TES: {file_path}")
        
        try:
            # Leer el archivo CSV
            df = pd.read_csv(file_path)
            self.logger.info(f"   📁 Archivo cargado: {len(df)} filas, {len(df.columns)} columnas")
            
            # Mostrar columnas encontradas
            self.logger.info("   📋 Columnas encontradas:")
            for i, col in enumerate(df.columns):
                self.logger.info(f"      {i+1}. {col}")
            
            # Crear DataFrame resultado
            result_df = pd.DataFrame()
            
            # PASO 1: Procesar fechas
            date_column = df.columns[0]  # Primera columna es fecha
            self.logger.info(f"   📅 Procesando fechas de columna: '{date_column}'")
            
            result_df['fecha'] = df[date_column].apply(self.convert_date_tes)
            
            # Eliminar filas sin fecha válida
            initial_rows = len(result_df)
            result_df = result_df.dropna(subset=['fecha'])
            valid_dates = len(result_df)
            
            self.logger.info(f"   ✅ Fechas válidas: {valid_dates}/{initial_rows}")
            
            if valid_dates == 0:
                self.logger.error("❌ No se encontraron fechas válidas")
                return None
            
            # PASO 2: Mapear columnas solicitadas
            target_columns = {
                'tes_1_año': {
                    'pattern': r'.*1\s*año.*pesos.*',
                    'exact_match': 'Tasa de interés Cero Cupón, Títulos de Tesorería (TES), pesos - 1 año',
                    'output_name': 'TES_pesos_1año'
                },
                'tes_5_años': {
                    'pattern': r'.*5\s*años.*pesos.*',
                    'exact_match': 'Tasa de interés Cero Cupón, Títulos de Tesorería (TES), pesos - 5 años',
                    'output_name': 'TES_pesos_5años'
                },
                'tes_10_años': {
                    'pattern': r'.*10\s*años.*pesos.*',
                    'exact_match': 'Tasa de interés Cero Cupón, Títulos de Tesorería (TES), pesos - 10 años',
                    'output_name': 'TES_pesos_10años'
                },
                'tes_uvr_1_año': {
                    'pattern': r'.*1\s*año.*uvr.*',
                    'exact_match': 'Tasa de interés Cero Cupón, Títulos de Tesorería (TES), UVR - 1 año',
                    'output_name': 'TES_uvr_1año'
                },
                'tes_uvr_5_años': {
                    'pattern': r'.*5\s*años.*uvr.*',
                    'exact_match': 'Tasa de interés Cero Cupón, Títulos de Tesorería (TES), UVR - 5 años',
                    'output_name': 'TES_uvr_5años'
                },
                'tes_uvr_10_años': {
                    'pattern': r'.*10\s*años.*uvr.*',
                    'exact_match': 'Tasa de interés Cero Cupón, Títulos de Tesorería (TES), UVR - 10 años',
                    'output_name': 'TES_uvr_10años'
                }
            }
            
            # PASO 3: Encontrar y procesar cada columna objetivo
            found_columns = {}
            
            for key, config in target_columns.items():
                column_found = None
                
                # Buscar coincidencia exacta primero
                if config['exact_match'] in df.columns:
                    column_found = config['exact_match']
                    self.logger.info(f"   ✅ Coincidencia exacta: {key} → '{column_found}'")
                else:
                    # Buscar con regex
                    pattern = re.compile(config['pattern'], re.IGNORECASE)
                    for col in df.columns:
                        if pattern.search(col):
                            column_found = col
                            self.logger.info(f"   ✅ Coincidencia regex: {key} → '{column_found}'")
                            break
                
                if column_found:
                    # Procesar valores numéricos
                    values = df[column_found].apply(self.convert_number_tes)
                    result_df[config['output_name']] = values
                    found_columns[key] = column_found
                    
                    # Estadísticas
                    valid_values = values.notna().sum()
                    total_values = len(values)
                    self.logger.info(f"      📊 Valores válidos: {valid_values}/{total_values} ({valid_values/total_values*100:.1f}%)")
                else:
                    self.logger.warning(f"   ⚠️ No encontrada: {key}")
            
            if not found_columns:
                self.logger.error("❌ No se encontraron columnas objetivo")
                return None
            
            # PASO 4: Limpiar datos finales
            # Eliminar filas donde TODAS las columnas TES son NaN
            tes_columns = [config['output_name'] for config in target_columns.values() 
                          if config['output_name'] in result_df.columns]
            
            if tes_columns:
                result_df = result_df.dropna(subset=tes_columns, how='all')
            
            # Ordenar por fecha
            result_df = result_df.sort_values('fecha').reset_index(drop=True)
            
            # PASO 5: Calcular estadísticas
            self.global_min_date = result_df['fecha'].min()
            self.global_max_date = result_df['fecha'].max()
            
            self.stats = {
                'archivo_procesado': file_path,
                'filas_totales': len(result_df),
                'periodo_inicio': self.global_min_date,
                'periodo_fin': self.global_max_date,
                'columnas_encontradas': found_columns,
                'columnas_procesadas': tes_columns
            }
            
            self.logger.info(f"\n✅ PROCESAMIENTO EXITOSO:")
            self.logger.info(f"   📊 Filas procesadas: {len(result_df)}")
            self.logger.info(f"   📅 Período: {self.global_min_date.strftime('%Y-%m-%d')} a {self.global_max_date.strftime('%Y-%m-%d')}")
            self.logger.info(f"   📈 Variables extraídas: {len(tes_columns)}")
  
            return result_df  # <-- Nota: cambié 'result' por 'result_df'
            
        except Exception as e:
            self.logger.error(f"❌ Error procesando archivo: {str(e)}")
            return None

    def generate_daily_index(self):
        """📅 Generar índice diario"""
        if self.global_min_date is None or self.global_max_date is None:
            self.logger.error("❌ No se pueden determinar fechas globales")
            return None
            
        self.daily_index = pd.DataFrame({
            'fecha': pd.date_range(start=self.global_min_date, end=self.global_max_date, freq='D')
        })
        
        self.logger.info(f"📅 Índice diario generado: {len(self.daily_index)} días")
        return self.daily_index

    def create_daily_series(self):
        """🔗 Crear series diarias con forward fill"""
        if self.processed_data is None or self.daily_index is None:
            self.logger.error("❌ Faltan datos procesados o índice diario")
            return None
            
        self.logger.info("🔗 Creando series temporales diarias...")

        self.processed_data = self.processed_data.sort_values('fecha')
        combined = pd.merge_asof(self.daily_index, self.processed_data, on='fecha', direction='backward')

        tes_columns = [col for col in combined.columns if col.startswith('TES_')]
        for col in tes_columns:
            combined[col] = combined[col].ffill()

        self.final_df = combined
        self.logger.info(f"✅ Series diarias creadas: {len(self.final_df)} días")

        return self.final_df

    def save_results(self, output_file='datos_tes_procesados.xlsx'):
        """💾 Guardar resultados procesados"""
        if self.final_df is None:
            self.logger.error("❌ No hay datos para guardar")
            return False

        try:
            self.logger.info(f"💾 Guardando resultados en: {output_file}")

            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                self.final_df.to_excel(writer, sheet_name='TES_Datos_Diarios', index=False)

                stats_df = pd.DataFrame([
                    ['Archivo procesado', self.stats['archivo_procesado']],
                    ['Filas totales', self.stats['filas_totales']],
                    ['Período inicio', self.stats['periodo_inicio'].strftime('%Y-%m-%d')],
                    ['Período fin', self.stats['periodo_fin'].strftime('%Y-%m-%d')],
                    ['Días totales', len(self.daily_index)],
                    ['Variables extraídas', len(self.stats['columnas_procesadas'])]
                ], columns=['Métrica', 'Valor'])
                stats_df.to_excel(writer, sheet_name='Estadisticas', index=False)

                mapping_data = []
                for key, original_name in self.stats['columnas_encontradas'].items():
                    mapping_data.append([key, original_name])
                mapping_df = pd.DataFrame(mapping_data, columns=['Variable_TES', 'Columna_Original'])
                mapping_df.to_excel(writer, sheet_name='Mapeo_Columnas', index=False)

                tes_cols = [col for col in self.final_df.columns if col.startswith('TES_')]
                self.logger.info(f"   📊 Columnas TES guardadas: {tes_cols}")

            self.logger.info(f"✅ Archivo guardado exitosamente: {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error guardando resultados: {str(e)}")
            return False

    def run(self, file_path=None, output_file='datos_tes_procesados.xlsx'):
        """🚀 Ejecutar procesamiento completo"""
        start_time = time.time()

        self.logger.info("🚀 Iniciando procesamiento TES...")

        if file_path is None:
            file_path = self.find_tes_file()
            if file_path is None:
                return False

        self.processed_data = self.process_tes_file(file_path)
        if self.processed_data is None:
            return False

        if self.generate_daily_index() is None:
            return False

        if self.create_daily_series() is None:
            return False

        if not self.save_results(output_file):
            return False

        end_time = time.time()
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🎯 PROCESAMIENTO COMPLETADO")
        self.logger.info("=" * 60)
        self.logger.info(f"⏱️  Tiempo total: {end_time - start_time:.2f} segundos")
        self.logger.info(f"📁 Archivo procesado: {file_path}")
        self.logger.info(f"📊 Variables extraídas: {self.stats['columnas_procesadas']}")
        self.logger.info(f"📅 Período: {self.global_min_date.strftime('%Y-%m-%d')} a {self.global_max_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"💾 Archivo guardado: {output_file}")
        self.logger.info("=" * 60)

        return True
       
        


# =============================================================================
# SECCIÓN 3: FUNCIONES INTEGRADAS PARA EL SISTEMA PRINCIPAL
# =============================================================================

def procesar_tasas_tes(file_path=None, output_file='datos_tes_procesados.xlsx'):
    """
    🏦 Función principal para procesar Tasas Cero Cupón TES
    
    Args:
        file_path (str, optional): Ruta específica al archivo. Si es None, busca automáticamente.
        output_file (str): Nombre del archivo Excel de salida.
    
    Returns:
        bool: True si el procesamiento fue exitoso, False en caso contrario.
    
    Ejemplo de uso:
        # Búsqueda automática
        procesar_tasas_tes()
        
        # Archivo específico
        procesar_tasas_tes('data/Tasas Cero Cupón TES.csv')
        
        # Con archivo de salida personalizado
        procesar_tasas_tes(output_file='mi_tes_data.xlsx')
    """
    processor = TESProcessor()
    return processor.run(file_path, output_file)

def procesar_pib_trimestral(file_path=None, output_file='pib_diario_procesado.xlsx',
                           search_paths=None, log_file='logs/pib_processor.log'):
    """
    🚀 Función principal para procesar el PIB trimestral
    
    Args:
        file_path (str, optional): Ruta específica al archivo PIB
        output_file (str): Archivo Excel de salida
        search_paths (list, optional): Rutas de búsqueda personalizadas
        log_file (str): Archivo de log
    
    Returns:
        bool: True si el procesamiento fue exitoso
        
    Ejemplo de uso:
        # Búsqueda automática
        success = procesar_pib_trimestral()
        
        # Archivo específico
        success = procesar_pib_trimestral('data/anexProduccionConstantesItrim2025.xlsx')
        
        # Personalizado
        success = procesar_pib_trimestral(
            output_file='mi_pib_data.xlsx',
            search_paths=['downloads', 'data/raw']
        )
    """
    processor = PIBProcessor(search_paths, log_file)
    return processor.run(file_path, output_file)

def ejecutar_eoe_universal_processor(
    config_file=None,  # No necesita configuración externa
    output_file=None,
    data_root=None,
    log_file=None
):
    """
    🌟 PROCESADOR EOE UNIVERSAL v2 - Todos los Índices de Confianza
    Función integrada compatible con el sistema principal
    
    ✅ Detecta automáticamente TODOS los archivos EOE
    ✅ Maneja archivos con MÚLTIPLES VARIABLES (ICC, IEC, ICE en uno solo)
    ✅ Convierte fechas "ene-80" → 1980-01-01 automáticamente  
    ✅ Salta filas vacías y headers automáticamente
    ✅ Valida rangos de valores para índices de confianza
    ✅ Genera Excel con estadísticas detalladas
    """
    
    # Configurar rutas por defecto si no se especifican
    if data_root is None:
        data_root = 'data/0_raw'  # Directorio por defecto
    
    if output_file is None:
        output_file = 'datos_eoe_universal_procesados.xlsx'
    
    if log_file is None:
        log_file = 'logs/eoe_universal_processor.log'
    
    # Crear directorios si no existen
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    
    # Crear y ejecutar procesador
    processor = EOEUniversalProcessor(data_root, log_file)
    return processor.run(output_file)


def procesar_archivo_eoc_especifico(archivo_path, output_file=None):
    """
    🎯 FUNCIÓN ESPECÍFICA para procesar tu archivo EOC_Abril_2025_Histórico 1.xlsx
    
    Este archivo contiene 3 variables específicas:
    - ICC: Índice de Confianza del Consumidor
    - IEC: Índice de Expectativas de los Consumidores  
    - ICE: Índice de Condiciones Económicas
    """
    
    if output_file is None:
        output_file = 'EOC_procesado_3_variables.xlsx'
    
    log_file = 'logs/eoc_processor.log'
    
    # Configurar logging
    logger = configurar_logging_eoe(log_file)
    
    logger.info("🎯 PROCESAMIENTO ESPECÍFICO ARCHIVO EOC")
    logger.info(f"📁 Archivo: {archivo_path}")
    logger.info(f"📊 Variables esperadas: ICC, IEC, ICE")
    logger.info("="*60)
    
    try:
        # Crear directorio temporal para simular estructura esperada
        temp_dir = 'temp_data'
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copiar archivo al directorio temporal si es necesario
        import shutil
        if not os.path.exists(os.path.join(temp_dir, os.path.basename(archivo_path))):
            shutil.copy2(archivo_path, temp_dir)
        
        # Crear procesador apuntando al directorio temporal


        processor = EOEUniversalProcessor(temp_dir, log_file)
        
        # Ejecutar procesamiento
        result = processor.run(output_file)
        
        # Limpiar directorio temporal
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        
        if result:
            logger.info("✅ Procesamiento específico completado exitosamente")
            logger.info(f"📊 Archivo generado: {output_file}")
            
            # Mostrar información específica de las 3 variables
            if processor.stats:
                logger.info("\n📈 RESUMEN DE LAS 3 VARIABLES:")
                for var, stats in processor.stats.items():
                    logger.info(f"   {stats['sigla']}: {stats['total_rows']} datos, "
                              f"rango [{stats['min_value']:.1f}, {stats['max_value']:.1f}], "
                              f"promedio {stats['mean_value']:.1f}")
        else:
            logger.error("❌ Error en procesamiento específico")
        
        return result
        
    except Exception as e:
        logger.error(f"💥 Error crítico en procesamiento específico: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


# =============================================================================
# SECCIÓN 4: PROCESADOR INTEGRADO PRINCIPAL
# =============================================================================

class IntegratedProcessor:
    """
    🌟 PROCESADOR INTEGRADO: TES + EOE UNIVERSAL
    Maneja ambos tipos de archivos de forma unificada
    """
    
    def __init__(self, log_file='logs/integrated_processor.log'):
        self.logger = self.configurar_logging(log_file)
        self.results = {
            'tes': None,
            'eoe': None
        }
        
        self.logger.info("=" * 80)
        self.logger.info("🌟 PROCESADOR INTEGRADO: TES + EOE UNIVERSAL")
        self.logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)
    
    def configurar_logging(self, log_file):
        """Configurar sistema de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('IntegratedProcessor')
    
    def procesar_tes(self, file_path=None, output_file='datos_tes_procesados.xlsx'):
        """Procesar archivos TES"""
        self.logger.info("\n🏦 INICIANDO PROCESAMIENTO TES")
        self.logger.info("-" * 60)
        
        result = procesar_tasas_tes(file_path, output_file)
        self.results['tes'] = result
        
        if result:
            self.logger.info("✅ Procesamiento TES completado exitosamente")
        else:
            self.logger.error("❌ Error en procesamiento TES")
        
        return result
    
    def procesar_eoe(self, data_root='data/0_raw', output_file='datos_eoe_universal_procesados.xlsx'):
        """Procesar archivos EOE"""
        self.logger.info("\n🌟 INICIANDO PROCESAMIENTO EOE UNIVERSAL")
        self.logger.info("-" * 60)
        
        result = ejecutar_eoe_universal_processor(
            data_root=data_root,
            output_file=output_file
        )
        self.results['eoe'] = result
        
        if result:
            self.logger.info("✅ Procesamiento EOE completado exitosamente")
        else:
            self.logger.error("❌ Error en procesamiento EOE")
        
        return result
    
    def procesar_todo(self, 
                     tes_file=None, 
                     eoe_data_root='data/0_raw',
                     output_dir='output'):
        """
        Procesar tanto TES como EOE de forma integrada
        
        Args:
            tes_file: Ruta al archivo TES (None para búsqueda automática)
            eoe_data_root: Directorio raíz para buscar archivos EOE
            output_dir: Directorio donde guardar los resultados
        """
        self.logger.info("\n🚀 PROCESAMIENTO INTEGRADO COMPLETO")
        self.logger.info("=" * 80)
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Procesar TES
        tes_output = os.path.join(output_dir, 'datos_tes_procesados.xlsx')
        tes_success = self.procesar_tes(tes_file, tes_output)
        
        # Procesar EOE
        eoe_output = os.path.join(output_dir, 'datos_eoe_universal_procesados.xlsx')
        eoe_success = self.procesar_eoe(eoe_data_root, eoe_output)
        
        # Resumen final
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 RESUMEN DE PROCESAMIENTO INTEGRADO")
        self.logger.info("=" * 80)
        self.logger.info(f"TES: {'✅ EXITOSO' if tes_success else '❌ ERROR'}")
        self.logger.info(f"EOE: {'✅ EXITOSO' if eoe_success else '❌ ERROR'}")
        self.logger.info(f"Directorio de salida: {output_dir}")
        self.logger.info("=" * 80)
        
        return {
            'tes': tes_success,
            'eoe': eoe_success,
            'output_dir': output_dir
        }


# =============================================================================
# SECCIÓN 5: FUNCIONES DE UTILIDAD Y MENÚ PRINCIPAL
# =============================================================================

def mostrar_menu():
    """Mostrar menú interactivo actualizado con Economía Integrada"""
    print("\n" + "=" * 80)
    print("🌟 PROCESADOR INTEGRADO: TES + EOE + PIB + ISE")
    print("=" * 80)
    print("\nOpciones disponibles:")
    print("1. 🏦 Procesar solo TES (Tasas Cero Cupón)")
    print("2. 🌟 Procesar solo EOE (Índices de Confianza)")
    print("3. 🏗️ Procesar solo PIB (Trimestral → Diario)")
    print("4. 📈 Procesar Economía Integrada (PIB + ISE)")  # ← NUEVA OPCIÓN
    print("5. 🚀 Procesar TODO (TES + EOE + PIB + ISE)")
    print("6. 🎯 Procesar archivo EOC específico (ICC, IEC, ICE)")
    print("7. 💸 Procesar Remesas + IED")
    print("8. ❌ Salir")
    print("-" * 80)
    
    while True:
        opcion = input("\nSeleccione una opción (1-8): ").strip()
        if opcion in ['1', '2', '3', '4', '5', '6', '7', '8']:
            return opcion
        else:
            print("⚠️  Opción inválida. Por favor, seleccione 1-8.")
# =============================================================================
# REEMPLAZA LA FUNCIÓN ejecutar_opcion_menu() EXISTENTE CON ESTA VERSIÓN
# =============================================================================

def ejecutar_opcion_menu(opcion):
    """Ejecutar la opción seleccionada del menú"""
    
    if opcion == '1':
        # Procesar solo TES
        print("\n🏦 PROCESAMIENTO TES")
        print("-" * 60)
        
        archivo = input("Ruta al archivo TES (Enter para búsqueda automática): ").strip()
        if not archivo:
            archivo = None
        
        output = input("Archivo de salida (Enter para 'datos_tes_procesados.xlsx'): ").strip()
        if not output:
            output = 'datos_tes_procesados.xlsx'
        
        success = procesar_tasas_tes(archivo, output)
        
        if success:
            print(f"\n✅ Procesamiento TES completado: {output}")
        else:
            print("\n❌ Error en procesamiento TES")
    
    elif opcion == '2':
        # Procesar solo EOE
        print("\n🌟 PROCESAMIENTO EOE UNIVERSAL")
        print("-" * 60)
        
        data_root = input("Directorio de datos (Enter para 'data/0_raw'): ").strip()
        if not data_root:
            data_root = 'data/0_raw'
        
        output = input("Archivo de salida (Enter para 'datos_eoe_universal_procesados.xlsx'): ").strip()
        if not output:
            output = 'datos_eoe_universal_procesados.xlsx'
        
        success = ejecutar_eoe_universal_processor(data_root=data_root, output_file=output)
        
        if success:
            print(f"\n✅ Procesamiento EOE completado: {output}")
        else:
            print("\n❌ Error en procesamiento EOE")
    
    elif opcion == '3':
        # Procesar solo PIB
        print("\n🏗️ PROCESAMIENTO PIB TRIMESTRAL")
        print("-" * 60)
        
        archivo = input("Ruta al archivo PIB (Enter para búsqueda automática): ").strip()
        if not archivo:
            archivo = None
        
        output = input("Archivo de salida (Enter para 'pib_diario_procesado.xlsx'): ").strip()
        if not output:
            output = 'pib_diario_procesado.xlsx'
        
        success = procesar_pib_trimestral(archivo, output)
        
        if success:
            print(f"\n✅ Procesamiento PIB completado: {output}")
            print("📊 Serie diaria generada con forward-fill")
        else:
            print("\n❌ Error en procesamiento PIB")
    
    elif opcion == '4':
        # Procesar Economía Integrada (PIB + ISE) - NUEVA OPCIÓN
        print("\n📈 PROCESAMIENTO ECONOMÍA INTEGRADA (PIB + ISE)")
        print("-" * 60)
        
        pib_file = input("Ruta al archivo PIB (Enter para búsqueda automática): ").strip()
        if not pib_file:
            pib_file = None
        
        ise_file = input("Ruta al archivo ISE (Enter para búsqueda automática): ").strip()
        if not ise_file:
            ise_file = None
        
        output = input("Archivo de salida (Enter para 'economia_integrada.xlsx'): ").strip()
        if not output:
            output = 'economia_integrada.xlsx'
        
        success = procesar_economia_integrada(pib_file, ise_file, output)
        
        if success:
            print(f"\n✅ Procesamiento Economía Integrada completado: {output}")
            print("📊 PIB (trimestral) + ISE (mensual) → Series diarias unificadas")
            print("📁 Hojas: Economia_Diaria, PIB_Trimestral, ISE_Mensual, Estadisticas")
        else:
            print("\n❌ Error en procesamiento de Economía Integrada")
    
    elif opcion == '5':
        # Procesar TODO - ACTUALIZAR PARA INCLUIR ECONOMÍA
        print("\n🚀 PROCESAMIENTO COMPLETO (TES + EOE + ECONOMÍA)")
        print("-" * 60)
        
        processor = IntegratedProcessor()
        
        tes_file = input("Ruta al archivo TES (Enter para búsqueda automática): ").strip()
        if not tes_file:
            tes_file = None
        
        eoe_root = input("Directorio EOE (Enter para 'data/0_raw'): ").strip()
        if not eoe_root:
            eoe_root = 'data/0_raw'
        
        pib_file = input("Ruta al archivo PIB (Enter para búsqueda automática): ").strip()
        if not pib_file:
            pib_file = None
        
        ise_file = input("Ruta al archivo ISE (Enter para búsqueda automática): ").strip()
        if not ise_file:
            ise_file = None
        
        output_dir = input("Directorio de salida (Enter para 'output'): ").strip()
        if not output_dir:
            output_dir = 'output'
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Procesar TES y EOE
        results = processor.procesar_todo(tes_file, eoe_root, output_dir)
        
        # Procesar Economía Integrada
        economia_output = os.path.join(output_dir, 'economia_integrada.xlsx')
        economia_success = procesar_economia_integrada(pib_file, ise_file, economia_output)
        
        # Mostrar resultados
        print(f"\n📊 Resultados guardados en: {output_dir}")
        print(f"TES: {'✅ EXITOSO' if results['tes'] else '❌ ERROR'}")
        print(f"EOE: {'✅ EXITOSO' if results['eoe'] else '❌ ERROR'}")
        print(f"ECONOMÍA (PIB+ISE): {'✅ EXITOSO' if economia_success else '❌ ERROR'}")
        
        if economia_success:
            print("📈 Serie unificada PIB+ISE generada con éxito")
    
    elif opcion == '6':
        # Procesar archivo EOC específico
        print("\n🎯 PROCESAMIENTO ARCHIVO EOC ESPECÍFICO")
        print("-" * 60)
        
        archivo = input("Ruta al archivo EOC: ").strip()
        
        if not archivo or not os.path.exists(archivo):
            print("❌ Archivo no encontrado")
            return
        
        output = input("Archivo de salida (Enter para 'EOC_procesado_3_variables.xlsx'): ").strip()
        if not output:
            output = 'EOC_procesado_3_variables.xlsx'
        
        success = procesar_archivo_eoc_especifico(archivo, output)
        
        if success:
            print(f"\n✅ Procesamiento EOC completado: {output}")
            print("📊 Variables procesadas: ICC, IEC, ICE")
        else:
            print("\n❌ Error en procesamiento EOC")

    elif opcion == '7':
        # Procesar remesas + IED
        print("\n💸 PROCESAMIENTO REMESAS + IED")
        print("-" * 60)

        # remesas (auto-detect)
        proc_r = RemesasProcessor()
        ok_r   = proc_r.run(None, 'remesas_diarias.xlsx')

        # IED (auto-detect)
        proc_i = IEDProcessor()
        ok_i   = proc_i.run(None, 'ied_diario.xlsx')

        if ok_r and ok_i:
            print("\n✅ Ambos procesos completados:")
            print("   • remesas_diarias.xlsx")
            print("   • ied_diario.xlsx")
        else:
            if not ok_r: print("❌ Error en remesas")
            if not ok_i: print("❌ Error en IED")

    elif opcion == '8':
        # Salir
        print("\n👋 ¡Hasta luego!")
        return False  # Señal para salir del bucle principal
    
    return True  # Continuar en el menú
# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    """
    🚀 EJECUTAR PROCESADOR INTEGRADO
    """
    
    import sys

    
    # Verificar argumentos de línea de comandos
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print("\n🌟 PROCESADOR INTEGRADO: TES + EOE UNIVERSAL")
            print("=" * 80)
            print("\nUso:")
            print("  python integrated_processor.py              # Menú interactivo")
            print("  python integrated_processor.py --tes        # Procesar solo TES")
            print("  python integrated_processor.py --eoe        # Procesar solo EOE")
            print("  python integrated_processor.py --all        # Procesar todo")
            print("  python integrated_processor.py --help       # Esta ayuda")
            print("\nEjemplos:")
            print("  python integrated_processor.py --tes archivo.csv")
            print("  python integrated_processor.py --eoe data/0_raw")
            print("  python integrated_processor.py --all")
            
        elif sys.argv[1] == '--tes':
            # Procesamiento directo TES
            file_path = sys.argv[2] if len(sys.argv) > 2 else None
            success = procesar_tasas_tes(file_path)
            sys.exit(0 if success else 1)
            
        elif sys.argv[1] == '--eoe':
            # Procesamiento directo EOE
            data_root = sys.argv[2] if len(sys.argv) > 2 else 'data/0_raw'
            success = ejecutar_eoe_universal_processor(data_root=data_root)
            sys.exit(0 if success else 1)
            
        elif sys.argv[1] == '--all':
            # Procesamiento completo
            processor = IntegratedProcessor()
            results = processor.procesar_todo()
            success = results['tes'] and results['eoe']
            sys.exit(0 if success else 1)

        elif sys.argv[1] == '--remesas':
            # Uso:
            #   python integrated_processor.py --remesas <archivo_excel> [<archivo_salida>]
            file_path = sys.argv[2] if len(sys.argv) > 2 else None
            output_file = sys.argv[3] if len(sys.argv) > 3 else 'remesas_diarias.xlsx'
            if not file_path:
                print("❌ Indica la ruta al Excel de remesas")
                sys.exit(1)
            proc = RemesasProcessor()
            ok = proc.run(file_path, output_file)
            sys.exit(0 if ok else 1)

        else:
            print(f"❌ Opción no reconocida: {sys.argv[1]}")
            print("Use --help para ver las opciones disponibles")
            sys.exit(1)

    else:
        # Modo interactivo con menú
        try:
            while True:
                opcion = mostrar_menu()
                
                if opcion == '6':
                    print("\n👋 ¡Hasta luego!")
                    break
                
                ejecutar_opcion_menu(opcion)
                
                input("\n📌 Presione Enter para continuar...")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Proceso interrumpido por el usuario")
        except Exception as e:
            print(f"\n💥 Error inesperado: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n🎯 Procesamiento finalizado")
        print("=" * 80)
            

    def generate_daily_index(self):
        """📅 Generar índice diario"""
        if self.global_min_date is None or self.global_max_date is None:
            self.logger.error("❌ No se pueden determinar fechas globales")
            return None
            
        self.daily_index = pd.DataFrame({
            'fecha': pd.date_range(start=self.global_min_date, end=self.global_max_date, freq='D')
        })
        
        self.logger.info(f"📅 Índice diario generado: {len(self.daily_index)} días")
        return self.daily_index

    def create_daily_series(self):
        """🔗 Crear series diarias con forward fill"""
        if self.processed_data is None or self.daily_index is None:
            self.logger.error("❌ Faltan datos procesados o índice diario")
            return None
            
        self.logger.info("🔗 Creando series temporales diarias...")
        
        # Merge usando merge_asof para forward fill
        self.processed_data = self.processed_data.sort_values('fecha')
        combined = pd.merge_asof(self.daily_index, self.processed_data, on='fecha', direction='backward')
        
        # Forward fill para rellenar valores faltantes
        tes_columns = [col for col in combined.columns if col.startswith('TES_')]
        for col in tes_columns:
            combined[col] = combined[col].ffill()
        
        self.final_df = combined
        self.logger.info(f"✅ Series diarias creadas: {len(self.final_df)} días")
        
        return self.final_df

    def save_results(self, output_file='datos_tes_procesados.xlsx'):
        """💾 Guardar resultados procesados"""
        if self.final_df is None:
            self.logger.error("❌ No hay datos para guardar")
            return False
            
        try:
            self.logger.info(f"💾 Guardando resultados en: {output_file}")
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Hoja 1: Datos diarios
                self.final_df.to_excel(writer, sheet_name='TES_Datos_Diarios', index=False)
                
                # Hoja 2: Estadísticas
                stats_df = pd.DataFrame([
                    ['Archivo procesado', self.stats['archivo_procesado']],
                    ['Filas totales', self.stats['filas_totales']],
                    ['Período inicio', self.stats['periodo_inicio'].strftime('%Y-%m-%d')],
                    ['Período fin', self.stats['periodo_fin'].strftime('%Y-%m-%d')],
                    ['Días totales', len(self.daily_index)],
                    ['Variables extraídas', len(self.stats['columnas_procesadas'])]
                ], columns=['Métrica', 'Valor'])
                
                stats_df.to_excel(writer, sheet_name='Estadisticas', index=False)
                
                # Hoja 3: Mapeo de columnas
                mapping_data = []
                for key, original_name in self.stats['columnas_encontradas'].items():
                    mapping_data.append([key, original_name])
                
                mapping_df = pd.DataFrame(mapping_data, columns=['Variable_TES', 'Columna_Original'])
                mapping_df.to_excel(writer, sheet_name='Mapeo_Columnas', index=False)
                
                # Show column info
                tes_cols = [col for col in self.final_df.columns if col.startswith('TES_')]
                self.logger.info(f"   📊 Columnas TES guardadas: {tes_cols}")
            
            self.logger.info(f"✅ Archivo guardado exitosamente: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error guardando resultados: {str(e)}")
            return False

    def run(self, file_path=None, output_file='datos_tes_procesados.xlsx'):
        """
        🚀 Ejecutar procesamiento completo
        """
        start_time = time.time()
        
        self.logger.info("🚀 Iniciando procesamiento TES...")
        
        # PASO 1: Encontrar archivo
        if file_path is None:
            file_path = self.find_tes_file()
            if file_path is None:
                return False
        
        # PASO 2: Procesar archivo
        self.processed_data = self.process_tes_file(file_path)
        if self.processed_data is None:
            return False
        
        # PASO 3: Generar índice diario
        if self.generate_daily_index() is None:
            return False
        
        # PASO 4: Crear series diarias
        if self.create_daily_series() is None:
            return False
        
        # PASO 5: Guardar resultados
        if not self.save_results(output_file):
            return False
        
        # Resumen final
        end_time = time.time()
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🎯 PROCESAMIENTO COMPLETADO")
        self.logger.info("=" * 60)
        self.logger.info(f"⏱️  Tiempo total: {end_time - start_time:.2f} segundos")
        self.logger.info(f"📁 Archivo procesado: {file_path}")
        self.logger.info(f"📊 Variables extraídas: {self.stats['columnas_procesadas']}")
        self.logger.info(f"📅 Período: {self.global_min_date.strftime('%Y-%m-%d')} a {self.global_max_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"💾 Archivo guardado: {output_file}")
        self.logger.info("=" * 60)
        
        return True


# =============================================================================
# SECCIÓN 2: EOE UNIVERSAL PROCESSOR - Índices de Confianza
# =============================================================================

def configurar_logging_eoe(log_file='logs/eoe_universal_processor.log'):
    """Configuración del sistema de logging para EOE"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('EOE_Universal_Processor')


