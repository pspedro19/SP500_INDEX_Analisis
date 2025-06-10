import os
import re
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from src.core.utils import configure_logging, convertir_valor, detectar_formato_fecha_inteligente, convertir_fecha_adaptativo

class BancoRepublicaProcessor:
    """
    🤖 PROCESADOR AUTOMÁTICO para archivos del Banco de la República
    INTEGRADO directamente en step_0_preprocess.py
    
    ✅ COMPLETAMENTE AUTOMÁTICO: Detecta y procesa cualquier archivo Excel del Banco República
    ✅ BUSCA EN SUBCARPETAS: Recorre todas las carpetas categorizadas automáticamente
    ✅ FORMATO COLOMBIANO: Maneja dd/mm/aaaa y comas como decimales
    ✅ INTEGRACIÓN TOTAL: Mismo patrón que los otros procesadores del sistema
    """
    
    def __init__(self, data_root='data/0_raw', log_file='logs/banco_republica.log'):
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
        self.logger.info("🤖 BANCO REPÚBLICA PROCESSOR - AUTOMÁTICO (INTEGRADO)")
        self.logger.info(f"Directorio de datos: {data_root}")
        self.logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)

    def detect_banco_republica_file(self, file_path):
        """🔍 DETECCIÓN AUTOMÁTICA: ¿Es un archivo del Banco República?"""
        try:
            df = pd.read_excel(file_path, sheet_name=0, header=None, nrows=10)
            indicators = 0
            
            # 1. Primera columna contiene "Fecha"
            if 'fecha' in str(df.iloc[0, 0]).lower():
                indicators += 2
                
            # 2. Formato típico dd/mm/aaaa
            if 'dd/mm/aaaa' in str(df.iloc[1, 0]):
                indicators += 3
                
            # 3. Datos con formato colombiano (comas)
            sample_value = str(df.iloc[2, 1]) if len(df.columns) > 1 else ""
            if ',' in sample_value and any(c.isdigit() for c in sample_value):
                indicators += 2
                
            # 4. Pie de página "Descargado de sistema del Banco..."
            try:
                bottom_df = pd.read_excel(file_path, sheet_name=0, header=None, skiprows=max(0, len(df)-5))
                for _, row in bottom_df.iterrows():
                    if any('descargado' in str(cell).lower() for cell in row if pd.notna(cell)):
                        indicators += 3
                        break
            except:
                pass
            
            return indicators >= 5
            
        except Exception:
            return False

    def auto_discover_files(self):
        """🔍 DESCUBRIMIENTO AUTOMÁTICO en todas las subcarpetas categorizadas"""
        self.logger.info("🔍 Buscando archivos del Banco de la República en subcarpetas...")
        
        discovered = []
        
        # Buscar recursivamente en TODAS las subcarpetas
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if file.endswith(('.xlsx', '.xls')) and not file.startswith('~'):
                    file_path = os.path.join(root, file)
                    
                    if self.detect_banco_republica_file(file_path):
                        variable_name = Path(file).stem
                        target_col, macro_type = self.extract_file_info(file_path)
                        
                        discovered.append({
                            'Variable': variable_name,
                            'Archivo': file_path,
                            'TARGET': target_col,
                            'Tipo_Macro': macro_type,
                            'Carpeta': os.path.basename(os.path.dirname(file_path))
                        })
        
        self.discovered_files = discovered
        self.logger.info(f"✅ Encontrados {len(discovered)} archivos del Banco República:")
        
        for file_info in discovered:
            self.logger.info(f"   📁 {file_info['Carpeta']}/{file_info['Variable']} → {file_info['Tipo_Macro']}")
        
        return discovered

    def extract_file_info(self, file_path):
        """📊 EXTRACCIÓN AUTOMÁTICA usando carpeta + contenido"""
        try:
            df = pd.read_excel(file_path, sheet_name=0, header=None, nrows=3)
            target_col = str(df.iloc[0, 1]) if len(df.columns) > 1 else "valor"
            
            # Usar nombre de carpeta como clasificador principal
            folder_name = os.path.basename(os.path.dirname(file_path)).lower()
            
            folder_to_macro = {
                'bond': 'bond',
                'business_confidence': 'business_confidence', 
                'car_registrations': 'car_registrations',
                'comm_loans': 'comm_loans',
                'commodities': 'commodities',
                'consumer_confidence': 'consumer_confidence',
                'economics': 'economics',
                'exchange_rate': 'exchange_rate',
                'exports': 'exports',
                'index_pricing': 'index_pricing',
                'leading_economic_index': 'leading_economic_index',
                'unemployment_rate': 'unemployment_rate'
            }
            
            if folder_name in folder_to_macro:
                macro_type = folder_to_macro[folder_name]
            else:
                # Fallback: clasificación por contenido
                header_text = target_col.lower()
                if any(term in header_text for term in ['colcap', 'índice', 'bursátil']):
                    macro_type = 'index_pricing'
                elif any(term in header_text for term in ['tasa de cambio', 'itcr']):
                    macro_type = 'exchange_rate'
                elif any(term in header_text for term in ['reservas', 'internacional']):
                    macro_type = 'exchange_rate'
                elif any(term in header_text for term in ['balanza', 'cuenta', 'inversión']):
                    macro_type = 'economics'
                else:
                    macro_type = 'economics'
                
            return target_col, macro_type
            
        except Exception:
            return "valor", "economics"

    def convert_colombian_date(self, date_str):
        """📅 Conversión de fechas formato colombiano"""
        if pd.isna(date_str) or not isinstance(date_str, str):
            return None
        try:
            return pd.to_datetime(date_str, format='%d/%m/%Y', dayfirst=True)
        except:
            try:
                return pd.to_datetime(date_str, dayfirst=True)
            except:
                return None

    def convert_colombian_number(self, value_str):
        """💰 Conversión de números formato colombiano"""
        if pd.isna(value_str):
            return None
        if isinstance(value_str, (int, float)):
            return float(value_str)
        if not isinstance(value_str, str):
            return None
            
        value_str = str(value_str).strip()
        if value_str in ['-', '', 'N/A', 'n/a']:
            return None
            
        try:
            # Formato colombiano: 1.234.567,89 → 1234567.89
            if '.' in value_str and ',' in value_str:
                value_str = value_str.replace('.', '').replace(',', '.')
            elif ',' in value_str and '.' not in value_str:
                value_str = value_str.replace(',', '.')
            elif '.' in value_str and ',' not in value_str:
                if value_str.count('.') == 1 and len(value_str.split('.')[1]) <= 2:
                    pass  # Es decimal
                else:
                    value_str = value_str.replace('.', '')  # Separador de miles
            return float(value_str)
        except (ValueError, TypeError):
            return None

    def process_file_automatically(self, file_info):
        """⚡ Procesamiento automático de un archivo"""
        variable = file_info['Variable']
        target_col = file_info['TARGET']
        macro_type = file_info['Tipo_Macro']
        file_path = file_info['Archivo']
        
        self.logger.info(f"\n📊 Procesando: {variable}")
        self.logger.info(f"   📁 Carpeta: {file_info['Carpeta']}")
        self.logger.info(f"   🎯 TARGET: {target_col}")
        self.logger.info(f"   📂 Tipo: {macro_type}")
        
        try:
            df = pd.read_excel(file_path, sheet_name=0, header=None)
            
            # Saltar headers y limpiar datos
            data_rows = df.iloc[2:].copy()
            data_rows = data_rows.dropna(subset=[0])
            data_rows = data_rows[~data_rows[0].astype(str).str.contains('Descargado|descargado', na=False)]
            
            if len(data_rows) == 0:
                self.logger.error(f"❌ No hay datos válidos en {variable}")
                return variable, None
            
            # Procesamiento automático: fecha (A) + valor (B)
            result_df = pd.DataFrame()
            result_df['fecha'] = data_rows[0].apply(self.convert_colombian_date)
            result_df = result_df.dropna(subset=['fecha'])
            
            if len(data_rows.columns) > 1:
                # Nombre estandarizado compatible con sistema principal
                clean_name = re.sub(r'[^\w\s]', '', target_col).replace(' ', '_')[:50]
                nuevo_nombre = f"{clean_name}_{variable}_{macro_type}"
                
                result_df[nuevo_nombre] = data_rows[1].apply(self.convert_colombian_number)
                result_df = result_df.dropna(subset=[nuevo_nombre])
            
            result_df = result_df.sort_values('fecha')
            
            if len(result_df) == 0:
                self.logger.error(f"❌ Sin valores válidos: {variable}")
                return variable, None
            
            # Actualizar fechas globales
            current_min = result_df['fecha'].min()
            current_max = result_df['fecha'].max()
            
            if self.global_min_date is None or current_min < self.global_min_date:
                self.global_min_date = current_min
            if self.global_max_date is None or current_max > self.global_max_date:
                self.global_max_date = current_max
            
            # Estadísticas (formato compatible)
            self.stats[variable] = {
                'macro_type': macro_type,
                'target_column': target_col,
                'total_rows': len(result_df),
                'valid_values': result_df.iloc[:, 1].notna().sum(),
                'coverage': 100.0,
                'date_min': current_min,
                'date_max': current_max,
                'nuevo_nombre': nuevo_nombre
            }
            
            self.logger.info(f"   ✅ {len(result_df)} valores válidos")
            self.logger.info(f"   📅 {current_min.strftime('%Y-%m-%d')} a {current_max.strftime('%Y-%m-%d')}")
            
            return variable, result_df
            
        except Exception as e:
            self.logger.error(f"❌ Error procesando {variable}: {str(e)}")
            return variable, None

    def generate_daily_index(self):
        """📅 Generar índice diario"""
        if self.global_min_date is None or self.global_max_date is None:
            self.logger.error("❌ No se pudieron determinar fechas globales")
            return None
            
        self.daily_index = pd.DataFrame({
            'fecha': pd.date_range(start=self.global_min_date, end=self.global_max_date, freq='D')
        })
        
        self.logger.info(f"📅 Índice diario: {len(self.daily_index)} días")
        return self.daily_index

    def combine_data(self):
        """🔗 Combinar datos con merge_asof"""
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
        self.logger.info(f"🔗 Datos combinados: {len(self.final_df)} filas, {len(self.final_df.columns)} columnas")
        return self.final_df

    def save_results(self, output_file):
        """💾 Guardar resultados"""
        if self.final_df is None:
            self.logger.error("❌ No hay datos para guardar")
            return False
            
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                self.final_df.to_excel(writer, sheet_name='Datos Diarios', index=False)
                df_stats = pd.DataFrame(self.stats).T
                df_stats.to_excel(writer, sheet_name='Estadisticas')
                
                meta = {
                    'Proceso': ['BancoRepublicaProcessor'],
                    'Fecha de proceso': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Total indicadores': [len(self.stats)],
                    'Periodo': [f"{self.global_min_date.strftime('%Y-%m-%d')} a {self.global_max_date.strftime('%Y-%m-%d')}"],
                    'Total días': [len(self.daily_index)]
                }
                pd.DataFrame(meta).to_excel(writer, sheet_name='Metadatos', index=False)
                
            self.logger.info(f"💾 Archivo guardado: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error guardando: {str(e)}")
            return False

    def run(self, output_file):
        """🚀 Ejecutar proceso completo"""
        start_time = time.time()
        
        if not self.auto_discover_files():
            self.logger.error("❌ No se encontraron archivos del Banco República")
            return False
        
        for file_info in self.discovered_files:
            var, df_processed = self.process_file_automatically(file_info)
            self.processed_data[var] = df_processed
        
        successful = len([df for df in self.processed_data.values() if df is not None])
        if successful == 0:
            self.logger.error("❌ No se procesó ningún archivo correctamente")
            return False
        
        self.generate_daily_index()
        self.combine_data()
        result = self.save_results(output_file)
        
        end_time = time.time()
        self.logger.info(f"\n⏱️  Tiempo: {end_time - start_time:.2f} segundos")
        self.logger.info(f"✅ Archivos procesados: {successful}")
        self.logger.info(f"🎯 Estado: {'COMPLETADO' if result else 'ERROR'}")
        
        return result

# FUNCIÓN INTEGRADA para usar como las otras
def ejecutar_banco_republica_processor(
    config_file=os.path.join(PROJECT_ROOT, 'pipelines/Data Engineering.xlsx'),
    output_file=os.path.join(PROJECT_ROOT, 'data/0_raw/datos_banco_republica_procesados.xlsx'),
    data_root=os.path.join(PROJECT_ROOT, 'data/0_raw'),
    log_file=os.path.join(PROJECT_ROOT, 'logs/banco_republica.log')
):
    """
    🤖 PROCESADOR DEL BANCO REPÚBLICA - INTEGRADO
    Función que sigue el mismo patrón que las otras del sistema
    
    ✅ Busca automáticamente en todas las subcarpetas categorizadas
    ✅ Detecta y procesa archivos del Banco República automáticamente
    ✅ Genera Excel independiente con misma estructura que los otros
    """
    processor = BancoRepublicaProcessor(data_root, log_file)
    return processor.run(output_file)

# FUNCIÓN PARA EJECUTAR TODOS LOS PROCESADORES (incluyendo Banco República)
def ejecutar_todos_los_procesadores():
    """
    🚀 Ejecuta TODOS los procesadores del sistema:
    1. MyInvesting Copy-Paste
    2. MyInvesting Normal  
    3. FRED Data
    4. Other Data
    5. 🆕 Banco República (AUTOMÁTICO)
    
    Mantiene los 5 Excel separados como está originalmente
    """
    print("🚀 Ejecutando TODOS los procesadores de datos económicos...")
    print("=" * 70)
    
    resultados = {}
    
    # 1. MyInvesting Copy-Paste
    print("\n1️⃣ MyInvesting Copy-Paste...")
    try:
        success1 = run_economic_data_processor()
        resultados['MyInvesting CP'] = success1
        print("✅ Completado" if success1 else "❌ Error")
    except Exception as e:
        print(f"❌ Error: {e}")
        resultados['MyInvesting CP'] = False
    
    # 2. MyInvesting Normal
    print("\n2️⃣ MyInvesting Normal...")
    try:
        success2 = ejecutar_myinvestingreportnormal()
        resultados['MyInvesting Normal'] = success2
        print("✅ Completado" if success2 else "❌ Error")
    except Exception as e:
        print(f"❌ Error: {e}")
        resultados['MyInvesting Normal'] = False
    
    # 3. FRED Data
    print("\n3️⃣ FRED Data...")
    try:
        success3 = run_fred_data_processor()
        resultados['FRED'] = success3
        print("✅ Completado" if success3 else "❌ Error")
    except Exception as e:
        print(f"❌ Error: {e}")
        resultados['FRED'] = False
    
    # 4. Other Data
    print("\n4️⃣ Other Data...")
    try:
        success4 = ejecutar_otherdataprocessor()
        resultados['Other'] = success4
        print("✅ Completado" if success4 else "❌ Error")
    except Exception as e:
        print(f"❌ Error: {e}")
        resultados['Other'] = False
    
    # 5. 🆕 Banco República
    print("\n5️⃣ 🆕 Banco República (AUTOMÁTICO)...")
    try:
        success5 = ejecutar_banco_republica_processor()
        resultados['Banco República'] = success5
        print("✅ Completado" if success5 else "❌ Error")
    except Exception as e:
        print(f"❌ Error: {e}")
        resultados['Banco República'] = False
    
    # Resumen final
    print("\n" + "=" * 70)
    print("📊 RESUMEN FINAL")
    print("=" * 70)
    
    exitosos = sum(resultados.values())
    total = len(resultados)
    
    print(f"📈 Procesadores exitosos: {exitosos}/{total}")
    print(f"\n📁 Archivos generados:")
    
    archivos_esperados = [
        "datos_economicos_procesados_cp.xlsx",
        "datos_economicos_normales_procesados.xlsx", 
        "datos_economicos_procesados_Fred.xlsx",
        "datos_economicos_other_procesados.xlsx",
        "datos_banco_republica_procesados.xlsx"
    ]
    
    for i, (nombre, success) in enumerate(resultados.items()):
        icono = "✅" if success else "❌"
        archivo = archivos_esperados[i] if success else "No generado"
        print(f"   {icono} {nombre}: {archivo}")
    
    print(f"\n🎯 Estado general: {'EXITOSO' if exitosos >= 3 else 'PARCIAL' if exitosos > 0 else 'FALLIDO'}")
    print(f"📂 Ubicación: data/0_raw/")
    
    return exitosos >= 3



# PROCESADOR DANE EXPORTACIONES - INTEGRADO AL SISTEMA
# Agregar este código al archivo step_0_preprocess.py existente

import pandas as pd
import numpy as np
import os
import re
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# CORRECCIÓN DANE PROCESSOR - Reemplazar en step_0_preprocess.py

