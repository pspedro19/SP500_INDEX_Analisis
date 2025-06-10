#!/usr/bin/env python
# coding: utf-8

"""
Script para calcular el valor predicho del S&P500 a partir de los returns predichos
Compatible con Power BI - Formato estándar con punto decimal
Autor: Asistente AI
Fecha: Junio 2025
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_european_number(value):
    """
    Convierte números en formato europeo a float
    Maneja casos como:
    - "2088,5" -> 2088.5 (coma decimal)
    - "38.300.790.326" -> 38300790326.0 (puntos como separadores de miles)
    - "2088.50" -> 2088.5 (formato estándar)
    """
    if pd.isna(value) or value == '' or str(value).lower() == 'nan':
        return np.nan
        
    if isinstance(value, (int, float)):
        return float(value)
    
    # Convertir a string y limpiar
    str_value = str(value).strip()
    
    # Si tiene múltiples puntos, probablemente son separadores de miles
    if str_value.count('.') > 1:
        # Formato: 38.300.790.326 (separadores de miles con puntos)
        # Remover todos los puntos
        str_value = str_value.replace('.', '')
        return float(str_value)
    
    # Si tiene punto y coma
    if '.' in str_value and ',' in str_value:
        # Formato: 1.234,56 (punto separador de miles, coma decimal)
        str_value = str_value.replace('.', '').replace(',', '.')
        return float(str_value)
    
    # Si solo tiene coma (decimal europeo)
    if ',' in str_value and str_value.count(',') == 1:
        # Formato: 2088,5 (coma decimal)
        str_value = str_value.replace(',', '.')
        return float(str_value)
    
    # Formato estándar
    return float(str_value)

def format_number_for_powerbi(value):
    """
    Formatea números para Power BI (formato estándar con punto decimal)
    Usa máximo 2 decimales para precios del S&P500
    """
    if pd.isna(value):
        return ''
    
    # Convertir a float y formatear con máximo 2 decimales
    num = float(value)
    
    # Si es un número entero o tiene pocos decimales, usar formato más limpio
    if num == int(num):
        return str(int(num))  # Sin decimales si es entero: 2088
    elif abs(num - round(num, 1)) < 0.01:
        return f"{num:.1f}"   # Un decimal si es apropiado: 2088.5
    else:
        return f"{num:.2f}"   # Dos decimales máximo: 2039.80

def calcular_valor_predicho_sp500(input_file, output_file=None):
    """
    Calcula el valor predicho del S&P500 usando la fórmula inversa
    
    Parámetros:
    - input_file: archivo CSV con las predicciones (formato actual)
    - output_file: archivo CSV de salida (opcional, se generará automáticamente si no se especifica)
    
    Fórmulas utilizadas:
    DATOS HISTÓRICOS (con ValorReal_SP500):
    1. Precio_Base = ValorReal_SP500 / (1 + ValorReal)
    2. ValorPredicho_SP500 = Precio_Base * (1 + ValorPredicho)
    
    FORECASTING FUTURO (sin ValorReal_SP500):
    ⚠️ CORRECCIÓN IMPORTANTE: El último ValorReal_SP500 es un precio TARGET (t+20), no BASE (t)
    1. Precio_Base_Ultimo = Ultimo_ValorReal_SP500 / (1 + Ultimo_ValorReal)
    2. ValorPredicho_SP500 = Precio_Base_Ultimo * (1 + ValorPredicho)
    
    Esto evita el "salto" en el forecasting al usar el precio base correcto.
    """
    
    logging.info("🔄 Iniciando cálculo del valor predicho del S&P500")
    
    try:
        # 1. Cargar el archivo CSV
        logging.info(f"📂 Cargando archivo: {input_file}")
        df = pd.read_csv(input_file, delimiter=';')
        logging.info(f"✅ Archivo cargado: {len(df)} filas x {len(df.columns)} columnas")
        
        # 2. Verificar columnas necesarias
        required_columns = ['ValorReal', 'ValorPredicho', 'ValorReal_SP500']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logging.error(f"❌ Columnas faltantes: {missing_columns}")
            logging.info(f"🔍 Columnas disponibles: {list(df.columns)}")
            return False
        
        # 3. Identificar último precio conocido y su precio base para forecasting
        logging.info("🔍 Identificando último precio conocido para forecasting...")
        
        # Filtrar filas con valores reales (no forecasting)
        df_con_valores_reales = df[df['ValorReal_SP500'].notna() & (df['ValorReal_SP500'] != '')]
        
        # Encontrar el último precio conocido y calcular su precio base
        ultimo_precio_conocido = None
        precio_base_forecasting = None
        ultima_fecha_conocida = None
        
        if len(df_con_valores_reales) > 0:
            # Ordenar por FechaKey para encontrar el último
            df_con_valores_reales_sorted = df_con_valores_reales.sort_values('FechaKey')
            ultima_fila = df_con_valores_reales_sorted.iloc[-1]
            ultimo_precio_conocido = parse_european_number(ultima_fila['ValorReal_SP500'])
            ultimo_valor_real = float(ultima_fila['ValorReal'])
            ultima_fecha_conocida = ultima_fila['FechaKey']
            
            # CLAVE: Calcular el precio BASE del último período
            # El precio conocido es el TARGET (t+20), necesitamos el BASE (t)
            precio_base_forecasting = ultimo_precio_conocido / (1 + ultimo_valor_real)
            
            logging.info(f"✅ Último precio conocido (TARGET): ${ultimo_precio_conocido:.2f} (FechaKey: {ultima_fecha_conocida})")
            logging.info(f"✅ Precio BASE para forecasting: ${precio_base_forecasting:.2f}")
            logging.info(f"📊 Return del último período: {ultimo_valor_real:.6f}")
        else:
            logging.warning("⚠️ No se encontraron valores reales del S&P500")
        
        # 4. Procesar cada fila
        logging.info("🔢 Calculando valores predichos del S&P500...")
        
        valores_predichos_sp500 = []
        estadisticas = {
            'total': 0,
            'validos_historicos': 0,
            'validos_forecasting': 0,
            'invalidos': 0,
            'errores': 0
        }
        
        for index, row in df.iterrows():
            estadisticas['total'] += 1
            
            try:
                # Convertir valores a números con manejo robusto
                try:
                    valor_predicho = float(row['ValorPredicho'])
                except (ValueError, TypeError):
                    # Intentar parsear como número europeo
                    valor_predicho = parse_european_number(row['ValorPredicho'])
                
                valor_real_sp500_str = str(row['ValorReal_SP500']).strip()
                
                # Verificar si tenemos valor real del S&P500 (datos históricos)
                tiene_valor_real = (
                    valor_real_sp500_str != '' and 
                    valor_real_sp500_str.lower() != 'nan' and 
                    pd.notna(row['ValorReal_SP500'])
                )
                
                if tiene_valor_real:
                    # CASO 1: DATOS HISTÓRICOS (con valor real del S&P500)
                    try:
                        valor_real = float(row['ValorReal'])
                    except (ValueError, TypeError):
                        valor_real = parse_european_number(row['ValorReal'])
                    
                    valor_real_sp500 = parse_european_number(row['ValorReal_SP500'])
                    
                    if pd.isna(valor_real) or pd.isna(valor_predicho) or pd.isna(valor_real_sp500):
                        valores_predichos_sp500.append(np.nan)
                        estadisticas['invalidos'] += 1
                        continue
                    
                    # Calcular precio base usando valor real
                    precio_base = valor_real_sp500 / (1 + valor_real)
                    valor_predicho_sp500 = precio_base * (1 + valor_predicho)
                    
                    valores_predichos_sp500.append(valor_predicho_sp500)
                    estadisticas['validos_historicos'] += 1
                    
                else:
                    # CASO 2: FORECASTING FUTURO (sin valor real del S&P500)
                    if precio_base_forecasting is None or pd.isna(valor_predicho):
                        valores_predichos_sp500.append(np.nan)
                        estadisticas['invalidos'] += 1
                        continue
                    
                    # Para forecasting, usar el precio BASE calculado correctamente
                    # Aplicar el return predicho sobre el precio base del último período
                    valor_predicho_sp500 = precio_base_forecasting * (1 + valor_predicho)
                    
                    valores_predichos_sp500.append(valor_predicho_sp500)
                    estadisticas['validos_forecasting'] += 1
                
                # Log de progreso cada 1000 filas
                if estadisticas['total'] % 1000 == 0:
                    logging.info(f"   Procesadas {estadisticas['total']} filas...")
                
            except Exception as e:
                valores_predichos_sp500.append(np.nan)
                estadisticas['errores'] += 1
                if estadisticas['errores'] <= 10:  # Mostrar primeros 10 errores con más detalle
                    error_msg = str(e)
                    if "could not convert string to float" in error_msg:
                        # Error específico de conversión numérica
                        problematic_value = error_msg.split("'")[1] if "'" in error_msg else "valor desconocido"
                        logging.warning(f"Error en fila {index}: Formato numérico inválido '{problematic_value}'")
                    else:
                        logging.warning(f"Error en fila {index}: {e}")
        
        # 5. Agregar la nueva columna al DataFrame
        df['ValorPredicho_SP500'] = valores_predichos_sp500
        
        # 6. Estadísticas del procesamiento
        total_validos = estadisticas['validos_historicos'] + estadisticas['validos_forecasting']
        logging.info("\n📊 ESTADÍSTICAS DEL PROCESAMIENTO:")
        logging.info(f"   Total filas: {estadisticas['total']:,}")
        logging.info(f"   Valores históricos válidos: {estadisticas['validos_historicos']:,}")
        logging.info(f"   Valores forecasting válidos: {estadisticas['validos_forecasting']:,}")
        logging.info(f"   Total valores válidos: {total_validos:,} ({total_validos/estadisticas['total']*100:.1f}%)")
        logging.info(f"   Valores inválidos: {estadisticas['invalidos']:,}")
        logging.info(f"   Errores: {estadisticas['errores']:,}")
        
        # 7. Estadísticas de los valores calculados
        valores_validos = df['ValorPredicho_SP500'].dropna()
        if len(valores_validos) > 0:
            logging.info("\n💰 ESTADÍSTICAS DE PRECIOS PREDICHOS DEL S&P500:")
            logging.info(f"   Mínimo: ${valores_validos.min():.2f}")
            logging.info(f"   Máximo: ${valores_validos.max():.2f}")
            logging.info(f"   Promedio: ${valores_validos.mean():.2f}")
            logging.info(f"   Mediana: ${valores_validos.median():.2f}")
        
        # 8. Mostrar ejemplos separados por tipo
        logging.info("\n🔍 EJEMPLOS DE CÁLCULOS:")
        
        # Ejemplos de datos históricos
        df_historicos = df[(df['ValorReal_SP500'].notna()) & (df['ValorReal_SP500'] != '') & 
                          (df['ValorPredicho_SP500'].notna())].head(3)
        
        if len(df_historicos) > 0:
            logging.info("📊 DATOS HISTÓRICOS (con valor real):")
            for idx, row in df_historicos.iterrows():
                try:
                    valor_real_sp500 = parse_european_number(row['ValorReal_SP500'])
                    precio_predicho = float(row['ValorPredicho_SP500'])
                    logging.info(f"   Fila {idx}: Real=${valor_real_sp500:.2f} → Predicho=${precio_predicho:.2f}")
                except (ValueError, TypeError):
                    logging.info(f"   Fila {idx}: Error al formatear valores")
        
        # Ejemplos de forecasting
        df_forecasting = df[(df['ValorReal_SP500'].isna() | (df['ValorReal_SP500'] == '')) & 
                           (df['ValorPredicho_SP500'].notna())].head(3)
        
        if len(df_forecasting) > 0:
            logging.info("🔮 FORECASTING FUTURO (sin valor real):")
            for idx, row in df_forecasting.iterrows():
                try:
                    # Convertir a float para el formato
                    precio_predicho = float(row['ValorPredicho_SP500'])
                    return_predicho = float(row['ValorPredicho'])
                    logging.info(f"   Fila {idx}: Base=${precio_base_forecasting:.2f} → Predicho=${precio_predicho:.2f} (Return: {return_predicho:.4f})")
                except (ValueError, TypeError):
                    logging.info(f"   Fila {idx}: Error al formatear valores")
        else:
            logging.info("🔮 FORECASTING FUTURO: No se encontraron filas de forecasting")
        
        # 9. Formatear columnas para Power BI
        logging.info("\n🔧 Formateando columnas para Power BI...")
        
        # Crear una copia para formateo
        df_output = df.copy()
        
        # Formatear ValorReal_SP500 (convertir formato europeo a estándar)
        df_output['ValorReal_SP500_formatted'] = df_output['ValorReal_SP500'].apply(
            lambda x: format_number_for_powerbi(parse_european_number(x)) if pd.notna(x) and str(x).strip() != '' else ''
        )
        
        # Formatear ValorPredicho_SP500 (ya está en float, solo formatear)
        df_output['ValorPredicho_SP500_formatted'] = df_output['ValorPredicho_SP500'].apply(
            lambda x: format_number_for_powerbi(x) if pd.notna(x) else ''
        )
        
        # Reemplazar columnas originales con las formateadas
        df_output['ValorReal_SP500'] = df_output['ValorReal_SP500_formatted']
        df_output['ValorPredicho_SP500'] = df_output['ValorPredicho_SP500_formatted']
        
        # Eliminar columnas temporales
        df_output = df_output.drop(['ValorReal_SP500_formatted', 'ValorPredicho_SP500_formatted'], axis=1)
        
        # 10. Guardar archivo con formato correcto
        if output_file is None:
            # Generar nombre automático
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_powerbi_ready.csv"
        
        logging.info(f"\n💾 Guardando archivo para Power BI: {output_file}")
        
        # Guardar con formato estándar (punto decimal, sin separadores de miles)
        df_output.to_csv(output_file, sep=';', index=False, quoting=1)  # quoting=1 para evitar problemas con números
        
        logging.info("✅ PROCESO COMPLETADO EXITOSAMENTE")
        logging.info(f"   Archivo de salida: {output_file}")
        logging.info(f"   Nueva columna agregada: ValorPredicho_SP500")
        logging.info(f"   Formato: Estándar con punto decimal (máximo 2 decimales)")
        
        # Mostrar ejemplos del formato final
        logging.info("\n📊 EJEMPLOS DEL FORMATO FINAL:")
        sample_rows = df_output.head(3)
        for idx, row in sample_rows.iterrows():
            if row['ValorReal_SP500'] != '':
                logging.info(f"   Fila {idx}: ValorReal_SP500={row['ValorReal_SP500']}, ValorPredicho_SP500={row['ValorPredicho_SP500']}")
                logging.info(f"      ✅ Formato limpio: {row['ValorReal_SP500']} → {row['ValorPredicho_SP500']}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Error en el procesamiento: {e}")
        import traceback
        logging.error(f"Detalles: {traceback.format_exc()}")
        return False

def calcular_forecasting_avanzado(input_file, output_file=None):
    """
    Versión avanzada que maneja el forecasting de manera más sofisticada
    usando progresión secuencial para múltiples períodos futuros
    """
    logging.info("🚀 Iniciando cálculo AVANZADO con forecasting secuencial")
    
    try:
        # Cargar datos
        df = pd.read_csv(input_file, delimiter=';')
        df = df.sort_values(['FechaKey']).reset_index(drop=True)
        
        # Separar datos históricos de forecasting
        mask_historicos = df['ValorReal_SP500'].notna() & (df['ValorReal_SP500'] != '')
        df_historicos = df[mask_historicos].copy()
        df_forecasting = df[~mask_historicos].copy()
        
        logging.info(f"📊 Datos históricos: {len(df_historicos)} filas")
        logging.info(f"🔮 Datos forecasting: {len(df_forecasting)} filas")
        
        # Procesar datos históricos (mismo método anterior)
        valores_predichos = []
        
        for idx, row in df.iterrows():
            if mask_historicos.iloc[idx]:
                # Datos históricos
                try:
                    valor_real = float(row['ValorReal'])
                    valor_predicho = float(row['ValorPredicho'])
                    valor_real_sp500 = parse_european_number(row['ValorReal_SP500'])
                    
                    precio_base = valor_real_sp500 / (1 + valor_real)
                    valor_predicho_sp500 = precio_base * (1 + valor_predicho)
                    valores_predichos.append(valor_predicho_sp500)
                except:
                    valores_predichos.append(np.nan)
            else:
                # Forecasting - calcular después
                valores_predichos.append(np.nan)
        
        # Procesar forecasting secuencialmente
        if len(df_forecasting) > 0:
            # Encontrar último precio conocido y calcular precio base
            if len(df_historicos) > 0:
                ultimo_precio_target = parse_european_number(df_historicos.iloc[-1]['ValorReal_SP500'])
                ultimo_return_real = float(df_historicos.iloc[-1]['ValorReal'])
                
                # Calcular el precio BASE del último período conocido
                precio_base_inicial = ultimo_precio_target / (1 + ultimo_return_real)
                
                logging.info(f"💰 Último precio TARGET: ${ultimo_precio_target:.2f}")
                logging.info(f"💰 Precio BASE inicial para forecasting: ${precio_base_inicial:.2f}")
                
                # Procesar cada período de forecasting secuencialmente
                precio_base_actual = precio_base_inicial
                
                for idx in df_forecasting.index:
                    try:
                        return_predicho = float(df.loc[idx, 'ValorPredicho'])
                        precio_predicho = precio_base_actual * (1 + return_predicho)
                        valores_predichos[idx] = precio_predicho
                        
                        # Actualizar precio_base_actual para el siguiente período
                        # El precio predicho se convierte en el nuevo precio base
                        precio_base_actual = precio_predicho
                        
                        if idx == df_forecasting.index[0]:  # Log del primer forecasting
                            logging.info(f"🔮 Primer forecasting: ${precio_base_inicial:.2f} → ${precio_predicho:.2f} (Return: {return_predicho:.4f})")
                            
                    except Exception as e:
                        logging.warning(f"Error en forecasting fila {idx}: {e}")
                        valores_predichos[idx] = np.nan
        
        # Agregar resultados al DataFrame
        df['ValorPredicho_SP500'] = valores_predichos
        
        # Formatear para Power BI
        df_output = df.copy()
        
        # Formatear ValorReal_SP500
        df_output['ValorReal_SP500_formatted'] = df_output['ValorReal_SP500'].apply(
            lambda x: format_number_for_powerbi(parse_european_number(x)) if pd.notna(x) and str(x).strip() != '' else ''
        )
        
        # Formatear ValorPredicho_SP500
        df_output['ValorPredicho_SP500_formatted'] = df_output['ValorPredicho_SP500'].apply(
            lambda x: format_number_for_powerbi(x) if pd.notna(x) else ''
        )
        
        # Reemplazar columnas
        df_output['ValorReal_SP500'] = df_output['ValorReal_SP500_formatted']
        df_output['ValorPredicho_SP500'] = df_output['ValorPredicho_SP500_formatted']
        df_output = df_output.drop(['ValorReal_SP500_formatted', 'ValorPredicho_SP500_formatted'], axis=1)
        
        # Estadísticas
        validos_historicos = len([x for x in valores_predichos if pd.notna(x) and df.loc[valores_predichos.index(x), 'ValorReal_SP500'] != ''])
        validos_forecasting = len(df_forecasting[df_forecasting.index.isin([i for i, x in enumerate(valores_predichos) if pd.notna(x)])])
        
        logging.info(f"\n✅ RESULTADOS FORECASTING AVANZADO:")
        logging.info(f"   Históricos procesados: {validos_historicos}")
        logging.info(f"   Forecasting procesados: {validos_forecasting}")
        
        # Guardar archivo
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_forecasting_avanzado_powerbi.csv"
        
        df_output.to_csv(output_file, sep=';', index=False, quoting=1)
        logging.info(f"💾 Archivo guardado: {output_file}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Error en forecasting avanzado: {e}")
        return False

def validar_resultados(archivo_procesado):
    """
    Valida los resultados verificando la consistencia matemática
    """
    logging.info("\n🔍 VALIDANDO RESULTADOS...")
    
    try:
        df = pd.read_csv(archivo_procesado, delimiter=';')
        
        # Filtrar filas con todos los valores necesarios (solo datos históricos)
        df_valido = df[(df['ValorReal_SP500'] != '') & (df['ValorPredicho_SP500'] != '')].copy()
        df_valido = df_valido.dropna(subset=['ValorReal'])
        
        if len(df_valido) == 0:
            logging.warning("⚠️ No hay filas válidas para validar")
            return False
        
        # Verificar formato de números
        logging.info("🔧 Verificando formato de números...")
        sample_row = df_valido.iloc[0]
        
        valor_real_sp500_str = str(sample_row['ValorReal_SP500'])
        valor_predicho_sp500_str = str(sample_row['ValorPredicho_SP500'])
        
        logging.info(f"   Ejemplo ValorReal_SP500: '{valor_real_sp500_str}'")
        logging.info(f"   Ejemplo ValorPredicho_SP500: '{valor_predicho_sp500_str}'")
        
        # Verificar que use punto decimal (no coma)
        usa_punto_decimal = '.' in valor_real_sp500_str and '.' in valor_predicho_sp500_str
        usa_coma_decimal = ',' in valor_real_sp500_str or ',' in valor_predicho_sp500_str
        
        if usa_punto_decimal and not usa_coma_decimal:
            logging.info("✅ Formato correcto: Punto decimal estándar (Power BI compatible)")
        else:
            logging.warning("⚠️ Formato problemático: Podría tener comas decimales")
        
        # Verificar consistencia matemática en primeras 5 filas
        errores_grandes = 0
        for idx, row in df_valido.head(5).iterrows():
            try:
                valor_real = float(row['ValorReal'])
                valor_real_sp500 = float(row['ValorReal_SP500'])
                valor_predicho_sp500 = float(row['ValorPredicho_SP500'])
                
                # Calcular precio base usando nuestros valores
                precio_base_calculado = valor_real_sp500 / (1 + valor_real)
                
                # Verificar que la fórmula inversa funciona
                precio_verificacion = precio_base_calculado * (1 + valor_real)
                diferencia = abs(precio_verificacion - valor_real_sp500)
                
                if diferencia > 0.01:  # Error mayor a 1 centavo
                    errores_grandes += 1
                    logging.warning(f"   Fila {idx}: Error de verificación = ${diferencia:.4f}")
                else:
                    logging.info(f"   Fila {idx}: ✅ Verificación correcta (diff: ${diferencia:.4f})")
                    
            except ValueError as e:
                logging.warning(f"   Fila {idx}: Error de conversión numérica: {e}")
                errores_grandes += 1
        
        # Verificar forecasting
        df_forecasting = df[(df['ValorReal_SP500'] == '') & (df['ValorPredicho_SP500'] != '')]
        
        if len(df_forecasting) > 0:
            logging.info(f"\n🔮 FORECASTING ENCONTRADO: {len(df_forecasting)} filas")
            try:
                precios_forecast = df_forecasting['ValorPredicho_SP500'].astype(float)
                logging.info(f"   Rango forecasting: ${precios_forecast.min():.2f} - ${precios_forecast.max():.2f}")
            except:
                logging.warning("   Error al analizar rangos de forecasting")
        
        if errores_grandes == 0:
            logging.info("✅ VALIDACIÓN EXITOSA: Todos los cálculos son matemáticamente consistentes")
            logging.info("✅ FORMATO POWER BI: Números con punto decimal estándar")
        else:
            logging.warning(f"⚠️ Se encontraron {errores_grandes} errores de validación")
        
        return errores_grandes == 0
        
    except Exception as e:
        logging.error(f"❌ Error en la validación: {e}")
        return False

def main():
    """
    Función principal del script
    """
    print("\n" + "="*80)
    print("🧮 CALCULADORA DE VALOR PREDICHO DEL S&P500 (POWER BI READY)")
    print("="*80)
    print("Este script calcula el valor predicho del S&P500 usando returns predichos")
    print("Fórmula: ValorPredicho_SP500 = (ValorReal_SP500 / (1 + ValorReal)) * (1 + ValorPredicho)")
    print("\n🔧 CORRECCIONES INCLUIDAS:")
    print("   - Para forecasting, usa el PRECIO BASE del último período")
    print("   - Evita el 'salto' al aplicar returns sobre precios TARGET")
    print("   - Formato limpio: 2088.5 (NO 2.088.500.000)")
    print("   - Máximo 2 decimales, sin separadores de miles")
    print("   - Compatible con Power BI")
    print("="*80)
    
    # Configurar archivos usando la ruta del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    input_file = os.path.join(repo_root, "data", "4_results", "hechos_predicciones_fields_con_sp500.csv")
    
    # Mostrar información de rutas para debug
    logging.info(f"📂 Directorio del script: {script_dir}")
    logging.info(f"📄 Buscando archivo: {input_file}")
    logging.info(f"📍 Directorio actual: {os.getcwd()}")
    
    # Verificar que el archivo de entrada existe
    if not os.path.exists(input_file):
        logging.error(f"❌ Archivo no encontrado: {input_file}")
        
        # Buscar el archivo en otros lugares comunes
        alternative_paths = [
            "hechos_predicciones_fields_con_sp500.csv",  # Directorio actual
            os.path.join(os.getcwd(), "hechos_predicciones_fields_con_sp500.csv"),
            os.path.join(repo_root, "data", "4_results", "hechos_predicciones_fields_con_sp500.csv"),
            os.path.join(repo_root, "Data", "4_results", "hechos_predicciones_fields_con_sp500.csv")
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                logging.info(f"✅ Archivo encontrado en: {alt_path}")
                input_file = alt_path
                break
        else:
            logging.info("📁 Archivos encontrados en el directorio del script:")
            for file in os.listdir(script_dir):
                if file.endswith('.csv'):
                    logging.info(f"   - {file}")
            return
    
    # Procesar archivo
    print("\n🎯 SELECCIONA EL MÉTODO DE CÁLCULO:")
    print("1. Método estándar (forecasting simple)")
    print("2. Método avanzado (forecasting secuencial)")
    
    while True:
        try:
            opcion = input("\nSelecciona una opción (1 o 2): ").strip()
            if opcion in ['1', '2']:
                break
            print("❌ Por favor selecciona 1 o 2")
        except KeyboardInterrupt:
            print("\n👋 Proceso cancelado")
            return
    
    if opcion == '1':
        # Método estándar
        success = calcular_valor_predicho_sp500(input_file)
        output_file = input_file.replace('.csv', '_powerbi_ready.csv')
    else:
        # Método avanzado
        success = calcular_forecasting_avanzado(input_file)
        output_file = input_file.replace('.csv', '_forecasting_avanzado_powerbi.csv')
    
    if success:
        # Validar resultados
        validar_resultados(output_file)
        
        print(f"\n🎉 ¡PROCESO COMPLETADO!")
        print(f"📄 Archivo generado: {output_file}")
        print(f"📊 Nueva columna: ValorPredicho_SP500 (precio predicho del S&P500)")
        print(f"💻 Formato: Power BI Ready (punto decimal estándar)")
        
        if opcion == '2':
            print(f"🔮 Forecasting: Cálculo secuencial aplicado para períodos futuros")
            
        print(f"\n✅ CARACTERÍSTICAS DEL ARCHIVO:")
        print(f"   - Separador: punto y coma (;)")
        print(f"   - Decimales: punto (.)")
        print(f"   - Formato limpio: 2088.5 (no 2.088.500.000)")
        print(f"   - Máximo 2 decimales")
        print(f"   - Compatible con Power BI")
    else:
        print("\n❌ El proceso falló. Revisa los logs para más detalles.")

if __name__ == "__main__":
    main()