#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONVERTIDOR CSV PARA POWER BI ESPAÑA - VERSIÓN MEJORADA
======================================================
Convierte archivos CSV al formato correcto para Power BI en español:
- Delimitador: punto y coma (;)
- Separador decimal: coma (,)
- Manejo robusto de errores y formatos inconsistentes
"""

import pandas as pd
import os
import csv
import chardet

def detectar_encoding(archivo):
    """Detecta la codificación del archivo"""
    try:
        with open(archivo, 'rb') as f:
            raw_data = f.read(10000)  # Lee los primeros 10KB
            result = chardet.detect(raw_data)
            return result['encoding']
    except:
        return 'utf-8'

def detectar_delimitador(archivo, encoding='utf-8'):
    """Detecta el delimitador usado en el CSV"""
    try:
        with open(archivo, 'r', encoding=encoding) as f:
            # Lee las primeras líneas para detectar el delimitador
            sample = f.read(1024)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            return delimiter
    except:
        return ','

def leer_csv_robusto(archivo_entrada):
    """Lee un CSV de forma robusta, manejando diferentes problemas comunes"""
    
    # Detectar encoding
    encoding = detectar_encoding(archivo_entrada)
    print(f"   🔍 Encoding detectado: {encoding}")
    
    # Detectar delimitador
    delimiter = detectar_delimitador(archivo_entrada, encoding)
    print(f"   🔍 Delimitador detectado: '{delimiter}'")
    
    # Intentar diferentes estrategias de lectura
    estrategias = [
        # Estrategia 1: Lectura normal
        {
            'params': {
                'delimiter': delimiter,
                'decimal': '.',
                'encoding': encoding,
                'quoting': csv.QUOTE_MINIMAL
            },
            'descripcion': 'Lectura normal'
        },
        # Estrategia 2: Con manejo de errores
        {
            'params': {
                'delimiter': delimiter,
                'decimal': '.',
                'encoding': encoding,
                'quoting': csv.QUOTE_ALL,
                'on_bad_lines': 'skip'
            },
            'descripcion': 'Saltando líneas problemáticas'
        },
        # Estrategia 3: Motor Python (más lento pero más robusto)
        {
            'params': {
                'delimiter': delimiter,
                'decimal': '.',
                'encoding': encoding,
                'engine': 'python',
                'quoting': csv.QUOTE_MINIMAL,
                'on_bad_lines': 'skip'
            },
            'descripcion': 'Motor Python con skip de errores'
        },
        # Estrategia 4: Sin quotes
        {
            'params': {
                'delimiter': delimiter,
                'decimal': '.',
                'encoding': encoding,
                'quoting': csv.QUOTE_NONE,
                'engine': 'python',
                'on_bad_lines': 'skip'
            },
            'descripcion': 'Sin procesamiento de comillas'
        }
    ]
    
    for i, estrategia in enumerate(estrategias, 1):
        try:
            print(f"   🧪 Probando estrategia {i}: {estrategia['descripcion']}")
            df = pd.read_csv(archivo_entrada, **estrategia['params'])
            print(f"   ✅ Éxito con estrategia {i}")
            return df
        except Exception as e:
            print(f"   ❌ Falló estrategia {i}: {str(e)[:100]}...")
            continue
    
    # Si todo falla, intentar leer línea por línea
    print("   🔧 Intentando lectura línea por línea...")
    try:
        return leer_linea_por_linea(archivo_entrada, delimiter, encoding)
    except Exception as e:
        raise Exception(f"No se pudo leer el archivo con ninguna estrategia: {str(e)}")

def leer_linea_por_linea(archivo_entrada, delimiter, encoding):
    """Lee el CSV línea por línea para manejar errores específicos"""
    filas_validas = []
    errores = []
    
    with open(archivo_entrada, 'r', encoding=encoding) as f:
        reader = csv.reader(f, delimiter=delimiter)
        headers = next(reader)  # Primera línea como headers
        
        for i, fila in enumerate(reader, start=2):  # Empezar en línea 2
            try:
                if len(fila) == len(headers):
                    filas_validas.append(fila)
                else:
                    errores.append(f"Línea {i}: esperaba {len(headers)} campos, encontró {len(fila)}")
                    # Intentar ajustar la fila
                    if len(fila) < len(headers):
                        fila.extend([''] * (len(headers) - len(fila)))
                    else:
                        fila = fila[:len(headers)]
                    filas_validas.append(fila)
            except Exception as e:
                errores.append(f"Línea {i}: {str(e)}")
    
    if errores:
        print(f"   ⚠️  Se encontraron {len(errores)} errores:")
        for error in errores[:5]:  # Mostrar solo los primeros 5
            print(f"      - {error}")
        if len(errores) > 5:
            print(f"      ... y {len(errores) - 5} errores más")
    
    # Crear DataFrame
    df = pd.DataFrame(filas_validas, columns=headers)
    print(f"   📊 Filas recuperadas: {len(df):,}")
    
    return df

def convertir_csv_powerbi_espanol(archivo_entrada, columnas_decimales, archivo_salida):
    """
    Convierte un CSV al formato de Power BI España
    
    Args:
        archivo_entrada (str): Nombre del archivo CSV original
        columnas_decimales (list): Lista de columnas que contienen números decimales
        archivo_salida (str): Nombre del archivo CSV de salida
    """
    try:
        print(f"\n🔄 Procesando: {archivo_entrada}")
        
        # Leer CSV original con estrategia robusta
        df = leer_csv_robusto(archivo_entrada)
        
        print(f"   📊 Filas leídas: {len(df):,}")
        print(f"   📋 Columnas: {len(df.columns)}")
        print(f"   📋 Nombres de columnas: {list(df.columns)}")
        print(f"   🔢 Columnas decimales especificadas: {', '.join(columnas_decimales)}")
        
        # Verificar que las columnas existen
        columnas_existentes = [col for col in columnas_decimales if col in df.columns]
        columnas_faltantes = [col for col in columnas_decimales if col not in df.columns]
        
        if columnas_faltantes:
            print(f"   ⚠️  Columnas no encontradas: {', '.join(columnas_faltantes)}")
            print(f"   💡 Columnas disponibles: {', '.join(df.columns.tolist())}")
        
        if columnas_existentes:
            print(f"   ✅ Columnas decimales encontradas: {', '.join(columnas_existentes)}")
        
        # Verificar tipos de datos de las columnas decimales
        for col in columnas_existentes:
            try:
                # Intentar convertir a numérico si no lo es
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"   🔢 {col}: {df[col].dtype}")
            except Exception as e:
                print(f"   ⚠️  Problema con columna {col}: {str(e)}")
        
        # Guardar en formato español (delimitador punto y coma, separador decimal coma)
        df.to_csv(
            archivo_salida,
            sep=';',           # Delimitador: punto y coma
            decimal=',',       # Separador decimal: coma
            index=False,       # Sin índice
            encoding='utf-8',  # Codificación UTF-8
            quoting=csv.QUOTE_MINIMAL  # Comillas mínimas
        )
        
        print(f"   ✅ Guardado: {archivo_salida}")
        
        # Verificar el archivo guardado
        try:
            df_test = pd.read_csv(archivo_salida, sep=';', decimal=',')
            print(f"   ✅ Verificación: {len(df_test)} filas en archivo de salida")
        except Exception as e:
            print(f"   ⚠️  Error en verificación: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

def main():
    """Función principal - Convierte ambos archivos"""
    
    print("🇪🇸 CONVERTIDOR CSV PARA POWER BI ESPAÑA - VERSIÓN MEJORADA")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    # Configuración de archivos
    archivos_config = [
        {
            'entrada': 'hechos_predicciones_fields_con_sp500_powerbi_ready.csv',
            'salida': 'hechos_predicciones_fields_POWERBI_ES1.csv',
            'columnas_decimales': ['ValorReal', 'ValorPredicho', 'ErrorAbsoluto', 'ErrorPorcentual']
        },
        {
            'entrada': 'hechos_metricas_modelo.csv',
            'salida': 'hechos_metricas_modelo_POWERBI_ES1.csv',
            'columnas_decimales': ['RMSE', 'MAE', 'R2', 'SMAPE']
        }
    ]
    
    archivos_procesados = 0

    # Procesar cada archivo
    for config in archivos_config:
        input_path = config['entrada']
        if not os.path.exists(input_path):
            alt_path = os.path.join(repo_root, 'data', '4_results', input_path)
            if os.path.exists(alt_path):
                input_path = alt_path
            else:
                print(f"\n❌ Archivo no encontrado: {input_path}")
                continue

        tamaño = os.path.getsize(input_path) / (1024 * 1024)  # MB
        print(f"\n📁 Archivo encontrado: {input_path} ({tamaño:.1f} MB)")

        if convertir_csv_powerbi_espanol(
            input_path,
            config['columnas_decimales'],
            config['salida']
        ):
            archivos_procesados += 1
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"✅ CONVERSIÓN COMPLETADA")
    print(f"📁 Archivos procesados: {archivos_procesados}/{len(archivos_config)}")
    print(f"\n🔧 FORMATO DE SALIDA:")
    print(f"   • Delimitador: punto y coma (;)")
    print(f"   • Separador decimal: coma (,)")
    print(f"   • Codificación: UTF-8")
    print(f"\n💡 PARA USAR EN POWER BI:")
    print(f"   1. Abre Power BI Desktop")
    print(f"   2. Obtener datos > Texto/CSV")
    print(f"   3. Selecciona el archivo *_POWERBI_ES1.csv")
    print(f"   4. En 'Delimitador' selecciona 'Punto y coma'")
    print(f"   5. ¡Los números decimales ahora se importarán correctamente!")

if __name__ == "__main__":
    main()

# INSTRUCCIONES DE USO:
# ===================
# 1. Instala la dependencia: pip install chardet
# 2. Guarda este código como 'convertir_powerbi_mejorado.py'
# 3. Coloca los archivos CSV originales en la misma carpeta
# 4. Ejecuta: python convertir_powerbi_mejorado.py
# 5. Usa los archivos *_POWERBI_ES1.csv generados en Power BI
#
# NOTAS:
# - Este script detecta automáticamente el encoding y delimitador
# - Maneja múltiples estrategias de lectura para archivos problemáticos
# - Proporciona información detallada sobre errores encontrados
# - Verifica el archivo de salida para asegurar que se guardó correctamente