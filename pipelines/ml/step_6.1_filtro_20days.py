import pandas as pd
import os

<<<<<<< HEAD
# Rutas de los archivos
archivo1_ruta = r"C:\Users\natus\Documents\Trabajo\PEDRO_PEREZ\Proyecto_Mercado_de_Valores\SP500_INDEX_Analisis\Data\2_processed\datos_economicos_1month_SP500_INFERENCE.xlsx"
archivo2_ruta = r"C:\Users\natus\Documents\Trabajo\PEDRO_PEREZ\Proyecto_Mercado_de_Valores\SP500_INDEX_Analisis\Data\3_trainingdata\ULTIMO_S&P500_final_FPI.xlsx"
=======
# Obtener el directorio actual y buscar la carpeta del proyecto
directorio_actual = os.getcwd()
directorio_proyecto = None

# Buscar hacia arriba en la estructura de directorios hasta encontrar SP500_INDEX_Analisis
temp_dir = directorio_actual
while temp_dir != os.path.dirname(temp_dir):  # Mientras no lleguemos a la raíz
    if "SP500_INDEX_Analisis" in os.path.basename(temp_dir):
        directorio_proyecto = temp_dir
        break
    temp_dir = os.path.dirname(temp_dir)

# Si no encontramos el directorio del proyecto en la ruta actual, buscarlo
if directorio_proyecto is None:
    # Buscar en el directorio actual y subdirectorios
    for root, dirs, files in os.walk(directorio_actual):
        if "SP500_INDEX_Analisis" in dirs:
            directorio_proyecto = os.path.join(root, "SP500_INDEX_Analisis")
            break
    
    # Si aún no se encuentra, asumir que estamos dentro del proyecto
    if directorio_proyecto is None:
        directorio_proyecto = directorio_actual

# Rutas de los archivos usando rutas relativas al proyecto
archivo1_ruta = os.path.join(directorio_proyecto, "Data", "2_processed", "datos_economicos_1month_SP500_INFERENCE.xlsx")
archivo2_ruta = os.path.join(directorio_proyecto, "Data", "3_trainingdata", "ULTIMO_S&P500_final_FPI.xlsx")
>>>>>>> 72d1a5de70d697d27da0fe65079b0cf71b11688e

# Ruta para el archivo de salida
directorio_salida = os.path.join(directorio_proyecto, "Data", "3_trainingdata")
archivo_salida = os.path.join(directorio_salida, "datos_economicos_filtrados.xlsx")

# Leer los archivos Excel
df1 = pd.read_excel(archivo1_ruta)
df2 = pd.read_excel(archivo2_ruta)

# Obtener las columnas del segundo archivo
columnas_referencia = df2.columns.tolist()

# Filtrar el DataFrame 1 para mantener solo las columnas que están en el DataFrame 2
columnas_comunes = [col for col in df1.columns if col in columnas_referencia]

# Eliminar la columna target si existe
columnas_comunes = [col for col in columnas_comunes if not col.endswith('_Return_Target')]

df_filtrado = df1[columnas_comunes]

# Verificar si hay columnas en la fecha (común en datos económicos)
columnas_fecha = [col for col in df_filtrado.columns if 'date' in col.lower() or 'fecha' in col.lower()]

# Imprimir información sobre el rango de fechas en el archivo original
if columnas_fecha:
    columna_fecha = columnas_fecha[0]
    
    # Asegurarse de que la columna de fecha es de tipo datetime
    if not pd.api.types.is_datetime64_any_dtype(df_filtrado[columna_fecha]):
        df_filtrado[columna_fecha] = pd.to_datetime(df_filtrado[columna_fecha], errors='coerce')
    
    # Información del archivo original
    fecha_inicio_original = df_filtrado[columna_fecha].min()
    fecha_fin_original = df_filtrado[columna_fecha].max()
    print(f"\nRango de fechas en el archivo original:")
    print(f"Fecha de inicio: {fecha_inicio_original.strftime('%Y-%m-%d')}")
    print(f"Fecha de fin: {fecha_fin_original.strftime('%Y-%m-%d')}")
    print(f"Total de días: {(fecha_fin_original - fecha_inicio_original).days + 1}")
    
    # Ordenar por fecha (orden descendente) para seleccionar los últimos 20 días
    df_filtrado = df_filtrado.sort_values(by=columna_fecha, ascending=False).head(20)
    
    # Después de seleccionar los 20 días, ordenar de nuevo pero ahora en orden ascendente (de más antigua a más reciente)
    df_filtrado = df_filtrado.sort_values(by=columna_fecha, ascending=True).reset_index(drop=True)
    
    # Información de los 20 días seleccionados
    fecha_inicio_seleccionada = df_filtrado[columna_fecha].min()
    fecha_fin_seleccionada = df_filtrado[columna_fecha].max()
    print(f"\nRango de fechas en los 20 días seleccionados:")
    print(f"Fecha de inicio: {fecha_inicio_seleccionada.strftime('%Y-%m-%d')}")
    print(f"Fecha de fin: {fecha_fin_seleccionada.strftime('%Y-%m-%d')}")
    print(f"Total de días: {(fecha_fin_seleccionada - fecha_inicio_seleccionada).days + 1}")
else:
    print("\nNo se encontró columna de fecha en los datos.")
    print("Se tomarán las últimas 20 filas asumiendo orden cronológico.")
    # Si no hay columna de fecha explícita, simplemente tomar las últimas 20 filas
    df_filtrado = df_filtrado.tail(20)
    # Invertir el orden para que vaya de más antiguo a más reciente
    df_filtrado = df_filtrado.iloc[::-1].reset_index(drop=True)

# Guardar el resultado en un nuevo archivo Excel
df_filtrado.to_excel(archivo_salida, index=False)

print(f"\nProceso completado. Archivo guardado en: {archivo_salida}")
print(f"Columnas originales en archivo 1: {len(df1.columns)}")
print(f"Columnas en archivo de referencia: {len(df2.columns)}")
print(f"Columnas conservadas: {len(columnas_comunes)}")
print(f"Número de filas en el archivo final: {len(df_filtrado)}")
print(f"\nLos datos están ordenados de la fecha más antigua ({fecha_inicio_seleccionada.strftime('%Y-%m-%d')}) a la más reciente ({fecha_fin_seleccionada.strftime('%Y-%m-%d')})")