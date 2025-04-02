import pandas as pd
import os

def cargar_archivo(ruta_archivo):
    """
    Carga un archivo Excel en un DataFrame de pandas.
    
    Args:
        ruta_archivo (str): Ruta al archivo Excel
        
    Returns:
        DataFrame: Datos del archivo cargado
    """
    try:
        print(f"Cargando archivo: {ruta_archivo}")
        return pd.read_excel(ruta_archivo)
    except Exception as e:
        print(f"Error al cargar {ruta_archivo}: {e}")
        return None

def filtrar_por_fecha(df, columna_fecha, fecha_inicio, fecha_fin):
    """
    Filtra un DataFrame por un rango de fechas.
    
    Args:
        df (DataFrame): DataFrame a filtrar
        columna_fecha (str): Nombre de la columna con fechas
        fecha_inicio (str): Fecha de inicio en formato 'YYYY-MM-DD'
        fecha_fin (str): Fecha de fin en formato 'YYYY-MM-DD'
        
    Returns:
        DataFrame: DataFrame filtrado
    """
    if df is None or columna_fecha not in df.columns:
        return df
    
    # Convertir a datetime si es necesario
    if not pd.api.types.is_datetime64_any_dtype(df[columna_fecha]):
        df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors='coerce')
    
    # Filtrar por rango de fechas
    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = pd.to_datetime(fecha_fin)
    
    return df[(df[columna_fecha] >= fecha_inicio) & (df[columna_fecha] <= fecha_fin)]



def imputar_valor_fecha_faltante(df, columna_fecha, fecha_base, fecha_objetivo):
    """
    Si 'fecha_objetivo' no existe en el DataFrame, copia la fila de 'fecha_base' y la agrega con la nueva fecha.
    """
    if df is None or columna_fecha not in df.columns:
        return df

    fecha_base = pd.to_datetime(fecha_base)
    fecha_objetivo = pd.to_datetime(fecha_objetivo)
    
    if fecha_objetivo in df[columna_fecha].values:
        return df  # No hace falta imputar

    # Buscar fila base
    fila_base = df[df[columna_fecha] == fecha_base]
    if fila_base.empty:
        print(f"No se encontró data para la fecha base {fecha_base.date()} en este archivo.")
        return df

    # Copiar fila y cambiar la fecha
    nueva_fila = fila_base.copy()
    nueva_fila[columna_fecha] = fecha_objetivo

    # Concatenar al DataFrame
    df = pd.concat([df, nueva_fila], ignore_index=True)
    print(f"Se imputó valor del {fecha_base.date()} para la fecha faltante {fecha_objetivo.date()}")
    return df


def merge_dataframes(dfs, columnas_merge, como_merge='outer'):
    """
    Combina múltiples DataFrames en uno solo.
    
    Args:
        dfs (list): Lista de DataFrames a combinar
        columnas_merge (list): Lista de columnas por las que hacer el merge
        como_merge (str): Tipo de merge (inner, outer, left, right)
        
    Returns:
        DataFrame: DataFrame combinado
    """
    if not dfs:
        return None
    
    # Filtrar DataFrames None
    dfs = [df for df in dfs if df is not None]
    
    if len(dfs) == 0:
        return None
    
    if len(dfs) == 1:
        return dfs[0]
    
    # Iniciar con el primer DataFrame
    resultado = dfs[0]
    
    # Combinar con el resto
    for i, df in enumerate(dfs[1:], 1):
        print(f"Combinando DataFrame {i} con el resultado")
        resultado = pd.merge(resultado, df, on=columnas_merge, how=como_merge)
    
    return resultado

# Configuraciones principales
# Carpeta de entrada (donde están los Excel)
ruta_carpeta = "Data/Macro/Documents_tomerge" # punto significa 'la carpeta actual'


# Carpeta de salida (donde se guarda el Excel combinado)
ruta_salida = "Data/Macro/pre_process"

archivos = [f for f in os.listdir(ruta_carpeta) if f.endswith('.xlsx') and not f.startswith('~$')]

print(f"Se encontraron {len(archivos)} archivos Excel en la carpeta.")

columna_fecha = "fecha"  # Ajusta al nombre de tu columna de fecha
columnas_merge = ["fecha"]  # Ajusta a las columnas comunes para el merge
fecha_inicio = "2014-01-01"
fecha_fin = "2025-03-27"
archivo_salida = os.path.join(ruta_salida, "MERGEDEXCELS.xlsx")

def main():
    """
    Función principal para ejecutar el proceso de combinación
    """
    print("Iniciando proceso de combinación de archivos Excel")
    
    # Cargar y filtrar cada archivo
    dfs = []
    for archivo in archivos:
        ruta_completa = os.path.join(ruta_carpeta, archivo)
        df = cargar_archivo(ruta_completa)
        if df is not None:
            print(f"Archivo {archivo} cargado correctamente con {len(df)} filas")
            df_filtrado = filtrar_por_fecha(df, columna_fecha, fecha_inicio, fecha_fin)
            df_filtrado = imputar_valor_fecha_faltante(df_filtrado, columna_fecha, "2025-03-26", "2025-03-27")
            print(f"Después de filtrar por fecha: {len(df_filtrado)} filas")
            dfs.append(df_filtrado)
    
    # Combinar los DataFrames
    df_combinado = merge_dataframes(dfs, columnas_merge)
    
    if df_combinado is None:
        print("No se pudo combinar los archivos")
        return False
    
    print(f"Combinación exitosa. Resultado tiene {len(df_combinado)} filas y {len(df_combinado.columns)} columnas")
    

        # Justo antes de guardar el archivo combinado, añade estas líneas:
    # Eliminar todas las columnas sin nombre (que comienzan con "Unnamed")
    columnas_a_mantener = [col for col in df_combinado.columns if not col.startswith('Unnamed')]
    df_combinado = df_combinado[columnas_a_mantener]


    # Guardar el resultado
    try:
        df_combinado.to_excel(archivo_salida, index=False)
        print(f"Archivo combinado guardado como: {archivo_salida}")
        return True
    except Exception as e:
        print(f"Error al guardar el archivo combinado: {e}")
        return False

if __name__ == "__main__":
    main()