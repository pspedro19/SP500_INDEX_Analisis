import re
import logging
import pandas as pd
import os

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Rutas actualizadas
archivo_entrada = archivo_entrada = "Data/processed/MERGEDEXCELS_CATEGORIZADO.xlsx"
carpeta_salida = "Data/processed"
archivo_salida = os.path.join(carpeta_salida, "MERGEDEXCELS_CATEGORIZADO_LIMPIO.xlsx")

# Crear carpeta si no existe
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

def limpiar_nombre_columna(nombre):
    """
    Limpia nombres de columnas duplicados como 'Denmark_Car_Registrations_MoM_Registrations'
    para convertirlos en 'Denmark_Car_Registrations_MoM'
    """
    sufijos = ["_MoM", "_YoY"]
    for sufijo in sufijos:
        if sufijo in nombre:
            base_name = nombre.split(sufijo)[0]
            componentes = base_name.split('_')
            terminos_post_sufijo = nombre.split(sufijo)[1].strip('_').split('_')
            nombre_limpio = base_name + sufijo
            if any(comp.lower() == term.lower() for comp in componentes for term in terminos_post_sufijo):
                return nombre_limpio
    return nombre

def main():
    try:
        if not os.path.exists(archivo_entrada):
            logging.error(f"El archivo de entrada no existe: {archivo_entrada}")
            return
        
        df = pd.read_excel(archivo_entrada)
        columnas_originales = list(df.columns)
        logging.info(f"Archivo cargado correctamente. Columnas: {len(columnas_originales)}")

        nombres_estandar = [
            "Denmark_Car_Registrations_MoM",
            "US_Car_Registrations_MoM",
            "SouthAfrica_Car_Registrations_MoM",
            "United_Kingdom_Car_Registrations_MoM",
            "Spain_Car_Registrations_MoM",
            "Singapore_NonOil_Exports_YoY",
            "Japan_M2_MoneySupply_YoY",
            "China_M2_MoneySupply_YoY",
            "US_Industrial_Production_MoM",
            "UK_Retail_Sales_MoM"
        ]

        renombres_directos = {}
        for estandar in nombres_estandar:
            for col in columnas_originales:
                if col.startswith(estandar) and col != estandar:
                    renombres_directos[col] = estandar

        nuevos_nombres = {}
        columnas_modificadas = 0
        for col in columnas_originales:
            if col in renombres_directos:
                nuevos_nombres[col] = renombres_directos[col]
                columnas_modificadas += 1
            else:
                nuevo_nombre = limpiar_nombre_columna(col)
                if nuevo_nombre != col:
                    nuevos_nombres[col] = nuevo_nombre
                    columnas_modificadas += 1

        if nuevos_nombres:
            df.rename(columns=nuevos_nombres, inplace=True)
            logging.info(f"Se modificaron {columnas_modificadas} nombres de columnas.")
            for original, nuevo in nuevos_nombres.items():
                logging.info(f"Renombrando: '{original}' -> '{nuevo}'")
        else:
            logging.info("No se encontraron columnas que necesiten ser renombradas.")

        df.to_excel(archivo_salida, index=False)
        logging.info(f"Archivo con columnas limpias guardado en: {archivo_salida}")
            
    except Exception as e:
        logging.error(f"Error en el procesamiento: {e}", exc_info=True)

if __name__ == "__main__":
    main()
