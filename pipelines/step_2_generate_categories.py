import re
import logging
import pandas as pd
import os

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Rutas
archivo_entrada = "Data/processed/MERGEDEXCELS.xlsx"
carpeta_salida = "Data/processed"
archivo_salida = os.path.join(carpeta_salida, "MERGEDEXCELS_CATEGORIZADO.xlsx")
archivo_diagnostico = os.path.join(carpeta_salida, "DIAGNOSTICO_CATEGORIAS.xlsx")

# Crear carpeta si no existe
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

# Diccionario de renombres
column_renames = {
    "Denmark_Car_Resistrations": "Denmark_Car_Registrations_MoM",
    "US_Car_Registrations": "US_Car_Registrations_MoM",
    "SouthAfrica_Car_Registrations": "SouthAfrica_Car_Registrations_MoM",
    "United_Kingdom_Car_Registrations": "United_Kingdom_Car_Registrations_MoM",
    "Spain_Car_Registrations": "Spain_Car_Registrations_MoM",
    "Singapore_NonOil_Exports": "Singapore_NonOil_Exports_YoY",
    "Japan_M2_MoneySupply": "Japan_M2_MoneySupply_YoY",
    "China_M2_MoneySupply": "China_M2_MoneySupply_YoY",
    "US_Industrial_Production": "US_Industrial_Production_MoM",
    "UK_Retail_Sales": "UK_Retail_Sales_MoM"
}

# Patrones regex mejorados para categorías
cat_patterns = {
    'bond': [
        r"bond", 
        r"yield"
    ],
    'business_confidence': [
        r"business[_\s]confidence", 
        r"business[_\s]climate"
    ],
    'car_registrations': [
        r"car[_\s]registrations", 
        r"vehicle[_\s]registrations", 
        r"auto[_\s]sales"
    ],
    'comm_loans': [
        r"comm[_\s]loans", 
        r"commercial[_\s]loans", 
        r"business[_\s]loans"
    ],
    'commodities': [
        r"commodit", 
        r"oil", 
        r"gold", 
        r"silver", 
        r"natural[_\s]gas"
    ],
    'consumer_confidence': [
        r"consumer[_\s]confidence", 
        r"consumer[_\s]sentiment"
    ],
    'economics': [
        r"money", 
        r"m[0-9]", 
        r"gdp", 
        r"inflation", 
        r"cpi", 
        r"ppi", 
        r"economic"
    ],
    'exchange_rate': [
        r"exchange[_\s]rate", 
        r"currency", 
        r"forex", 
        r"usd", 
        r"eur", 
        r"jpy"
    ],
    'exports': [
        r"export", 
        r"import", 
        r"trade[_\s]balance"
    ],
    'index_pricing': [
        r"industrial[_\s]production", 
        r"retail[_\s]sales", 
        r"price(?!.*unemployment)", 
        r"index(?!.*unemployment)", 
        r"stock"
    ],
    'leading_economic_index': [
        r"leading[_\s]economic[_\s]index", 
        r"economic[_\s]indicator"
    ],
    'unemployment_rate': [
        r"unemployment", 
        r"employment", 
        r"labor", 
        r"job",
        r"payrolls",
        r"claims",
        r"jobless"
    ]
}

# Orden de prioridad para categorías (las primeras tienen mayor prioridad)
categorias_prioritarias = [
    'unemployment_rate',
    'car_registrations',
    'exports',
    'economics',
    'bond',
    'business_confidence',
    'comm_loans',
    'commodities',
    'consumer_confidence',
    'exchange_rate',
    'leading_economic_index',
    'index_pricing'
]

# Compilar todos los patrones con IGNORECASE
regex_patterns = {
    cat: re.compile('|'.join(f'({pattern})' for pattern in patterns), re.IGNORECASE)
    for cat, patterns in cat_patterns.items()
}

def categorize_column(col_name: str) -> str:
    """
    Determina la categoría de una columna basándose en patrones regex mejorados
    con sistema de prioridad para resolver conflictos.
    """
    # Verificar primero palabras clave específicas de alta prioridad
    if re.search(r"unemploy|employment|payrolls|claims|jobless", col_name, re.IGNORECASE):
        return "unemployment_rate"
        
    # Verificar cada categoría en orden de prioridad
    matching_categories = []
    for category in categorias_prioritarias:
        if regex_patterns[category].search(col_name):
            # Retornar inmediatamente categorías de alta prioridad
            if category == 'unemployment_rate':
                return category
            matching_categories.append(category)
    
    # Si encontramos alguna coincidencia, devolver la primera (mayor prioridad)
    if matching_categories:
        return matching_categories[0]
            
    # Reglas especiales para casos difíciles
    if "_MoM" in col_name or "_YoY" in col_name:
        if any(word.lower() in col_name.lower() for word in ["Production", "Sales", "Index"]):
            return "index_pricing"
        if any(word.lower() in col_name.lower() for word in ["Money", "M2"]):
            return "economics"
            
    return "Sin categoría"

def main():
    try:
        # Verificar si el archivo existe
        if not os.path.exists(archivo_entrada):
            logging.error(f"El archivo de entrada no existe: {archivo_entrada}")
            return
            
        # Cargar el archivo Excel
        df = pd.read_excel(archivo_entrada)
        logging.info(f"Archivo cargado correctamente. Dimensiones: {df.shape}")
        
        # Guardar lista de columnas originales para verificación
        columnas_originales = list(df.columns)
        
        # Verificar qué columnas del diccionario existen en el DataFrame
        columnas_encontradas = [col for col in column_renames.keys() if col in columnas_originales]
        columnas_no_encontradas = [col for col in column_renames.keys() if col not in columnas_originales]
        
        if columnas_no_encontradas:
            logging.warning(f"Las siguientes columnas no se encontraron en el archivo: {columnas_no_encontradas}")
        
        if columnas_encontradas:
            logging.info(f"Columnas a renombrar: {columnas_encontradas}")
            
            # Renombrar columnas específicas (solo las que existen)
            rename_dict = {k: v for k, v in column_renames.items() if k in columnas_originales}
            df.rename(columns=rename_dict, inplace=True)
        else:
            logging.info("No se encontraron columnas para renombrar.")
        
        # Generar categorías para cada columna
        categorias = []
        columnas_sin_categoria = []
        resultados_categorias = []
        
        for col in df.columns:
            cat = categorize_column(col)
            categorias.append(cat)
            resultados_categorias.append({"Columna": col, "Categoría": cat})
            if cat == "Sin categoría":
                columnas_sin_categoria.append(col)
        
        # Para diagnóstico: guardar lista detallada de todas las categorizaciones
        df_resultados = pd.DataFrame(resultados_categorias)
        df_resultados.to_excel(archivo_diagnostico, index=False)
        logging.info(f"Resultados detallados de categorización guardados en: {archivo_diagnostico}")
        
        # Crear un DataFrame con las categorías
        df_categoria = pd.DataFrame([categorias], columns=df.columns)
        
        # Para registro: contar cuántas columnas hay en cada categoría
        cat_counts = {}
        for cat in categorias:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            
        logging.info("Resumen de categorización:")
        for cat, count in sorted(cat_counts.items()):
            logging.info(f"  - {cat}: {count} columnas")
        
        if columnas_sin_categoria:
            logging.warning(f"Hay {len(columnas_sin_categoria)} columnas sin categoría.")
        
        # Agregar fila de categorías arriba del DataFrame original
        df_categorizado = pd.concat([df_categoria, df], ignore_index=True)
        
        # Guardar el archivo
        df_categorizado.to_excel(archivo_salida, index=False)
        logging.info(f"Archivo categorizado guardado correctamente en: {archivo_salida}")
            
    except Exception as e:
        logging.error(f"Error en el procesamiento: {e}", exc_info=True)

if __name__ == "__main__":
    main()