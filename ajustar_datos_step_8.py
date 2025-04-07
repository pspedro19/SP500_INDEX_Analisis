import pandas as pd

input_file = "Data/Macro/Training/all_models_predictions.csv"
output_file = "Data/Macro/Power_bi/archivo_para_powerbi.csv"

# Leer todo como texto
df = pd.read_csv(input_file, encoding='utf-8', dtype=str)

# Detectar y reemplazar los decimales: punto → coma
df['Valor_Real'] = df['Valor_Real'].str.replace(".", ",")
df['Valor_Predicho'] = df['Valor_Predicho'].str.replace(".", ",")
df['RMSE'] = df['RMSE'].str.replace(".", ",")

# Arreglar la columna de hiperparámetros
col_hiper = [col for col in df.columns if 'parámetros' in col][0]
df[col_hiper] = df[col_hiper].apply(lambda x: f'"{x}"' if not str(x).startswith('"') else x)

# Guardar con delimitador punto y coma y decimal con coma
df.to_csv(output_file, index=False, sep=';', encoding='utf-8', quoting=1)  # quoting=1 = QUOTE_MINIMAL

print(f"✅ CSV listo para Power BI (formato español): {output_file}")
