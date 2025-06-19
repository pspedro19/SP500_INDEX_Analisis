try:  # pragma: no cover - optional dependency
    import pandas as pd
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    pd = None
    np = None


def parse_european_number(value):
    """Convert numbers in European format to float, including scientific notation."""
    if pd is not None:
        if pd.isna(value) or value == "" or str(value).lower() == "nan":
            return float("nan")
    else:
        if value is None or value == "" or str(value).lower() == "nan":
            return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    
    str_value = str(value).strip()
    
    # Handle scientific notation (e.g., "-6,00E-06" -> "-6.00E-06")
    if 'E' in str_value.upper() or 'e' in str_value:
        # Replace comma with period before E notation
        if ',' in str_value and ('E' in str_value or 'e' in str_value):
            # Find position of E or e
            e_pos = max(str_value.upper().find('E'), str_value.find('e'))
            if e_pos > 0:
                # Replace comma with period only in the mantissa (before E)
                mantissa = str_value[:e_pos].replace(',', '.')
                exponent = str_value[e_pos:]
                str_value = mantissa + exponent
        return float(str_value)
    
    # Original logic for non-scientific notation
    if str_value.count(".") > 1:
        str_value = str_value.replace(".", "")
        return float(str_value)
    if "." in str_value and "," in str_value:
        str_value = str_value.replace(".", "").replace(",", ".")
        return float(str_value)
    if "," in str_value and str_value.count(",") == 1:
        str_value = str_value.replace(",", ".")
        return float(str_value)
    return float(str_value)


def format_number_for_powerbi(value):
    """Format numeric value for Power BI España with comma as decimal separator - PRESERVA TODOS LOS DECIMALES."""
    if pd is not None:
        if pd.isna(value):
            return ""
    else:
        if value is None or (isinstance(value, float) and value != value):
            return ""
    
    # Convertir a string para preservar precisión
    str_val = str(float(value))
    
    # Si es entero, devolver sin decimales
    if '.' not in str_val or str_val.endswith('.0'):
        return str(int(float(value)))
    
    # Para números muy pequeños en notación científica
    if 'e' in str_val.lower():
        # Mantener notación científica pero cambiar separador decimal
        return str_val.replace('.', ',')
    
    # Para números normales, usar hasta 8 decimales para preservar precisión
    num = float(value)
    if abs(num) < 0.001:
        # Números muy pequeños: usar hasta 8 decimales
        formatted = f"{num:.8f}".rstrip('0').rstrip('.')
    elif abs(num) < 1:
        # Números menores a 1: usar hasta 6 decimales
        formatted = f"{num:.6f}".rstrip('0').rstrip('.')
    else:
        # Números >= 1: usar hasta 4 decimales
        formatted = f"{num:.4f}".rstrip('0').rstrip('.')
    
    # Cambiar punto por coma para formato español
    return formatted.replace(".", ",")


from typing import Any


def compute_predicted_sp500(df: Any) -> Any:
    """Add ValorPredicho_SP500 column using the last known real price."""
    if pd is None or np is None:
        raise ImportError("pandas and numpy are required for compute_predicted_sp500")
    df = df.copy()
    df["ValorPredicho_SP500"] = np.nan

    historicos = df[df["ValorReal_SP500"].notna() & (df["ValorReal_SP500"] != "")]
    precio_base = None
    if not historicos.empty:
        last = historicos.iloc[-1]
        last_price = parse_european_number(last["ValorReal_SP500"])
        last_return = parse_european_number(last["ValorReal"])
        if not pd.isna(last_price) and not pd.isna(last_return):
            precio_base = last_price / (1 + last_return)

    for idx, row in df.iterrows():
        pred = row["ValorPredicho"]
        if pd.isna(pred):
            continue
        pred = parse_european_number(pred)  # Use parse_european_number to handle scientific notation
        if pd.isna(pred):  # Skip if conversion failed
            continue
        if row["ValorReal_SP500"] != "" and pd.notna(row["ValorReal_SP500"]):
            vr = parse_european_number(row["ValorReal_SP500"])
            vr_ret = parse_european_number(row["ValorReal"])  # Also use parse_european_number for consistency
            if pd.isna(vr) or pd.isna(vr_ret):  # Skip if conversion failed
                continue
            base = vr / (1 + vr_ret)
            df.at[idx, "ValorPredicho_SP500"] = base * (1 + pred)
        elif precio_base is not None:
            df.at[idx, "ValorPredicho_SP500"] = precio_base * (1 + pred)

    return df


def format_for_powerbi(df: Any) -> Any:
    """Format DataFrame for Power BI España with proper decimal separators."""
    if pd is None:
        raise ImportError("pandas is required for format_for_powerbi")

    df = df.copy()
    
    # Define all numeric columns that need Spanish formatting
    numeric_columns = [
        "ValorReal", "ValorPredicho", "ErrorAbsoluto", "ErrorPorcentual",
        "ValorReal_SP500", "ValorPredicho_SP500"
    ]
    
    # Format each numeric column
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: format_number_for_powerbi(parse_european_number(x)) 
                if pd.notna(x) and str(x).strip() != "" else ""
            )
    
    return df
