try:  # pragma: no cover - optional dependency
    import pandas as pd
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    pd = None
    np = None


def parse_european_number(value):
    """Convert numbers in European format to float."""
    if pd is not None:
        if pd.isna(value) or value == "" or str(value).lower() == "nan":
            return float("nan")
    else:
        if value is None or value == "" or str(value).lower() == "nan":
            return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    str_value = str(value).strip()
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
    """Format numeric value for Power BI with up to two decimals."""
    if pd is not None:
        if pd.isna(value):
            return ""
    else:
        if value is None or (isinstance(value, float) and value != value):
            return ""
    num = float(value)
    if num == int(num):
        return str(int(num))
    elif abs(num - round(num, 1)) < 0.01:
        return f"{num:.1f}"
    return f"{num:.2f}"


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
        last_return = float(last["ValorReal"])
        precio_base = last_price / (1 + last_return)

    for idx, row in df.iterrows():
        pred = row["ValorPredicho"]
        if pd.isna(pred):
            continue
        pred = float(pred)
        if row["ValorReal_SP500"] != "" and pd.notna(row["ValorReal_SP500"]):
            vr = parse_european_number(row["ValorReal_SP500"])
            vr_ret = float(row["ValorReal"])
            base = vr / (1 + vr_ret)
            df.at[idx, "ValorPredicho_SP500"] = base * (1 + pred)
        elif precio_base is not None:
            df.at[idx, "ValorPredicho_SP500"] = precio_base * (1 + pred)

    return df


def format_for_powerbi(df: Any) -> Any:
    if pd is None:
        raise ImportError("pandas is required for format_for_powerbi")

    df = df.copy()
    df["ValorReal_SP500"] = df["ValorReal_SP500"].apply(
        lambda x: format_number_for_powerbi(parse_european_number(x)) if pd.notna(x) and str(x).strip() != "" else ""
    )
    df["ValorPredicho_SP500"] = df["ValorPredicho_SP500"].apply(
        lambda x: format_number_for_powerbi(x) if pd.notna(x) else ""
    )
    return df
