from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    import pandas as pd

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas may be unavailable
    pd = None  # type: ignore

__all__ = [
    "STANDARD_NAMES",
    "limpiar_nombre_columna",
    "clean_dataframe_columns",
]

STANDARD_NAMES: List[str] = [
    "Denmark_Car_Registrations_MoM",
    "US_Car_Registrations_MoM",
    "SouthAfrica_Car_Registrations_MoM",
    "United_Kingdom_Car_Registrations_MoM",
    "Spain_Car_Registrations_MoM",
    "Singapore_NonOil_Exports_YoY",
    "Japan_M2_MoneySupply_YoY",
    "China_M2_MoneySupply_YoY",
    "US_Industrial_Production_MoM",
    "UK_Retail_Sales_MoM",
]


def limpiar_nombre_columna(nombre: str) -> str:
    """Return *nombre* without repeated terms after a known suffix."""
    sufijos = ["_MoM", "_YoY"]
    for sufijo in sufijos:
        if sufijo in nombre:
            base_name = nombre.split(sufijo)[0]
            componentes = base_name.split("_")
            terminos_post = nombre.split(sufijo)[1].strip("_").split("_")
            nombre_limpio = base_name + sufijo
            if any(comp.lower() == term.lower() for comp in componentes for term in terminos_post):
                return nombre_limpio
    return nombre


def clean_dataframe_columns(
    df: "pd.DataFrame", standard_names: List[str] | None = None
) -> Tuple["pd.DataFrame", Dict[str, str]]:
    """Clean duplicate column patterns in *df*.

    Parameters
    ----------
    df:
        DataFrame whose columns will be cleaned.
    standard_names:
        Optional list of standard base names to normalize. If ``None``,
        :data:`STANDARD_NAMES` is used.

    Returns
    -------
    tuple
        A tuple ``(clean_df, renamed)`` where ``clean_df`` contains the
        cleaned DataFrame and ``renamed`` maps original column names to the
        new names.
    """
    if pd is None:
        raise ImportError("pandas is required for clean_dataframe_columns")

    if standard_names is None:
        standard_names = STANDARD_NAMES

    columnas_originales = list(df.columns)
    renombres_directos: Dict[str, str] = {}
    for estandar in standard_names:
        for col in columnas_originales:
            if col.startswith(estandar) and col != estandar:
                renombres_directos[col] = estandar

    nuevos_nombres: Dict[str, str] = {}
    for col in columnas_originales:
        if col in renombres_directos:
            nuevos_nombres[col] = renombres_directos[col]
        else:
            nuevo_nombre = limpiar_nombre_columna(col)
            if nuevo_nombre != col:
                nuevos_nombres[col] = nuevo_nombre

    if nuevos_nombres:
        df = df.rename(columns=nuevos_nombres)

    return df, nuevos_nombres
