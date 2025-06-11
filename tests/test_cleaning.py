import pytest

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas optional
    pd = None

from sp500_analysis.application.preprocessing.cleaning import (
    limpiar_nombre_columna,
    clean_dataframe_columns,
)


def test_limpiar_nombre_columna_removes_duplicate():
    assert (
        limpiar_nombre_columna("Denmark_Car_Registrations_MoM_Registrations")
        == "Denmark_Car_Registrations_MoM"
    )


def test_clean_dataframe_columns_renames_columns():
    if pd is None:
        pytest.skip("pandas not available")
    df = pd.DataFrame(
        [[1, 2, 3]],
        columns=[
            "Denmark_Car_Registrations_MoM_Registrations",
            "US_Car_Registrations_MoM_extra",
            "Other",
        ],
    )
    cleaned, renamed = clean_dataframe_columns(df)
    assert "Denmark_Car_Registrations_MoM" in cleaned.columns
    assert "US_Car_Registrations_MoM" in cleaned.columns
    assert (
        renamed["Denmark_Car_Registrations_MoM_Registrations"]
        == "Denmark_Car_Registrations_MoM"
    )
    assert renamed["US_Car_Registrations_MoM_extra"] == "US_Car_Registrations_MoM"
    assert "Other" in cleaned.columns
