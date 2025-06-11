import pandas as pd
from sp500_analysis.config import settings
from sp500_analysis.application.preprocessing.processors.filter_20days import (
    filter_last_n_days,
)

# Directorios de entrada y salida obtenidos de la configuración
archivo1_ruta = settings.processed_dir / "datos_economicos_1month_SP500_INFERENCE.xlsx"
archivo2_ruta = settings.training_dir / "ULTIMO_S&P500_final_FPI.xlsx"

# Ruta para el archivo de salida
archivo_salida = settings.training_dir / "datos_economicos_filtrados.xlsx"


def main() -> None:
    """Filter economic input data to the last configured days."""
    df1 = pd.read_excel(archivo1_ruta)
    df2 = pd.read_excel(archivo2_ruta)

    df_filtrado = filter_last_n_days(
        df1,
        df2,
        days=settings.forecast_horizon_1month,
        date_col=settings.date_col,
        target_suffix=settings.target_suffix,
    )

    df_filtrado.to_excel(archivo_salida, index=False)
    print(f"Proceso completado. Archivo guardado en: {archivo_salida}")


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
