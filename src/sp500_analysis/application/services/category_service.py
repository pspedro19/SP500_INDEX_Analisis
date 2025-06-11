from __future__ import annotations

import logging
import re
from datetime import datetime

from sp500_analysis.config.settings import settings
from sp500_analysis.shared.logging.logger import configurar_logging

try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

# Mapping of wrong column names to corrected versions
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
    "UK_Retail_Sales": "UK_Retail_Sales_MoM",
}

# Regex patterns to identify column categories
cat_patterns = {
    "bond": [r"bond", r"yield"],
    "business_confidence": [r"business[_\s]confidence", r"business[_\s]climate"],
    "car_registrations": [
        r"car[_\s]registrations",
        r"vehicle[_\s]registrations",
        r"auto[_\s]sales",
    ],
    "comm_loans": [r"comm[_\s]loans", r"commercial[_\s]loans", r"business[_\s]loans"],
    "commodities": [r"commodit", r"oil", r"gold", r"silver", r"natural[_\s]gas"],
    "consumer_confidence": [r"consumer[_\s]confidence", r"consumer[_\s]sentiment"],
    "economics": [r"money", r"m[0-9]", r"gdp", r"inflation", r"cpi", r"ppi", r"economic"],
    "exchange_rate": [r"exchange[_\s]rate", r"currency", r"forex", r"usd", r"eur", r"jpy"],
    "exports": [r"export", r"import", r"trade[_\s]balance"],
    "index_pricing": [
        r"industrial[_\s]production",
        r"retail[_\s]sales",
        r"price(?!.*unemployment)",
        r"index(?!.*unemployment)",
        r"stock",
    ],
    "leading_economic_index": [r"leading[_\s]economic[_\s]index", r"economic[_\s]indicator"],
    "unemployment_rate": [
        r"unemployment",
        r"employment",
        r"labor",
        r"job",
        r"payrolls",
        r"claims",
        r"jobless",
    ],
}

# Priority order for categories
categorias_prioritarias = [
    "unemployment_rate",
    "car_registrations",
    "exports",
    "economics",
    "bond",
    "business_confidence",
    "comm_loans",
    "commodities",
    "consumer_confidence",
    "exchange_rate",
    "leading_economic_index",
    "index_pricing",
]

# Precompile patterns for performance
regex_patterns = {
    cat: re.compile("|".join(f"({p})" for p in patterns), re.IGNORECASE) for cat, patterns in cat_patterns.items()
}


def categorize_column(col_name: str) -> str:
    """Return the detected category for *col_name*."""
    if re.search(r"unemploy|employment|payrolls|claims|jobless", col_name, re.IGNORECASE):
        return "unemployment_rate"

    matching = []
    for category in categorias_prioritarias:
        if regex_patterns[category].search(col_name):
            if category == "unemployment_rate":
                return category
            matching.append(category)

    if matching:
        return matching[0]

    if "_MoM" in col_name or "_YoY" in col_name:
        if any(word.lower() in col_name.lower() for word in ["Production", "Sales", "Index"]):
            return "index_pricing"
        if any(word.lower() in col_name.lower() for word in ["Money", "M2"]):
            return "economics"
    return "Sin categoría"


class CategoryService:
    """Service that generates category rows for the economic dataset."""

    def generate_categories(self) -> bool:
        log_file = settings.log_dir / f"category_service_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        configurar_logging(str(log_file))
        logging.info("Running CategoryService")

        if pd is None:  # pragma: no cover - runtime guard
            logging.error("pandas not available")
            return False

        from sp500_analysis.shared.io.file_utils import read_dataframe, write_dataframe

        input_file = settings.preprocess_dir / "MERGEDEXCELS.xlsx"
        output_file = settings.preprocess_dir / "MERGEDEXCELS_CATEGORIZADO.xlsx"
        diag_file = settings.preprocess_dir / "DIAGNOSTICO_CATEGORIAS.xlsx"

        if not input_file.exists():
            logging.error("El archivo de entrada no existe: %s", input_file)
            return False

        df = read_dataframe(input_file)
        logging.info("Archivo cargado correctamente. Dimensiones: %s", df.shape)

        columnas_originales = list(df.columns)
        columnas_encontradas = [col for col in column_renames if col in columnas_originales]
        columnas_no_encontradas = [col for col in column_renames if col not in columnas_originales]

        if columnas_no_encontradas:
            logging.warning("Las siguientes columnas no se encontraron en el archivo: %s", columnas_no_encontradas)

        if columnas_encontradas:
            logging.info("Columnas a renombrar: %s", columnas_encontradas)
            rename_dict = {k: v for k, v in column_renames.items() if k in columnas_originales}
            df.rename(columns=rename_dict, inplace=True)
        else:
            logging.info("No se encontraron columnas para renombrar.")

        categorias = []
        columnas_sin_categoria = []
        resultados_categorias = []
        for col in df.columns:
            cat = categorize_column(col)
            categorias.append(cat)
            resultados_categorias.append({"Columna": col, "Categoría": cat})
            if cat == "Sin categoría":
                columnas_sin_categoria.append(col)

        df_resultados = pd.DataFrame(resultados_categorias)
        write_dataframe(df_resultados, diag_file)
        logging.info("Resultados detallados de categorización guardados en: %s", diag_file)

        df_categoria = pd.DataFrame([categorias], columns=df.columns)

        cat_counts = {}
        for cat in categorias:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        logging.info("Resumen de categorización:")
        for cat, count in sorted(cat_counts.items()):
            logging.info("  - %s: %s columnas", cat, count)

        if columnas_sin_categoria:
            logging.warning("Hay %s columnas sin categoría.", len(columnas_sin_categoria))

        df_categorizado = pd.concat([df_categoria, df], ignore_index=True)
        write_dataframe(df_categorizado, output_file)
        logging.info("Archivo categorizado guardado correctamente en: %s", output_file)

        return True


def run_categories() -> bool:
    """Entry point used by the CLI or scripts via the DI container."""
    from sp500_analysis.shared.container import container, setup_container

    setup_container()
    service: CategoryService = container.resolve("category_service")
    return service.generate_categories()
