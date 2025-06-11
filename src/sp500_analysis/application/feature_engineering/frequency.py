from __future__ import annotations

import re

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional
    pd = None


COLUMN_FREQUENCIES = {
    "PRICE_Australia_10Y_Bond_bond": "D",
    "PRICE_Italy_10Y_Bond_bond": "D",
    "PRICE_Japan_10Y_Bond_bond": "D",
    "PRICE_UK_10Y_Bond_bond": "D",
    "PRICE_Germany_10Y_Bond_bond": "D",
    "PRICE_Canada_10Y_Bond_bond": "D",
    "PRICE_China_10Y_Bond_bond": "D",
    "PRICE_CrudeOil_WTI_commodities": "D",
    "PRICE_Gold_Spot_commodities": "D",
    "PRICE_Silver_Spot_commodities": "D",
    "PRICE_Copper_Futures_commodities": "D",
    "PRICE_Platinum_Spot_commodities": "D",
    "PRICE_EUR_USD_Spot_exchange_rate": "D",
    "PRICE_GBP_USD_Spot_exchange_rate": "D",
    "PRICE_JPY_USD_Spot_exchange_rate": "D",
    "PRICE_CNY_USD_Spot_exchange_rate": "D",
    "PRICE_AUD_USD_Spot_exchange_rate": "D",
    "PRICE_CAD_USD_Spot_exchange_rate": "D",
    "PRICE_MXN_USD_Spot_exchange_rate": "D",
    "PRICE_EUR_GBP_Cross_exchange_rate": "D",
    "PRICE_USD_COP_Spot_exchange_rate": "D",
    "ULTIMO_S&P500_Index_index_pricing": "D",
    "ULTIMO_NASDAQ_Composite_index_pricing": "D",
    "ULTIMO_Russell_2000_index_pricing": "D",
    "ULTIMO_FTSE_100_index_pricing": "D",
    "ULTIMO_Nikkei_225_index_pricing": "D",
    "ULTIMO_DAX_30_index_pricing": "D",
    "PRICE_Shanghai_Composite_index_pricing": "D",
    "ULTIMO_VIX_VolatilityIndex_index_pricing": "D",
    "ESI_GACDISA_US_Empire_State_Index_business_confidence": "M",
    "ESI_AWCDISA_US_Empire_State_Index_business_confidence": "M",
    "AAII_Bearish_AAII_Investor_Sentiment_consumer_confidence": "W",
    "AAII_Bull-Bear Spread_AAII_Investor_Sentiment_consumer_confidence": "W",
    "AAII_Bullish_AAII_Investor_Sentiment_consumer_confidence": "W",
    "PutCall_strike_Put_Call_Ratio_SPY_consumer_confidence": "W",
    "PutCall_bid_Put_Call_Ratio_SPY_consumer_confidence": "W",
    "PutCall_ask_Put_Call_Ratio_SPY_consumer_confidence": "W",
    "PutCall_vol_Put_Call_Ratio_SPY_consumer_confidence": "W",
    "PutCall_delta_Put_Call_Ratio_SPY_consumer_confidence": "W",
    "PutCall_gamma_Put_Call_Ratio_SPY_consumer_confidence": "W",
    "PutCall_theta_Put_Call_Ratio_SPY_consumer_confidence": "W",
    "PutCall_vega_Put_Call_Ratio_SPY_consumer_confidence": "W",
    "PutCall_rho_Put_Call_Ratio_SPY_consumer_confidence": "W",
    "NFCI_Chicago_Fed_NFCI_leading_economic_index": "M",
    "ANFCI_Chicago_Fed_NFCI_leading_economic_index": "M",
    "Actual_US_ISM_Manufacturing_business_confidence": "M",
    "Actual_US_ISM_Services_business_confidence": "M",
    "Actual_US_Philly_Fed_Index_business_confidence": "M",
    "Actual_France_Business_Climate_business_confidence": "M",
    "Actual_EuroZone_Business_Climate_business_confidence": "M",
    "Actual_US_Consumer_Confidence_consumer_confidence": "M",
    "Actual_China_PMI_Manufacturing_economics": "Q",
    "Actual_US_ConferenceBoard_LEI_leading_economic_index": "D",
    "Actual_Japan_Leading_Indicator_leading_economic_index": "D",
    "DGS10_US_10Y_Treasury_bond": "M",
    "DGS2_US_2Y_Treasury_bond": "M",
    "AAA_Corporate_Bond_AAA_Spread_bond": "D",
    "BAA10YM_Corporate_Bond_BBB_Spread_bond": "M",
    "BAMLH0A0HYM2_High_Yield_Bond_Spread_bond": "M",
    "DNKSLRTCR03GPSAM_Denmark_Car_Registrations_MoM": "M",
    "USASLRTCR03GPSAM_US_Car_Registrations_MoM": "M",
    "ZAFSLRTCR03GPSAM_SouthAfrica_Car_Registrations_MoM": "M",
    "GBRSLRTCR03GPSAM_United_Kingdom_Car_Registrations_MoM": "M",
    "ESPSLRTCR03GPSAM_Spain_Car_Registrations_MoM": "M",
    "BUSLOANS_US_Commercial_Loans_comm_loans": "M",
    "CREACBM027NBOG_US_RealEstate_Commercial_Loans_comm_loans": "M",
    "TOTALSL_US_Consumer_Credit_comm_loans": "M",
    "CSCICP02EZM460S_EuroZone_Consumer_Confidence_consumer_confidence": "M",
    "CSCICP02CHQ460S_Switzerland_Consumer_Confidence_consumer_confidence": "M",
    "UMCSENT_Michigan_Consumer_Sentiment_consumer_confidence": "M",
    "CPIAUCSL_US_CPI_economics": "M",
    "CPILFESL_US_Core_CPI_economics": "M",
    "PCE_US_PCE_economics": "M",
    "PCEPILFE_US_Core_PCE_economics": "M",
    "PPIACO_US_PPI_economics": "M",
    "INDPRO_US_Industrial_Production_MoM_economics": "M",
    "CSUSHPINSA_US_CaseShiller_HomePrice_economics": "M",
    "GDP_US_GDP_Growth_economics": "M",
    "TCU_US_Capacity_Utilization_economics": "M",
    "PERMIT_US_Building_Permits_economics": "M",
    "HOUST_US_Housing_Starts_economics": "M",
    "FEDFUNDS_US_FedFunds_Rate_economics": "M",
    "ECBDFR_ECB_Deposit_Rate_economics": "M",
    "WALCL_Fed_Balance_Sheet_economics": "M",
    "Price_Dollar_Index_DXY_index_pricing": "M",
    "PRICE_US_Unemployment_Rate_unemployment_rate": "M",
    "PRICE_US_Nonfarm_Payrolls_unemployment_rate": "M",
    "PRICE_US_Initial_Jobless_Claims_unemployment_rate": "M",
    "PRICE_US_JOLTS_unemployment_rate": "M",
    "Actual_Eurozone_Unemployment_Rate_unemployment_rate": "M",
    "UNRATE_US_Unemployment_Rate_unemployment_rate": "M",
    "PAYEMS_US_Nonfarm_Payrolls_unemployment_rate": "M",
    "ICSA_US_Initial_Jobless_Claims_unemployment_rate": "M",
    "DGS10_DGS10_bond": "M",
}


def infer_frequencies(df: pd.DataFrame) -> dict[str, str]:
    """Return a frequency mapping for all columns."""
    frequencies = COLUMN_FREQUENCIES.copy()
    for col in df.columns:
        if col in {"date", "id"}:
            continue
        if col not in frequencies:
            name = col.lower()
            if any(k in name for k in ["actual_", "rate", "unemployment", "cpi", "ppi", "gdp", "ism", "confidence"]):
                frequencies[col] = "M"
            elif "price_" in name and any(m in name for m in ["bond", "spot", "index", "composite"]):
                frequencies[col] = "D"
            else:
                frequencies[col] = "M"
    return frequencies


def rename_dataframe(
    dataset: pd.DataFrame, datetime_column: str, target_columns: str | None, date_format: str | None
) -> pd.DataFrame:
    renamed = dataset.rename(columns={datetime_column: "date"})
    renamed = renamed[renamed["date"].apply(lambda x: isinstance(x, str) and not re.search(r"[a-zA-Z]", x))]
    if date_format:
        renamed["date"] = pd.to_datetime(renamed["date"], format=date_format).dt.strftime("%Y-%m-%d")
    else:
        renamed["date"] = pd.to_datetime(renamed["date"], infer_datetime_format=True).dt.strftime("%Y-%m-%d")

    first_col = dataset.columns[0]
    if first_col != datetime_column:
        renamed = renamed.rename(columns={first_col: "id"})
    else:
        renamed.insert(1, "id", "serie_default")

    cols_to_keep = ["date", "id"]
    if target_columns:
        cols_to_keep.extend(target_columns.split(","))
    return renamed[cols_to_keep]


def impute_time_series_ffill(
    dataset: pd.DataFrame, datetime_column: str = "date", id_column: str = "id"
) -> pd.DataFrame:
    df = dataset.copy()
    if df.isnull().values.any():
        df = df.ffill()
    return df


def resample_to_business_day(
    dataset: pd.DataFrame,
    input_frequency: str,
    column_date: str = "date",
    id_column: str = "id",
    output_frequency: str = "B",
) -> pd.DataFrame:
    df = dataset.copy()
    df[column_date] = pd.to_datetime(df[column_date], format="%Y-%m-%d")
    df = df.drop_duplicates(subset=[column_date], keep="last")
    df = df.set_index(column_date).sort_index()
    if input_frequency != output_frequency and len(df) > 0:
        resampled = df.asfreq(freq=output_frequency, method="ffill")
    else:
        resampled = df.copy()
    resampled = resampled.reset_index()
    resampled[column_date] = resampled[column_date].dt.strftime("%Y-%m-%d")
    return resampled


def convert_dataframe(
    df: pd.DataFrame, excluded_column: str | None, id_column: str = "id", datetime_column: str = "date"
) -> pd.DataFrame:
    cols = df.columns.difference([id_column, datetime_column] + ([excluded_column] if excluded_column else []))
    for col in cols:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False).str.replace(r"[^\d\.-]", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
