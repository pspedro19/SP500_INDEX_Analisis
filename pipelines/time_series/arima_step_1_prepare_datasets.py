#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prep Datasets · ARIMA / SARIMAX (Básico & Extendido)
===================================================
Procesa los archivos <INSTRUMENT>_business_days.csv para SP500, EURUSD y USDJPY.
Aplica el flujo derive -> lag -> log/log1p -> diff -> robust‑scale
y genera un CSV por modelo: <INSTRUMENT>_{ARIMA|SARIMAX_BASIC|SARIMAX_EXTENDED}.csv
Incluye la columna `date` en la salida y no genera filas sintéticas.
"""

import logging
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# 0 · Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("preprocess_arima_sarimax.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1 · Inventario canónico de columnas
# ──────────────────────────────────────────────────────────────────────────────
CANONICAL_COLUMNS = {
    "SP500": {
        "target": "sp500_close",
        "basic": [
            "vix", "term_spread", "cpi_yoy_lag1m", "wti_crude_raw",
            "unemployment_lag1m", "volume_lag1"
        ],
        "extended": [
            "vix", "term_spread", "cpi_yoy_lag1m", "wti_crude_raw",
            "unemployment_lag1m", "volume_lag1", "norm_range",
            "pct_gap", "vol_ma5"
        ],
    },
    "EURUSD": {
        "target": "eurusd_close",
        "basic": [
            "dxy_index_raw", "interest_rate_diff", "eu_unemployment_lag1m",
            "us_unemployment_lag1m", "eu_consumer_confidence_raw"
        ],
        "extended": [
            "dxy_index_raw", "interest_rate_diff", "eu_unemployment_lag1m",
            "us_unemployment_lag1m", "eu_consumer_confidence_raw",
            "norm_range", "pct_gap"
        ],
    },
    "USDJPY": {
        "target": "usdjpy_close",
        "basic": [
            "interest_rate_diff", "japan_leading_lag1m", "us_unemployment_lag1m",
            "nikkei_225_raw", "japan_m2_raw"
        ],
        "extended": [
            "interest_rate_diff", "japan_leading_lag1m", "us_unemployment_lag1m",
            "nikkei_225_raw", "japan_m2_raw", "norm_range", "pct_gap"
        ],
    },
}
DROP_PRIORITY = ["pct_gap", "vol_ma5", "norm_range", "volume_lag1"]

# ──────────────────────────────────────────────────────────────────────────────
# 2 · Limpieza específica SP500
# ──────────────────────────────────────────────────────────────────────────────
def clean_sp500_format(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina comas de miles en SP500 para columnas numéricas."""
    for c in ["sp500_close", "sp500_open", "sp500_high", "sp500_low", "sp500_volume"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            logger.info(f"Limpieza SP500: eliminado separadores de miles en {c}")
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 3 · Utilidades generales
# ──────────────────────────────────────────────────────────────────────────────
def read_csv_with_date_index(path: Path) -> pd.DataFrame:
    """Lee CSV y usa la mejor columna‐fecha como índice."""
    encodings = ["utf-8", "latin1", "cp1252"]
    seps = [",", ";", "\t", "|"]
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                date_col = next(
                    (c for c in df.columns if any(k in c.lower() for k in ("date","fecha","index"))),
                    df.columns[0]
                )
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                if df[date_col].notna().sum() < len(df)*0.5:
                    continue
                df.set_index(date_col, inplace=True)
                df.index.name = "date"
                logger.info(f"CSV leído: {path.name} (enc={enc}, sep='{sep}')")
                return df
            except Exception:
                continue
    logger.warning(f"Lectura heurística fallida para {path.name}; usando pandas.read_csv por defecto")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.index.name = "date"
    return df

def convert_objects_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas objeto a numéricas."""
    for c in df.select_dtypes("object"):
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
            .replace({"nan":np.nan,"NA":np.nan,"":np.nan,"-":np.nan})
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")
        logger.info(f"Convertida a numérica: {c} (NaN: {df[c].isna().sum()})")
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 4 · Transformaciones derive -> lag -> log -> diff
# ──────────────────────────────────────────────────────────────────────────────
def derive_features(df: pd.DataFrame, instr: str) -> pd.DataFrame:
    """Deriva características según instrumento."""
    logger.info(f"Derivando variables para {instr}...")
    
    # Term spread
    if {"ust10y", "fedfunds"}.issubset(df.columns):
        df["term_spread"] = df["ust10y"] - df["fedfunds"]
        logger.info(f"Derivada: term_spread (NaN: {df['term_spread'].isna().sum()})")
    
    # Interest rate differentials
    if instr == "EURUSD" and {"fed_funds", "ecb_rate"}.issubset(df.columns):
        df["interest_rate_diff"] = df["fed_funds"] - df["ecb_rate"]
        logger.info(f"Derivada: interest_rate_diff (EURUSD)")
    if instr == "USDJPY" and {"fed_funds", "boj_rate"}.issubset(df.columns):
        df["interest_rate_diff"] = df["fed_funds"] - df["boj_rate"]
        logger.info(f"Derivada: interest_rate_diff (USDJPY)")

    # Range and gap
    hi, lo, cl, op = (f"{instr.lower()}_{suf}" for suf in ("high", "low", "close", "open"))
    if {hi, lo, cl}.issubset(df.columns):
        df["norm_range"] = (df[hi] - df[lo]) / df[cl] * 100
        logger.info(f"Derivada: norm_range (NaN: {df['norm_range'].isna().sum()})")
    if {op, cl}.issubset(df.columns):
        df["pct_gap"] = (df[op] - df[cl].shift(1)) / df[cl].shift(1) * 100
        logger.info(f"Derivada: pct_gap (NaN: {df['pct_gap'].isna().sum()})")
        
    return df

def apply_lags(df: pd.DataFrame, instr: str) -> pd.DataFrame:
    """Aplica rezagos según reglas de oro."""
    logger.info("Aplicando rezagos...")
    
    # Economic indicators
    if "cpi_yoy_raw" in df:
        df["cpi_yoy_lag1m"] = df["cpi_yoy_raw"].shift(21)  # ~21 días = 1 mes
        logger.info(f"Rezago: cpi_yoy_lag1m (+21 días)")
    if "unemployment_raw" in df:
        df["unemployment_lag1m"] = df["unemployment_raw"].shift(3)
        logger.info(f"Rezago: unemployment_lag1m (+3 días)")
    
    # Instrument specific
    if instr == "EURUSD" and "eu_unemployment_raw" in df:
        df["eu_unemployment_lag1m"] = df["eu_unemployment_raw"].shift(30)
        logger.info(f"Rezago: eu_unemployment_lag1m (+30 días)")
    if instr == "USDJPY" and "japan_leading_indicator_raw" in df:
        df["japan_leading_lag1m"] = df["japan_leading_indicator_raw"].shift(30)
        logger.info(f"Rezago: japan_leading_lag1m (+30 días)")
    
    # Volume
    vol = f"{instr.lower()}_volume"
    if vol in df:
        df[vol] = df[vol].replace(0, np.nan)
        df["volume_lag1"] = df[vol].shift(1)
        logger.info(f"Rezago: volume_lag1 (+1 día)")
        
        df["vol_ma5"] = df[vol].rolling(5).mean().shift(1)
        logger.info(f"Media móvil: vol_ma5 (5 días)")
        
    return df

def apply_log(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica transformaciones logarítmicas."""
    logger.info("Aplicando log/log1p...")
    
    for c in df.columns:
        # Skip already transformed or derived columns
        if any(x in c for x in ("_log", "_d1", "spread", "gap", "range")):
            continue
            
        # Verify positive values
        ser = df[c].dropna()
        if ser.empty or (ser <= 0).any():
            logger.info(f"Omitiendo log para {c} - Contiene valores no positivos")
            continue
            
        # Apply log1p for volume, log for others
        if "volume" in c or "vol_" in c:
            df[f"{c}_log"] = np.log1p(df[c])
            logger.info(f"Log1p: {c}")
        else:
            df[f"{c}_log"] = np.log(df[c])
            logger.info(f"Log: {c}")
            
    return df

def is_stationary(s, alpha=0.05) -> bool:
    """Verifica estacionariedad con ADF y KPSS tests."""
    if len(s) < 20: 
        return False
    try:
        adf_p = adfuller(s, regression="c")[1]
        kpss_p = kpss(s, regression="c", nlags="auto")[1]
        return adf_p < alpha and kpss_p > alpha
    except:
        return False

def apply_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica diferenciación selectiva basada en estacionariedad."""
    logger.info("Aplicando diferenciación selectiva...")
    
    for c in df.columns:
        # Skip already differenced or stationary columns
        if any(x in c for x in ("_d1", "_diff", "spread", "gap", "range")):
            continue
            
        # Prefer log version if exists
        base = f"{c}_log" if f"{c}_log" in df else c
        series = df[base].dropna()
        
        # Check stationarity and apply diff if needed
        if len(series) >= 20:
            if not is_stationary(series):
                df[f"{base}_d1"] = series.diff()
                logger.info(f"Diferenciación: {base} -> {base}_d1")
            else:
                logger.info(f"Serie {base} ya es estacionaria")
        else:
            logger.warning(f"Serie {base} tiene <20 valores válidos ({len(series)})")
            
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 5 · Construcción y limpieza de datasets
# ──────────────────────────────────────────────────────────────────────────────
def _pick(df, col):
    """Selecciona mejor versión transformada disponible."""
    for suf in ("_log_d1", "_d1", "_log", ""):
        if (cand := col + suf) in df.columns:
            return cand
    logger.warning(f"No encontrada transformación para {col}")
    return None

def build_frames(df: pd.DataFrame, instr: str) -> dict:
    """Construye DataFrames para cada modelo."""
    logger.info(f"Construyendo frames para {instr}...")
    
    # Target selection with best transformation
    tgt_raw = CANONICAL_COLUMNS[instr]["target"]
    tgt = _pick(df, tgt_raw) or tgt_raw
    logger.info(f"Target seleccionado: {tgt}")

    # Initialize frames
    frames = {
        "arima": pd.DataFrame(index=df.index),
        "sarimax_basic": pd.DataFrame(index=df.index),
        "sarimax_extended": pd.DataFrame(index=df.index)
    }
    
    # Add target to all models
    for m in frames:
        frames[m][tgt] = df[tgt]
    
    # Add exogenous variables by model type
    for kind in ("basic", "extended"):
        for col in CANONICAL_COLUMNS[instr][kind]:
            if (p := _pick(df, col)):
                frames[f"sarimax_{kind}"][p] = df[p]
                logger.info(f"Variable {p} -> {kind}")
    
    # VIF and parsimony for SARIMAX models
    for key in ("sarimax_basic", "sarimax_extended"):
        frame = frames[key]
        if frame.shape[1] <= 1:
            logger.warning(f"{key} sin variables exógenas")
            continue
            
        # Prepare for VIF calculation
        exog = frame.drop(columns=[tgt])
        
        # Handle NaNs for VIF
        clean = exog.replace([np.inf, -np.inf], np.nan).fillna(exog.mean())
        
        # Remove all-NaN columns
        all_na_cols = clean.isna().all()
        if all_na_cols.any():
            bad_cols = all_na_cols[all_na_cols].index.tolist()
            logger.warning(f"Columnas con todos NaN: {bad_cols}")
            clean = clean.drop(columns=bad_cols)
            exog = exog.drop(columns=bad_cols)
        
        # VIF loop
        while clean.shape[1] >= 2:
            try:
                # Calculate VIF
                vif = pd.Series(
                    [variance_inflation_factor(clean.values, i) for i in range(clean.shape[1])],
                    index=clean.columns
                ).sort_values(ascending=False)
                
                # Accept if all VIF <= 5
                if vif.iloc[0] <= 5:
                    logger.info(f"VIF aceptable para {key}")
                    break
                
                # Remove highest VIF
                bad = vif.index[0]
                logger.warning(f"{key}: VIF {vif.iloc[0]:.2f} -> eliminar {bad}")
                exog.drop(columns=[bad], inplace=True)
                clean.drop(columns=[bad], inplace=True)
            except Exception as e:
                logger.error(f"Error VIF: {e}")
                break
        
        # Parsimony check
        N = len(frame)
        maxf = int(np.sqrt(N))
        
        # Remove variables to maintain parsimony
        while exog.shape[1] > maxf:
            # Find by priority
            drop = next((d for d in DROP_PRIORITY if any(d in c for c in exog.columns)), None)
            if not drop:
                drop = exog.columns[-1]
                
            drop_col = next((c for c in exog.columns if drop in c), exog.columns[-1])
            logger.warning(f"{key} k={exog.shape[1]} > √N={maxf} -> eliminar {drop_col}")
            exog.drop(columns=[drop_col], inplace=True)
        
        # Apply robust scaling
        if exog.shape[1]:
            try:
                scaler = RobustScaler()
                to_scale = exog.fillna(exog.mean())
                scaled = scaler.fit_transform(to_scale)
                exog.loc[:, :] = pd.DataFrame(
                    scaled, index=exog.index, columns=exog.columns
                )
                logger.info(f"Escalado aplicado a {len(exog.columns)} vars en {key}")
            except Exception as e:
                logger.error(f"Error en escalado: {e}")
        
        # Update model frame
        frames[key] = pd.concat([frame[[tgt]], exog], axis=1)
        logger.info(f"Frame final {key}: {frames[key].shape}")
    
    return frames

def clean_dataset(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Limpia dataset sin eliminar demasiados datos."""
    logger.info(f"Limpiando dataset {model_name}: {df.shape}")
    
    if df.empty:
        logger.warning(f"DataFrame vacío para {model_name}")
        return df
    
    # Get target column (first column)
    tgt = df.columns[0]
    
    # Find first valid target
    start = df[tgt].first_valid_index()
    if start is None:
        logger.warning(f"No hay valores válidos en target para {model_name}")
        return df  # Return empty frame with original columns
    
    # Start from first valid target
    subset = df.loc[start:].copy()
    
    # Calculate NaN percentage by row
    na_pct = subset.isna().sum(axis=1) / len(subset.columns)
    
    # Keep rows with less than 20% NaN
    result = subset[na_pct <= 0.2].copy()
    logger.info(f"Tras filtrar >20% NaN: {len(subset)} -> {len(result)} filas")
    
    if not result.empty:
        # Fill remaining NaNs with forward fill
        na_before = result.isna().sum().sum()
        result = result.ffill()
        na_after = result.isna().sum().sum()
        
        # Fill remaining NaNs with column means
        if na_after > 0:
            col_means = result.mean()
            for col in result.columns:
                mask = result[col].isna()
                if mask.any():
                    result.loc[mask, col] = col_means[col]
            na_final = result.isna().sum().sum()
            logger.info(f"NaN imputados: {na_before} -> {na_after} -> {na_final}")
    else:
        # If all rows have too many NaNs, keep rows with valid target
        logger.warning(f"Todas las filas tienen >20% NaN, usando solo filas con target válido")
        result = subset.dropna(subset=[tgt])
        
        if not result.empty:
            result = result.ffill()
            # Fill remaining with column means
            col_means = result.mean()
            for col in result.columns:
                mask = result[col].isna()
                if mask.any():
                    result.loc[mask, col] = col_means[col]
            logger.info(f"Limpieza mínima: {len(result)} filas con target válido")
    
    return result

# ──────────────────────────────────────────────────────────────────────────────
# 6 · Guardado con incluir columna date
# ──────────────────────────────────────────────────────────────────────────────
def save_retry(df: pd.DataFrame, path: Path, retries=3):
    """Guarda DataFrame incluyendo la columna date."""
    # Reset index para incluir fecha en columna
    out = df.reset_index().rename(columns={df.index.name or "index":"date"})
    for i in range(retries):
        try:
            if out.empty:
                logger.warning(f"{path.name} vacío, no se generan filas sintéticas.")
                # Guardamos un dataframe vacío con las columnas correctas
                cols = ["date"] + df.columns.tolist()
                out = pd.DataFrame(columns=cols)
            out.to_csv(path, index=False)
            logger.info(f"Guardado: {path.name} ({len(out)} filas × {len(out.columns)} cols)")
            return
        except PermissionError:
            path = path.with_name(f"{path.stem}_v{i+2}{path.suffix}")
            logger.warning(f"Bloqueado, intentando con {path.name}")
    logger.error(f"No se pudo guardar {path.name}")

# ──────────────────────────────────────────────────────────────────────────────
# 7 · Proceso completo por instrumento
# ──────────────────────────────────────────────────────────────────────────────
def process(instrument: str, src: Path, out_dir: Path):
    """Procesa un instrumento completo."""
    logger.info(f" Procesando {instrument}")
    
    # Lectura y verificación
    df = read_csv_with_date_index(src)
    if df.empty:
        logger.error(f"{instrument}: archivo vacío")
        return False

    # Limpieza específica SP500
    if instrument == "SP500":
        df = clean_sp500_format(df)
        logger.info(f"Aplicada limpieza especial de formato para SP500")

    # Conversión y transformaciones
    df = convert_objects_to_numeric(df)
    logger.info(f"DataFrame inicial: {df.shape[0]} filas × {df.shape[1]} columnas")

    # Verificar presencia del target
    target = CANONICAL_COLUMNS[instrument]["target"]
    if target not in df.columns:
        logger.error(f"Target {target} no encontrado en {instrument}")
        return False
    
    target_na = df[target].isna().sum()
    target_pct = target_na / len(df) * 100
    logger.info(f"Target {target}: {len(df)-target_na}/{len(df)} valores válidos ({target_pct:.2f}% NaN)")

    # Aplicar pipeline
    df = (df.pipe(derive_features, instrument)
            .pipe(apply_lags, instrument)
            .pipe(apply_log)
            .pipe(apply_diff))
    
    logger.info(f"DataFrame transformado: {df.shape[0]} filas × {df.shape[1]} columnas")

    # Construir frames para modelos
    frames = build_frames(df, instrument)
    
    # Crear directorio del instrumento
    inst_dir = out_dir / instrument
    inst_dir.mkdir(parents=True, exist_ok=True)

    # Procesar y guardar cada modelo
    for model, frame in frames.items():
        # Limpiar dataset
        clean_df = clean_dataset(frame, f"{instrument}_{model}")
        
        # Verificar resultados de limpieza
        rows_before = len(frame)
        rows_after = len(clean_df)
        lost_pct = (rows_before - rows_after) / rows_before * 100 if rows_before > 0 else 0
        
        logger.info(f"Resultado limpieza {model}: {rows_before} -> {rows_after} filas ({lost_pct:.2f}% eliminadas)")
        
        # Guardar con índice de fecha como columna
        save_path = inst_dir / f"{instrument}_{model.upper()}.csv"
        save_retry(clean_df, save_path)
        logger.info(f"Guardado dataset {instrument}_{model.upper()}.csv")

    logger.info(f" {instrument} procesado con éxito")
    return True

def main():
    """Procesa todos los instrumentos."""
    logger.info(" Iniciando preparación de datos para modelos ARIMA/SARIMAX")
    
    input_dir = Path(".")
    output_dir = Path("../data/1_preprocess_ts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mapeo de instrumentos a archivos
    mapping = {
        "SP500":  "SP500_business_days.csv",
        "EURUSD": "EURUSD_business_days.csv",
        "USDJPY": "USDJPY_business_days.csv",
    }

    # Buscar archivos en diferentes ubicaciones
    for inst, fname in mapping.items():
        paths = [
            Path(f"./{fname}"),
            Path(f"../data/1_preprocess_ts/{fname}"),
            Path(f"./data/1_preprocess_ts/{fname}"),
        ]
        
        path = next((p for p in paths if p.exists()), None)
        if not path:
            logger.error(f"No se encontró {fname}")
            continue
            
        logger.info(f"Procesando {inst} con {path}")
        process(inst, path, output_dir)
    
    logger.info(" Procesamiento completo")

if __name__ == "__main__":
    main()