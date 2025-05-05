#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prep Datasets · ARIMA / SARIMAX (Básico & Extendido) con Transformaciones Inversas
================================================================
Procesa los archivos <INSTRUMENT>_business_days.csv para SP500, EURUSD y USDJPY.
Aplica el flujo derive -> lag -> log/log1p -> diff -> robust‑scale
y genera un CSV por modelo: <INSTRUMENT>_{ARIMA|SARIMAX_BASIC|SARIMAX_EXTENDED}.csv
Incluye la columna `date` en la salida y no genera filas sintéticas.
Añade columnas inversas para validar cada transformación.
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
    # Primero intentamos con punto y coma que es común en archivos europeos
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
        date_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ("date","fecha","index"))),
            df.columns[0]
        )
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if df[date_col].notna().sum() >= len(df)*0.5:
            df.set_index(date_col, inplace=True)
            df.index.name = "date"
            logger.info(f"CSV leído: {path.name} (enc=utf-8, sep=';')")
            return df
    except Exception as e:
        logger.warning(f"Error al leer con punto y coma: {e}")
    
    # Si falla, intentar con otros separadores
    encodings = ["utf-8", "latin1", "cp1252"]
    seps = [",", "\t", "|"]
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
# 4 · Transformaciones derive -> lag -> log -> diff (Con inversas)
# ──────────────────────────────────────────────────────────────────────────────
def derive_features(df: pd.DataFrame, instr: str) -> pd.DataFrame:
    """Deriva características según instrumento."""
    logger.info(f"Derivando variables para {instr}...")
    
    # Term spread
    if {"ust10y", "fedfunds"}.issubset(df.columns):
        df["term_spread"] = df["ust10y"] - df["fedfunds"]
        # Inversa: reconstituir componentes
        df["term_spread_inv_ust10y"] = df["term_spread"] + df["fedfunds"]
        df["term_spread_inv_fedfunds"] = df["ust10y"] - df["term_spread"]
        logger.info(f"Derivada: term_spread (NaN: {df['term_spread'].isna().sum()}) + inversas")
    
    # Interest rate differentials
    if instr == "EURUSD" and {"fed_funds", "ecb_rate"}.issubset(df.columns):
        df["interest_rate_diff"] = df["fed_funds"] - df["ecb_rate"]
        # Inversas
        df["interest_rate_diff_inv_fed"] = df["interest_rate_diff"] + df["ecb_rate"]
        df["interest_rate_diff_inv_ecb"] = df["fed_funds"] - df["interest_rate_diff"]
        logger.info(f"Derivada: interest_rate_diff (EURUSD) + inversas")

    if instr == "USDJPY" and {"fed_funds", "boj_rate"}.issubset(df.columns):
        df["interest_rate_diff"] = df["fed_funds"] - df["boj_rate"]
        # Inversas
        df["interest_rate_diff_inv_fed"] = df["interest_rate_diff"] + df["boj_rate"]
        df["interest_rate_diff_inv_boj"] = df["fed_funds"] - df["interest_rate_diff"]
        logger.info(f"Derivada: interest_rate_diff (USDJPY) + inversas")

    # Range and gap
    hi, lo, cl, op = (f"{instr.lower()}_{suf}" for suf in ("high", "low", "close", "open"))
    if {hi, lo, cl}.issubset(df.columns):
        df["norm_range"] = (df[hi] - df[lo]) / df[cl] * 100
        # Inversas (reconstrucción parcial ya que hay múltiples soluciones)
        # Aquí reconstruimos high asumiendo que low y close son conocidos
        df["norm_range_inv_high"] = df[lo] + (df["norm_range"] * df[cl] / 100)
        logger.info(f"Derivada: norm_range (NaN: {df['norm_range'].isna().sum()}) + inversa")

    if {op, cl}.issubset(df.columns):
        df["pct_gap"] = (df[op] - df[cl].shift(1)) / df[cl].shift(1) * 100
        # Inversa para reconstruir open conociendo el close anterior
        df["pct_gap_inv_open"] = df[cl].shift(1) * (1 + df["pct_gap"]/100)
        logger.info(f"Derivada: pct_gap (NaN: {df['pct_gap'].isna().sum()}) + inversa")
        
    return df

def apply_lags(df: pd.DataFrame, instr: str) -> pd.DataFrame:
    """Aplica rezagos según reglas de oro."""
    logger.info("Aplicando rezagos...")
    
    # Economic indicators
    if "cpi_yoy_raw" in df:
        df["cpi_yoy_lag1m"] = df["cpi_yoy_raw"].shift(21)  # ~21 días = 1 mes
        # Inversa del rezago (lead)
        df["cpi_yoy_lag1m_inv"] = df["cpi_yoy_lag1m"].shift(-21)
        logger.info(f"Rezago: cpi_yoy_lag1m (+21 días) + inversa")
        
    if "unemployment_raw" in df:
        df["unemployment_lag1m"] = df["unemployment_raw"].shift(3)
        # Inversa del rezago (lead)
        df["unemployment_lag1m_inv"] = df["unemployment_lag1m"].shift(-3)
        logger.info(f"Rezago: unemployment_lag1m (+3 días) + inversa")
    
    # Instrument specific
    if instr == "EURUSD" and "eu_unemployment_raw" in df:
        df["eu_unemployment_lag1m"] = df["eu_unemployment_raw"].shift(30)
        # Inversa del rezago (lead)
        df["eu_unemployment_lag1m_inv"] = df["eu_unemployment_lag1m"].shift(-30)
        logger.info(f"Rezago: eu_unemployment_lag1m (+30 días) + inversa")
        
    if instr == "USDJPY" and "japan_leading_indicator_raw" in df:
        df["japan_leading_lag1m"] = df["japan_leading_indicator_raw"].shift(30)
        # Inversa del rezago (lead)
        df["japan_leading_lag1m_inv"] = df["japan_leading_lag1m"].shift(-30)
        logger.info(f"Rezago: japan_leading_lag1m (+30 días) + inversa")
    
    # Volume
    vol = f"{instr.lower()}_volume"
    if vol in df:
        df[vol] = df[vol].replace(0, np.nan)
        df["volume_lag1"] = df[vol].shift(1)
        # Inversa del rezago (lead)
        df["volume_lag1_inv"] = df["volume_lag1"].shift(-1)
        logger.info(f"Rezago: volume_lag1 (+1 día) + inversa")
        
        df["vol_ma5"] = df[vol].rolling(5).mean().shift(1)
        # No hay inversa exacta de la media móvil, pero guardamos original para comparar
        df["vol_ma5_orig"] = df[vol]
        logger.info(f"Media móvil: vol_ma5 (5 días) + original")
        
    return df

def apply_log(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica transformaciones logarítmicas y añade inversas."""
    logger.info("Aplicando log/log1p con inversas (exp/expm1)...")
    
    for c in df.columns:
        # Skip already transformed or derived columns
        if any(x in c for x in ("_log", "_d1", "spread", "gap", "range", "_inv")):
            continue
            
        # Verify positive values
        ser = df[c].dropna()
        if ser.empty or (ser <= 0).any():
            logger.info(f"Omitiendo log para {c} - Contiene valores no positivos")
            continue
            
        # Apply log1p for volume, log for others
        if "volume" in c or "vol_" in c:
            df[f"{c}_log"] = np.log1p(df[c])
            # Inversa log1p -> expm1
            df[f"{c}_log_inv"] = np.expm1(df[f"{c}_log"])
            logger.info(f"Log1p: {c} + inversa expm1")
        else:
            df[f"{c}_log"] = np.log(df[c])
            # Inversa log -> exp
            df[f"{c}_log_inv"] = np.exp(df[f"{c}_log"])
            logger.info(f"Log: {c} + inversa exp")
            
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
    """Aplica diferenciación selectiva con inversas mejoradas."""
    logger.info("Aplicando diferenciación selectiva con inversas mejoradas...")
    
    # Guardamos las columnas originales para validación
    orig_values = {}
    
    for c in df.columns:
        # Skip already differenced or stationary columns
        if any(x in c for x in ("_d1", "_diff", "spread", "gap", "range", "_inv")):
            continue
        
        # Detect if this is a log-transformed series
        is_log_transform = c.endswith("_log")
        is_log_component = "_log" in c
        
        # Guardar valores originales para validación
        series = df[c].dropna()
        if len(series) > 0:
            orig_values[c] = series.copy()
        
        # Skip short series
        if len(series) < 20:
            logger.warning(f"Serie {c} tiene <20 valores válidos ({len(series)})")
            continue
        
        # Apply diff if needed
        if not is_stationary(series):
            df[f"{c}_d1"] = series.diff()
            
            # Get first valid value for reconstruction
            first_valid = series.first_valid_index()
            if first_valid is None:
                logger.warning(f"No se pudo crear inversa para {c}_d1, no hay valores iniciales")
                continue
            
            # Get initial value - critical for accuracy
            initial_value = series.loc[first_valid]
            logger.info(f"Valor inicial para {c}: {initial_value} en {first_valid}")
            
            # Fill first value (NaN from diff) and compute cumsum
            diff_series = df[f"{c}_d1"].copy()
            diff_series.loc[first_valid] = initial_value
            
            # Llenar NaN correctamente para cumsum
            diff_filled = diff_series.fillna(0).copy()
            cumsum_result = diff_filled.cumsum()
            
            # Apply appropriate inverse transformation
            if is_log_transform or is_log_component:
                # Para series con transformación logarítmica, aplicamos exp
                df[f"{c}_d1_inv"] = np.exp(cumsum_result)
                logger.info(f"Inversión completa (cumsum+exp): {c} -> {c}_d1_inv")
                
                # Validar primeros valores
                first_orig = series.iloc[0] if len(series) > 0 else np.nan
                first_inv = df[f"{c}_d1_inv"].iloc[0] if len(df[f"{c}_d1_inv"]) > 0 else np.nan
                logger.info(f"Validación del primer valor de {c}: Original={first_orig}, Invertido={first_inv}, Error={first_orig-first_inv if pd.notna(first_orig) and pd.notna(first_inv) else 'N/A'}")
            else:
                # Para series regulares, solo cumsum
                df[f"{c}_d1_inv"] = cumsum_result
                logger.info(f"Inversión simple (cumsum): {c} -> {c}_d1_inv")
                
                # Validar primeros valores
                first_orig = series.iloc[0] if len(series) > 0 else np.nan
                first_inv = df[f"{c}_d1_inv"].iloc[0] if len(df[f"{c}_d1_inv"]) > 0 else np.nan
                logger.info(f"Validación del primer valor de {c}: Original={first_orig}, Invertido={first_inv}, Error={first_orig-first_inv if pd.notna(first_orig) and pd.notna(first_inv) else 'N/A'}")
        else:
            logger.info(f"Serie {c} ya es estacionaria, no se aplica diff")
    
    # Validar transformaciones
    validate_transformations(df, orig_values)
    
    return df

def validate_transformations(df: pd.DataFrame, orig_values: dict):
    """Valida las transformaciones y sus inversas."""
    logger.info("Validando transformaciones inversas...")
    
    for c, orig_series in orig_values.items():
        # Buscar columnas de inversión relacionadas
        inv_cols = [col for col in df.columns if col.startswith(f"{c}_") and col.endswith("_inv")]
        
        for inv_col in inv_cols:
            # Calcular error relativo promedio
            common_idx = orig_series.index.intersection(df[inv_col].dropna().index)
            if len(common_idx) > 0:
                orig_vals = orig_series.loc[common_idx]
                inv_vals = df[inv_col].loc[common_idx]
                
                # Calcular error relativo promedio para valores no-NaN
                valid_mask = orig_vals.notna() & inv_vals.notna()
                if valid_mask.sum() > 0:
                    rel_error = ((orig_vals[valid_mask] - inv_vals[valid_mask]) / 
                                orig_vals[valid_mask]).abs().mean()
                    
                    logger.info(f"Error relativo promedio para {c} vs {inv_col}: {rel_error:.6f}")
                    
                    # Mostrar algunos valores específicos
                    for i, idx in enumerate(common_idx[:5]):
                        if i < 5 and pd.notna(orig_vals.loc[idx]) and pd.notna(inv_vals.loc[idx]):
                            err = orig_vals.loc[idx] - inv_vals.loc[idx]
                            rel_err = err / orig_vals.loc[idx] if abs(orig_vals.loc[idx]) > 1e-10 else np.nan
                            logger.info(f"  Validación {idx}: Original={orig_vals.loc[idx]:.6f}, Invertido={inv_vals.loc[idx]:.6f}, Error abs={err:.6f}, Error rel={rel_err:.2%}")

# ──────────────────────────────────────────────────────────────────────────────
# 5 · Construcción y limpieza de datasets (con inversas para RobustScaler)
# ──────────────────────────────────────────────────────────────────────────────
def _pick(df, col):
    """Selecciona mejor versión transformada disponible."""
    for suf in ("_log_d1", "_d1", "_log", ""):
        if (cand := col + suf) in df.columns:
            return cand
    logger.warning(f"No encontrada transformación para {col}")
    return None

def build_frames(df: pd.DataFrame, instr: str) -> dict:
    """Construye DataFrames para cada modelo con columnas de inversa para escalado."""
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
        # Añadir inversa del target si existe
        if f"{tgt}_inv" in df.columns:
            frames[m][f"{tgt}_inv"] = df[f"{tgt}_inv"]
            logger.info(f"Añadida inversa {tgt}_inv al frame {m}")
    
    # Add exogenous variables by model type
    for kind in ("basic", "extended"):
        for col in CANONICAL_COLUMNS[instr][kind]:
            if (p := _pick(df, col)):
                frames[f"sarimax_{kind}"][p] = df[p]
                # Añadir inversas si existen
                if f"{p}_inv" in df.columns:
                    frames[f"sarimax_{kind}"][f"{p}_inv"] = df[f"{p}_inv"]
                    logger.info(f"Variable {p} -> {kind} (+ inversa {p}_inv)")
                else:
                    logger.info(f"Variable {p} -> {kind} (sin inversa)")
    
    # VIF and parsimony for SARIMAX models
    for key in ("sarimax_basic", "sarimax_extended"):
        frame = frames[key]
        
        # Separar columnas originales de inversas
        orig_cols = [c for c in frame.columns if not c.endswith("_inv")]
        inv_cols = [c for c in frame.columns if c.endswith("_inv")]
        
        if len(orig_cols) <= 1:
            logger.warning(f"{key} sin variables exógenas")
            continue
            
        # Prepare for VIF calculation
        exog = frame[orig_cols].drop(columns=[tgt])
        
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
                
                # También eliminar la columna inversa si existe
                bad_inv = f"{bad}_inv"
                if bad_inv in inv_cols:
                    inv_cols.remove(bad_inv)
                    logger.info(f"Eliminado {bad_inv} debido a VIF alto de {bad}")
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
            
            # También eliminar la columna inversa si existe
            drop_inv = f"{drop_col}_inv"
            if drop_inv in inv_cols:
                inv_cols.remove(drop_inv)
                logger.info(f"Eliminado {drop_inv} debido a restricción de parsimonia")
        
        # Apply robust scaling with storage of inverse transform
        if exog.shape[1]:
            try:
                # Guardar las columnas originales antes del escalado
                for col in exog.columns:
                    exog[f"{col}_pre_scale"] = exog[col].copy()
                    logger.info(f"Guardado valor pre-escala para {col}")
                
                # Aplicar escalado
                scaler = RobustScaler()
                to_scale = exog.fillna(exog.mean())
                scaled = scaler.fit_transform(to_scale)
                
                # Guardamos el centro y escala para cada columna (para la inversa)
                scale_params = pd.DataFrame({
                    'column': exog.columns,
                    'center': scaler.center_,
                    'scale': scaler.scale_
                })
                logger.info(f"Parámetros de escalado: {scale_params}")
                
                # Aplicar el escalado
                scaled_df = pd.DataFrame(
                    scaled, index=exog.index, columns=exog.columns
                )
                
                # Añadir columnas originales e inversas del escalado
                for i, col in enumerate(exog.columns):
                    # Aplicar la transformación inversa del escalado
                    center = scaler.center_[i]
                    scale = scaler.scale_[i]
                    
                    # La inversa del escalado robusto es x_scaled * scale + center
                    scaled_df[f"{col}_unscaled"] = scaled_df[col] * scale + center
                    logger.info(f"Creada inversa de escalado para {col}: center={center}, scale={scale}")
                
                # Reemplazar las columnas originales con las escaladas
                exog.loc[:, exog.columns] = scaled_df[exog.columns]
                
                # Añadir columnas de valores pre-escalado e inversa de escalado
                for col in exog.columns:
                    exog[f"{col}_unscaled"] = scaled_df[f"{col}_unscaled"]
                
                logger.info(f"Escalado aplicado a {len(exog.columns)} vars en {key} + inversas")
                
                # Validar escalado
                for col in exog.columns:
                    if f"{col}_pre_scale" in exog.columns and f"{col}_unscaled" in exog.columns:
                        # Calcular error de escalado inverso
                        valid_mask = exog[f"{col}_pre_scale"].notna() & exog[f"{col}_unscaled"].notna()
                        if valid_mask.sum() > 0:
                            error = (exog.loc[valid_mask, f"{col}_pre_scale"] - 
                                     exog.loc[valid_mask, f"{col}_unscaled"]).abs().mean()
                            logger.info(f"Error promedio de inversa de escalado para {col}: {error:.9f}")
            except Exception as e:
                logger.error(f"Error en escalado: {e}")
        
        # Update model frame
        # Unir el target, las variables exógenas escaladas y las columnas inversas
        frames[key] = pd.concat([
            frame[[tgt]], 
            exog,
            frame[[c for c in inv_cols if c in frame.columns]]
        ], axis=1)
        
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
    logger.info(f"Primera fecha con target válido: {start}")
    
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
    
    # Verificación final de transformaciones inversas
    validate_final_inversions(result)
    
    return result

def validate_final_inversions(df: pd.DataFrame):
    """Valida las inversiones finales en el dataset limpio."""
    logger.info("Validación final de inversiones:")
    
    # Buscar pares de columnas originales e inversas
    orig_inv_pairs = []
    for col in df.columns:
        if col.endswith("_d1"):
            inv_col = f"{col}_inv"
            if inv_col in df.columns:
                orig_inv_pairs.append((col, inv_col))
    
    # Verificar cada par
    for orig_col, inv_col in orig_inv_pairs:
        # Si es una transformación logarítmica
        if "_log_d1" in orig_col:
            base_col = orig_col.replace("_log_d1", "")
            # Buscar el valor original si existe
            if base_col in df.columns:
                # Comparar algunos valores específicos
                for i in range(min(5, len(df))):
                    idx = df.index[i]
                    if pd.notna(df.loc[idx, base_col]) and pd.notna(df.loc[idx, inv_col]):
                        orig_val = df.loc[idx, base_col]
                        inv_val = df.loc[idx, inv_col]
                        error = orig_val - inv_val
                        rel_error = error / orig_val if abs(orig_val) > 1e-10 else np.nan
                        logger.info(f"Validación {idx} - {base_col} vs {inv_col}: Original={orig_val:.6f}, Invertido={inv_val:.6f}, Error={error:.6f}, Error Rel={rel_error:.2%}")
        
        # Verificación específica para inversiones conocidas
        if "eurusd_close_log_d1_inv" in df.columns:
            for date_str in ["2014-01-02", "2014-01-03"]:
                if date_str in df.index:
                    val = df.loc[date_str, "eurusd_close_log_d1_inv"]
                    logger.info(f"EURUSD {date_str}: {val:.6f}")
                else:
                    # Buscar fecha similar
                    similar_dates = [idx for idx in df.index if str(idx).startswith(date_str[:7])]
                    if similar_dates:
                        for sim_date in similar_dates[:2]:
                            val = df.loc[sim_date, "eurusd_close_log_d1_inv"]
                            logger.info(f"EURUSD {sim_date}: {val:.6f}")
        
        if "dxy_index_raw_log_d1_inv" in df.columns:
            for date_str in ["2014-01-02", "2014-01-03"]:
                if date_str in df.index:
                    val = df.loc[date_str, "dxy_index_raw_log_d1_inv"]
                    logger.info(f"DXY {date_str}: {val:.6f}")
                else:
                    # Buscar fecha similar
                    similar_dates = [idx for idx in df.index if str(idx).startswith(date_str[:7])]
                    if similar_dates:
                        for sim_date in similar_dates[:2]:
                            val = df.loc[sim_date, "dxy_index_raw_log_d1_inv"]
                            logger.info(f"DXY {sim_date}: {val:.6f}")

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
    
    # Guardar valores originales para validación final
    orig_values = {}
    for key_col in [target] + CANONICAL_COLUMNS[instrument]["basic"]:
        if key_col in df.columns:
            orig_values[key_col] = df[key_col].copy()
            logger.info(f"Guardado valor original para {key_col}")

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
        
        # Validación final para fechas específicas
        validate_specific_dates(clean_df, orig_values, instrument)
        
        # Guardar con índice de fecha como columna
        save_path = inst_dir / f"{instrument}_{model.upper()}.csv"
        save_retry(clean_df, save_path)
        logger.info(f"Guardado dataset {instrument}_{model.upper()}.csv")

    logger.info(f" {instrument} procesado con éxito")
    return True

def validate_specific_dates(df, orig_values, instrument):
    """Validación específica para fechas importantes."""
    logger.info("Validación para fechas específicas:")
    
    # Fechas clave a validar
    key_dates = ["2014-01-02", "2014-01-03"]
    
    for date_str in key_dates:
        if date_str in df.index:
            logger.info(f"Validación {date_str}:")
            
            # Para EURUSD, validar el tipo de cambio
            if instrument == "EURUSD":
                if "eurusd_close_log_d1_inv" in df.columns:
                    inv_val = df.loc[date_str, "eurusd_close_log_d1_inv"]
                    
                    # Buscar fecha equivalente en original
                    orig_date = None
                    for idx in orig_values.get("eurusd_close", pd.Series()).index:
                        if str(idx).replace('/', '-').endswith(date_str[5:]):
                            orig_date = idx
                            break
                    
                    if orig_date is not None and orig_date in orig_values.get("eurusd_close", pd.Series()).index:
                        orig_val = orig_values["eurusd_close"].loc[orig_date]
                        error = orig_val - inv_val if pd.notna(orig_val) and pd.notna(inv_val) else np.nan
                        logger.info(f"  EURUSD: Fecha orig={orig_date}, orig={orig_val}, inv={inv_val}, error={error}")
            
            # Para todos los instrumentos, validar DXY si está disponible
            if "dxy_index_raw_log_d1_inv" in df.columns:
                inv_val = df.loc[date_str, "dxy_index_raw_log_d1_inv"]
                
                # Buscar fecha equivalente en original
                orig_date = None
                for idx in orig_values.get("dxy_index_raw", pd.Series()).index:
                    if str(idx).replace('/', '-').endswith(date_str[5:]):
                        orig_date = idx
                        break
                
                if orig_date is not None and orig_date in orig_values.get("dxy_index_raw", pd.Series()).index:
                    orig_val = orig_values["dxy_index_raw"].loc[orig_date]
                    error = orig_val - inv_val if pd.notna(orig_val) and pd.notna(inv_val) else np.nan
                    logger.info(f"  DXY: Fecha orig={orig_date}, orig={orig_val}, inv={inv_val}, error={error}")

def main():
    """Procesa todos los instrumentos."""
    logger.info(" Iniciando preparación de datos para modelos ARIMA/SARIMAX con transformaciones inversas mejoradas")
    
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
    
    logger.info(" Procesamiento completo con transformaciones inversas mejoradas")

if __name__ == "__main__":
    main()