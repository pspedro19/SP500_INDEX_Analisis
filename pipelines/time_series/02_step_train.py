#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento de Modelos ARIMA/SARIMAX
======================================
Entrena modelos para cada instrumento (SP500, EURUSD, USDJPY) con los
datasets preparados. Implementa enfoque híbrido de optimización de
hiperparámetros y validación walk-forward siguiendo las 12 reglas de oro.

Salida:
- Modelos entrenados en formato .pkl
- Archivo de predicciones para todos los modelos (all_models_predictions.csv)
- Métricas y resultados de validación
- Visualizaciones de modelos y pronósticos
"""

import logging
from sp500_analysis.shared.logging.logger import configurar_logging, setup_logging, get_logger
import os
import json
import pickle
import random
import numpy as np
import pandas as pd
import optuna
import pmdarima as pm
import yaml
import traceback
import sys
from pathlib import Path
from datetime import datetime
from math import sqrt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
import statsmodels.api as sm
import warnings
from joblib import Parallel, delayed, dump, load

# Suprimir advertencias
warnings.filterwarnings("ignore")

# Importar utilidades de visualización
from sp500_analysis.shared.visualization import plotters as plots
# CONFIGURACIÓN Y ESTRUCTURAS DE DATOS
# ====================================


@dataclass
class Config:
    """Configuración centralizada del pipeline."""

    # Rutas del proyecto
    project_root: Path
    input_dir: Path
    output_dir: Path
    models_dir: Path
    logs_dir: Path

    # Parámetros globales
    random_seed: int = 42
    forecast_horizon: int = 21  # Días hábiles (1 mes)

    # Instrumentos y tipos de modelos
    instruments: List[str] = field(default_factory=lambda: ["SP500", "EURUSD", "USDJPY"])
    model_types: List[str] = field(default_factory=lambda: ["ARIMA", "SARIMAX_BASIC", "SARIMAX_EXTENDED"])

    # Optimización de hiperparámetros
    optimization_method: str = "hybrid"
    max_trials: int = 50
    cv_splits: int = 5
    test_size: int = 21
    n_jobs: int = -1  # -1 para usar todos los núcleos disponibles

    # Parámetros ARIMA/SARIMAX
    arima_start_p: int = 0
    arima_start_q: int = 0
    arima_max_p: int = 3
    arima_max_q: int = 3
    arima_start_P: int = 0
    arima_start_Q: int = 0
    arima_max_P: int = 1
    arima_max_Q: int = 1
    seasonal_period: int = 5  # 5 días hábiles = 1 semana

    def __post_init__(self):
        """Convertir strings a Path si es necesario."""
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)
        if isinstance(self.input_dir, str):
            self.input_dir = Path(self.input_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.models_dir, str):
            self.models_dir = Path(self.models_dir)
        if isinstance(self.logs_dir, str):
            self.logs_dir = Path(self.logs_dir)

    @classmethod
    def from_yaml(cls, yaml_file: Optional[str] = None) -> 'Config':
        """Carga configuración desde un archivo YAML."""
        if yaml_file and os.path.exists(yaml_file):
            with open(yaml_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)

        # Configuración por defecto si no se proporciona archivo
        project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
        return cls(
            project_root=project_root,
            input_dir=project_root / "data" / "1_preprocess_ts",
            output_dir=project_root / "data" / "2_trainingdata_ts",
            models_dir=project_root / "models",
            logs_dir=project_root / "logs",
        )


@dataclass
class ModelParameters:
    """Parámetros de modelo ARIMA/SARIMAX."""

    order: Tuple[int, int, int]
    seasonal_order: Optional[Tuple[int, int, int, int]] = None
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convierte los parámetros a un diccionario."""
        result = {"p": self.order[0], "d": self.order[1], "q": self.order[2]}

        if self.seasonal_order:
            result.update(
                {
                    "P": self.seasonal_order[0],
                    "D": self.seasonal_order[1],
                    "Q": self.seasonal_order[2],
                    "m": self.seasonal_order[3],
                }
            )

        return result


@dataclass
class ModelMetrics:
    """Métricas de evaluación del modelo."""

    rmse: float
    mae: float
    mape: Optional[float] = None
    hit_direction: Optional[float] = None
    naive_rmse: Optional[float] = None
    naive_drift_rmse: Optional[float] = None
    improvement_vs_naive: Optional[float] = None
    improvement_vs_drift: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convierte las métricas a un diccionario."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ====================================
# CONFIGURACIÓN DE LOGGING
# ====================================


# ====================================
# UTILIDADES DE DATOS
# ====================================


def load_data(config: Config, instrument: str, model_type: str) -> Optional[pd.DataFrame]:
    """
    Carga el dataset preprocesado para un instrumento y tipo de modelo.

    Args:
        config: Configuración global
        instrument: Nombre del instrumento (SP500, EURUSD, USDJPY)
        model_type: Tipo de modelo (ARIMA, BASIC, EXTENDED)

    Returns:
        DataFrame con datos cargados o None si hay error
    """
    logger = get_logger(__name__, instrument=instrument, model_type=model_type)

    # Crear lista de posibles rutas a verificar en orden de prioridad
    paths_to_try = [
        # 1. Subdirectorio del instrumento con nombre exacto (ubicación correcta según estructura)
        config.input_dir / instrument / f"{instrument}_{model_type}.csv",
        # 2. Subdirectorio del instrumento con prefijo SARIMAX_
        config.input_dir / instrument / f"{instrument}_SARIMAX_{model_type}.csv",
        # 3. Directorio raíz con nombre exacto (ubicación alternativa)
        config.input_dir / f"{instrument}_{model_type}.csv",
        # 4. Directorio raíz con prefijo SARIMAX_
        config.input_dir / f"{instrument}_SARIMAX_{model_type}.csv",
    ]

    # Probar cada ruta posible
    file_path = None
    for path in paths_to_try:
        if path.exists():
            file_path = path
            logger.info(f"Archivo encontrado: {file_path}")
            break

    # Verificar si encontramos algún archivo
    if file_path is None:
        logger.error(f"Archivo no encontrado para {instrument} - {model_type} después de probar múltiples rutas")
        return None

    # Cargar datos
    try:
        df = pd.read_csv(file_path)

        # Convertir fecha a datetime y establecer como índice
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Ordenar por fecha
        df = df.sort_index()

        # Verificar datos faltantes
        if df.isnull().any().any():
            logger.warning(
                f"Los datos contienen valores faltantes. Columnas afectadas: "
                f"{df.columns[df.isnull().any()].tolist()}"
            )

        logger.info(f"Datos cargados: {len(df)} filas, {df.shape[1]} columnas")
        return df
    except Exception as e:
        logger.error(f"Error cargando datos desde {file_path}: {str(e)}")
        return None


def split_data(df: pd.DataFrame, min_train_size: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide los datos según esquema 7-2-1:
    - 7 años: entrenamiento
    - 2 años: validación
    - 1 año: test

    Args:
        df: DataFrame con datos completos
        min_train_size: Tamaño mínimo para conjunto de entrenamiento

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    logger = get_logger(__name__)

    # Verificar datos suficientes
    if df is None or len(df) < min_train_size:
        raise ValueError(f"Datos insuficientes para división: {len(df)} < {min_train_size}")

    # Calcular tamaños
    total_len = len(df)
    test_size = min(252, int(total_len * 0.1))  # ~1 año o 10% si es menor
    val_size = min(504, int(total_len * 0.2))  # ~2 años o 20% si es menor

    # Asegurar que hay suficientes datos
    if total_len < (val_size + test_size + min_train_size):
        # Ajustar proporciones
        available = total_len - min_train_size
        test_size = max(21, int(available * 0.33))  # Al menos 21 días para test
        val_size = max(42, available - test_size)  # Resto para validación
        logger.warning(
            f"Datos insuficientes, ajustando: train={total_len-(val_size+test_size)}, "
            f"val={val_size}, test={test_size}"
        )

    # Dividir datos
    train_end = total_len - (val_size + test_size)
    val_end = total_len - test_size

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    logger.info(f"Datos divididos: Training={len(df_train)}, Validation={len(df_val)}, Test={len(df_test)}")

    return df_train, df_val, df_test


def generate_forecast_dates(last_date: pd.Timestamp, steps: int, freq: str = 'B') -> pd.DatetimeIndex:
    """
    Genera fechas futuras para pronóstico.

    Args:
        last_date: Última fecha conocida
        steps: Número de fechas a generar
        freq: Frecuencia ('B' para días hábiles)

    Returns:
        DatetimeIndex con fechas futuras
    """
    return pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq=freq)


# ====================================
# ANÁLISIS DE SERIES TEMPORALES
# ====================================


def determine_integration_orders(series: pd.Series, seasonal_period: int = 5) -> Tuple[int, int]:
    """
    Determina los órdenes de integración d y D mediante tests formales.

    Args:
        series: Serie temporal
        seasonal_period: Período estacional (5 para días hábiles semanales)

    Returns:
        tuple: (d, D)
    """
    logger = get_logger(__name__)

    # Determinar d
    try:
        adf_result = adfuller(series, regression='c')
        adf_pvalue = adf_result[1]

        if adf_pvalue > 0.05:  # Serie no estacionaria
            # Probar con primera diferencia
            diff1 = series.diff().dropna()

            if len(diff1) > 10:  # Verificar que hay suficientes datos
                adf_diff1 = adfuller(diff1, regression='c')[1]

                # CORRECCIÓN: Default a d=1, solo usar d=2 si es absolutamente necesario
                if adf_diff1 <= 0.05:
                    d = 1  # Primera diferencia ya es estacionaria
                else:
                    # Probar segunda diferencia
                    diff2 = diff1.diff().dropna()
                    if len(diff2) > 10:
                        adf_diff2 = adfuller(diff2, regression='c')[1]
                        # Solo usar d=2 si la prueba es concluyente
                        d = 2 if adf_diff2 <= 0.01 else 1
                    else:
                        d = 1  # Default conservador
            else:
                logger.warning("Serie demasiado corta para segunda diferenciación, se usa d=1")
                d = 1
        else:
            d = 0  # Serie ya estacionaria
    except Exception as e:
        logger.warning(f"Error en test ADF: {str(e)}. Usando d=1 por defecto.")
        d = 1

    # Verificar estacionariedad con KPSS (confirmación)
    try:
        kpss_result = kpss(series, regression='c')
        kpss_pvalue = kpss_result[1]

        # Ajustar d si hay discrepancia entre tests
        if d == 0 and kpss_pvalue < 0.05:
            logger.info("Discrepancia entre tests: ADF indica estacionariedad pero KPSS no. Se ajusta a d=1.")
            d = 1
    except Exception as e:
        logger.warning(f"Error en test KPSS: {str(e)}, se mantiene decisión de ADF")

    # Determinar D (diferenciación estacional)
    # Primero aplicamos la diferenciación regular
    try:
        diff_series = series.diff(periods=1).dropna() if d > 0 else series

        # Luego verificamos si la serie necesita diferenciación estacional
        if len(diff_series) > seasonal_period * 2:
            seasonal_diff = diff_series.diff(periods=seasonal_period).dropna()
            adf_seasonal = adfuller(seasonal_diff, regression='c')[1]

            # Si la diferencia estacional es significativamente más estacionaria, sugerimos D=1
            D = 1 if adf_seasonal < 0.01 else 0
        else:
            logger.warning("Serie demasiado corta para diferenciación estacional, se usa D=0")
            D = 0
    except Exception as e:
        D = 0
        logger.warning(f"Error evaluando diferenciación estacional: {str(e)}, se usa D=0")

    logger.info(f"Órdenes de integración determinados: d={d}, D={D}")
    return d, D


def check_stationarity_for_variables(
    df: pd.DataFrame, exog_cols: List[str], output_dir: Path
) -> Dict[str, Dict[str, Any]]:
    """
    Verifica la estacionariedad de cada variable exógena y documenta resultados.

    Args:
        df: DataFrame con variables
        exog_cols: Lista de variables exógenas
        output_dir: Directorio para guardar resultados

    Returns:
        dict: Resultados de tests para cada variable
    """
    logger = get_logger(__name__)
    results = {}

    for col in exog_cols:
        series = df[col].dropna()
        if len(series) < 30:
            results[col] = {"status": "insufficient_data"}
            logger.warning(f"Variable {col}: datos insuficientes para test de estacionariedad")
            continue

        # Test ADF
        try:
            adf_result = adfuller(series, regression='c')
            adf_pvalue = adf_result[1]

            # Test KPSS
            kpss_result = kpss(series, regression='c')
            kpss_pvalue = kpss_result[1]

            # Determinar estacionariedad
            # - ADF: H0 = tiene raíz unitaria (no estacionaria)
            # - KPSS: H0 = es estacionaria
            if adf_pvalue <= 0.05 and kpss_pvalue > 0.05:
                status = "stationary"
            elif adf_pvalue > 0.05 and kpss_pvalue <= 0.05:
                status = "non_stationary"
            elif adf_pvalue <= 0.05 and kpss_pvalue <= 0.05:
                status = "uncertain_trend_stationary"
            else:
                status = "uncertain"

            results[col] = {"status": status, "adf_pvalue": adf_pvalue, "kpss_pvalue": kpss_pvalue}

            logger.info(f"Variable {col}: {status} (ADF p={adf_pvalue:.4f}, KPSS p={kpss_pvalue:.4f})")

        except Exception as e:
            results[col] = {"status": "error", "error": str(e)}
            logger.error(f"Error en test de estacionariedad para {col}: {str(e)}")

    # Guardar resultados para auditoría
    output_path = output_dir / "stationarity_audit.csv"
    pd.DataFrame.from_dict(results, orient='index').to_csv(output_path)
    logger.info(f"Resultados de estacionariedad guardados en {output_path}")

    return results


def test_granger_causality(
    df: pd.DataFrame, target_col: str, exog_cols: List[str], max_lag: int = 5, output_dir: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Realiza pruebas de causalidad de Granger para variables exógenas.

    Args:
        df: DataFrame con datos
        target_col: Variable objetivo
        exog_cols: Variables exógenas a probar
        max_lag: Máximo rezago a considerar
        output_dir: Directorio para guardar resultados

    Returns:
        dict: Resultados de causalidad para cada variable
    """
    logger = get_logger(__name__)
    results = {}

    y = df[target_col]

    for col in exog_cols:
        # Preparar datos para el test
        data = pd.concat([y, df[col]], axis=1).dropna()

        if len(data) < max_lag + 10:
            results[col] = {"status": "insufficient_data"}
            logger.warning(f"Variable {col}: datos insuficientes para test de Granger")
            continue

        try:
            # Ejecutar test de Granger para múltiples rezagos
            granger_results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

            # Extraer p-valores para cada rezago (usamos test F)
            p_values = {}
            min_p = float('inf')
            best_lag = 0

            for lag, result in granger_results.items():
                # Extraer p-valor del test F (índice [0][0])
                p = result[0]['ssr_ftest'][1]
                p_values[lag] = p

                if p < min_p:
                    min_p = p
                    best_lag = lag

            # Determinar si hay causalidad significativa
            has_causality = min_p < 0.05

            results[col] = {
                "has_causality": has_causality,
                "best_lag": best_lag,
                "min_p_value": min_p,
                "p_values": p_values,
            }

            status = "causa" if has_causality else "no causa"
            logger.info(f"Granger: {col} {status} {target_col} (mejor lag={best_lag}, p={min_p:.4f})")

        except Exception as e:
            results[col] = {"status": "error", "error": str(e)}
            logger.error(f"Error en test de Granger para {col}: {str(e)}")

    # Guardar resultados para auditoría si se proporciona output_dir
    if output_dir is not None:
        output_data = [
            {
                "variable": var,
                "has_causality": results[var].get("has_causality", False),
                "best_lag": results[var].get("best_lag", 0),
                "min_p_value": results[var].get("min_p_value", 1.0),
                "status": results[var].get("status", "ok"),
            }
            for var in results
        ]
        output_path = output_dir / "granger_causality_audit.csv"
        pd.DataFrame(output_data).to_csv(output_path, index=False)
        logger.info(f"Resultados de causalidad guardados en {output_path}")

    return results


def analyze_coefficient_stability(
    df: pd.DataFrame,
    target_col: str,
    exog_cols: List[str],
    model_name: str,
    output_dir: Path,
    window_size: int = 252,
    use_sarimax: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Analiza la estabilidad de coeficientes a lo largo del tiempo usando ventanas deslizantes.

    Args:
        df: DataFrame completo
        target_col: Variable objetivo
        exog_cols: Variables exógenas
        model_name: Nombre del modelo para guardar resultados
        output_dir: Directorio para guardar resultados
        window_size: Tamaño de ventana (252 días ≈ 1 año)
        use_sarimax: Si es True, usa SARIMAX en lugar de OLS

    Returns:
        DataFrame: Coeficientes por ventana
    """
    logger = get_logger(__name__)

    if not exog_cols:
        logger.warning("No se pueden analizar coeficientes sin variables exógenas")
        return None

    # Preparar resultados
    coef_results = []

    # Iterar por ventanas móviles
    for start in range(0, len(df) - window_size, window_size // 2):  # 50% de solapamiento
        end = start + window_size
        window_df = df.iloc[start:end]

        # Entrenar modelo en esta ventana
        try:
            endog = window_df[target_col]
            exog = window_df[exog_cols]

            if use_sarimax:
                # Usar SARIMAX para extraer coeficientes
                d, _ = determine_integration_orders(endog)
                model = SARIMAX(
                    endog, exog=exog, order=(1, d, 1), enforce_stationarity=False, enforce_invertibility=False
                )
                results = model.fit(disp=False)
                params = results.params

                # Extraer solo coeficientes exógenos
                coefs = {col: params[col] for col in exog_cols if col in params}
            else:
                # Ajustar modelo con statsmodels (más simple para extraer coeficientes)
                model = sm.OLS(endog, sm.add_constant(exog))
                results = model.fit()

                # Extraer coeficientes
                coefs = results.params.to_dict()
                if 'const' in coefs:
                    del coefs['const']  # Eliminar constante

            # Agregar fechas de la ventana
            coefs['window_start'] = window_df.index[0]
            coefs['window_end'] = window_df.index[-1]

            coef_results.append(coefs)

        except Exception as e:
            logger.warning(f"Error al ajustar modelo en ventana {start}-{end}: {str(e)}")

    # Convertir a DataFrame
    if coef_results:
        coef_df = pd.DataFrame(coef_results).set_index('window_start')

        # Calcular estadísticas de estabilidad para cada coeficiente
        stability_stats = {}
        for col in exog_cols:
            if col in coef_df.columns:
                col_values = coef_df[col].dropna()

                if len(col_values) < 2:
                    logger.warning(f"No hay suficientes valores para análisis de estabilidad de {col}")
                    continue

                col_mean = col_values.mean()
                col_std = col_values.std()
                sign_changes = ((col_values.shift(1) * col_values) < 0).sum()

                # Coeficiente de variación (cuando tiene sentido)
                if abs(col_mean) > 1e-6:
                    cv = col_std / abs(col_mean)
                else:
                    cv = float('inf')

                # Prueba de significancia de coeficientes
                is_significant = (abs(col_values) / col_std).mean() > 1.96

                stability_stats[col] = {
                    'mean': col_mean,
                    'std': col_std,
                    'cv': cv,
                    'sign_changes': sign_changes,
                    'significant': is_significant,
                    'stability_score': 1.0 / (1.0 + cv + sign_changes),
                }

        # Guardar para análisis
        output_path = output_dir / f"coefficient_stability_{model_name}.csv"
        pd.DataFrame.from_dict(stability_stats, orient='index').to_csv(output_path)
        logger.info(f"Análisis de estabilidad guardado en {output_path}")

        return coef_df

    logger.warning("No se pudieron calcular coeficientes estables")
    return None


# ====================================
# OPTIMIZACIÓN DE HIPERPARÁMETROS
# ====================================


def run_auto_arima(
    y_train: pd.Series, X_train: Optional[pd.DataFrame], d: int, D: int, seasonal: bool, config: Config
) -> Tuple[Optional[pm.arima.ARIMA], Tuple[int, int, int], Optional[Tuple[int, int, int, int]]]:
    """
    Ejecuta auto_arima para encontrar orden óptimo inicial.

    Args:
        y_train: Serie temporal de entrenamiento
        X_train: Variables exógenas (opcional)
        d: Orden de integración
        D: Orden de integración estacional
        seasonal: Si se debe incluir componente estacional
        config: Configuración global

    Returns:
        Tupla con (modelo, order, seasonal_order)
    """
    logger = get_logger(__name__)

    try:
        logger.info(f"Iniciando búsqueda con auto_arima (seasonal={seasonal})")

        auto_model = pm.auto_arima(
            y_train,
            exogenous=X_train,
            start_p=config.arima_start_p,
            start_q=config.arima_start_q,
            max_p=config.arima_max_p,
            max_q=config.arima_max_q,
            d=d,  # Fijamos d según tests
            start_P=config.arima_start_P,
            start_Q=config.arima_start_Q,
            max_P=config.arima_max_P,
            max_Q=config.arima_max_Q,
            D=D,  # Fijamos D según tests
            m=config.seasonal_period if seasonal else 1,
            seasonal=seasonal,
            information_criterion='aic',
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            random_state=config.random_seed,
        )

        order = auto_model.order
        seasonal_order = auto_model.seasonal_order if seasonal else None

        logger.info(f"auto_arima sugiere: order={order}, seasonal_order={seasonal_order}")
        return auto_model, order, seasonal_order

    except Exception as e:
        logger.warning(f"Error en auto_arima: {str(e)}. Usando valores predeterminados.")
        order = (1, d, 1)
        seasonal_order = (1, D, 1, config.seasonal_period) if seasonal else None
        return None, order, seasonal_order


def evaluate_candidate_model(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    y_train: pd.Series,
    X_train: Optional[pd.DataFrame],
    p: int,
    d: int,
    q: int,
    seasonal_order: Optional[Tuple[int, int, int, int]],
    is_sarimax: bool,
) -> float:
    """
    Evalúa un modelo candidato con parámetros específicos.

    Args:
        train_idx: Índices de entrenamiento para CV
        val_idx: Índices de validación para CV
        y_train: Serie temporal completa
        X_train: Variables exógenas completas (opcional)
        p: Orden AR
        d: Orden de integración
        q: Orden MA
        seasonal_order: Orden estacional (opcional)
        is_sarimax: Si es un modelo SARIMAX

    Returns:
        RMSE en validación
    """
    try:
        # Dividir datos según índices
        cv_train = y_train.iloc[train_idx]
        cv_val = y_train.iloc[val_idx]

        # Comprobar datos suficientes
        if len(cv_train) < 10 or len(cv_val) < 1:
            return float('inf')

        if is_sarimax and X_train is not None:
            cv_train_exog = X_train.iloc[train_idx] if X_train is not None else None
            cv_val_exog = X_train.iloc[val_idx] if X_train is not None else None

            # Asegurar que no haya NaNs
            if cv_train_exog is not None and cv_train_exog.isnull().any().any():
                cv_train_exog = cv_train_exog.fillna(method='ffill').fillna(method='bfill')

            if cv_val_exog is not None and cv_val_exog.isnull().any().any():
                cv_val_exog = cv_val_exog.fillna(method='ffill').fillna(method='bfill')

            # Crear y entrenar modelo SARIMAX
            if seasonal_order:
                model = SARIMAX(
                    cv_train,
                    exog=cv_train_exog,
                    order=(p, d, q),
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
            else:
                model = SARIMAX(
                    cv_train,
                    exog=cv_train_exog,
                    order=(p, d, q),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

            # SARIMAX usa disp=False
            try:
                result = model.fit(disp=False, maxiter=50)

                # Predecir con SARIMAX
                pred = result.forecast(steps=len(cv_val), exog=cv_val_exog)
            except:
                # Método más simple si falla la optimización estándar
                result = model.fit(method='css', maxiter=50)
                pred = result.forecast(steps=len(cv_val), exog=cv_val_exog)

        else:
            # Crear y entrenar modelo ARIMA (¡sin disp!)
            model = ARIMA(cv_train, order=(p, d, q))

            try:
                # ARIMA no acepta disp, probar método estándar primero
                result = model.fit()
                # Predecir con ARIMA
                pred = result.forecast(steps=len(cv_val))
            except:
                # Método alternativo si falla el predeterminado
                result = model.fit(method='css')
                pred = result.forecast(steps=len(cv_val))

        # Usar RMSE como métrica
        rmse = np.sqrt(mean_squared_error(cv_val, pred))
        return rmse

    except Exception as e:
        # No devolver el error para no interrumpir el proceso
        return float('inf')  # Penalizar configuraciones que fallan


def search_hyperparameters(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    target_col: str,
    exog_cols: Optional[List[str]],
    config: Config,
    model_type: str,
) -> Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int, int]], Dict[str, Any]]:
    """
    Implementa estrategias de búsqueda de hiperparámetros (auto_arima, optuna, hybrid).

    Args:
        df_train: DataFrame de entrenamiento
        df_val: DataFrame de validación
        target_col: Columna objetivo
        exog_cols: Columnas exógenas (para SARIMAX)
        config: Configuración global
        model_type: Tipo de modelo ("ARIMA" o "SARIMAX_*")

    Returns:
        tuple: (order, seasonal_order, best_params)
    """
    logger = get_logger(__name__, model_type=model_type)

    # Preparar datos
    y_train = df_train[target_col]
    X_train = df_train[exog_cols] if exog_cols else None
    y_val = df_val[target_col]
    X_val = df_val[exog_cols] if exog_cols else None

    # Determinar si es modelo estacional
    is_sarimax = model_type.startswith("SARIMAX")
    seasonal = is_sarimax  # Activar componente estacional sólo para SARIMAX

    # Determinar órdenes de integración (d, D)
    d, D = determine_integration_orders(y_train, seasonal_period=config.seasonal_period)

    if config.optimization_method == "auto_arima":
        # Estrategia 1: Sólo auto_arima
        logger.info(f"Utilizando auto_arima para búsqueda de hiperparámetros (seasonal={seasonal})")

        _, order, seasonal_order = run_auto_arima(y_train, X_train, d, D, seasonal, config)

        # Crear diccionario de parámetros
        best_params = {"p": order[0], "d": order[1], "q": order[2]}

        if seasonal_order:
            best_params.update(
                {"P": seasonal_order[0], "D": seasonal_order[1], "Q": seasonal_order[2], "m": seasonal_order[3]}
            )

        return order, seasonal_order, best_params

    elif config.optimization_method == "optuna":
        # Estrategia 2: Sólo Optuna
        logger.info(f"Utilizando Optuna para búsqueda de hiperparámetros")

        # Configuración de CV temporal
        tscv = TimeSeriesSplit(n_splits=config.cv_splits, test_size=config.test_size)
        cv_splits_data = list(tscv.split(y_train))

        # Función objetivo para Optuna
        def objective(trial):
            # Sugerir hiperparámetros
            p = trial.suggest_int("p", 0, config.arima_max_p)
            q = trial.suggest_int("q", 0, config.arima_max_q)

            if is_sarimax:
                P = trial.suggest_int("P", 0, config.arima_max_P)
                Q = trial.suggest_int("Q", 0, config.arima_max_Q)
                seasonal_order = (P, D, Q, config.seasonal_period)
            else:
                seasonal_order = None

            # Early-pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Validación cruzada con paralelización opcional
            if config.n_jobs != 1:
                errors = Parallel(n_jobs=config.n_jobs)(
                    delayed(evaluate_candidate_model)(
                        train_idx, val_idx, y_train, X_train, p, d, q, seasonal_order, is_sarimax
                    )
                    for train_idx, val_idx in cv_splits_data
                )
            else:
                errors = [
                    evaluate_candidate_model(train_idx, val_idx, y_train, X_train, p, d, q, seasonal_order, is_sarimax)
                    for train_idx, val_idx in cv_splits_data
                ]

            # Filtrar errores infinitos
            errors = [e for e in errors if e != float('inf')]

            # Si todos los modelos fallaron, penalizar
            if not errors:
                return float('inf')

            return float(np.mean(errors))

        # Crear y optimizar estudio
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler(seed=config.random_seed),
        )

        study.optimize(objective, n_trials=config.max_trials, n_jobs=1)  # Parallelism inside objective

        # Obtener mejores parámetros
        best_params = study.best_params
        order = (best_params["p"], d, best_params["q"])

        if is_sarimax:
            seasonal_order = (best_params["P"], D, best_params["Q"], config.seasonal_period)
        else:
            seasonal_order = None

        # Actualizar diccionario para incluir órdenes de integración
        best_params["d"] = d
        if is_sarimax:
            best_params["D"] = D
            best_params["m"] = config.seasonal_period

        return order, seasonal_order, best_params

    else:  # hybrid
        # Estrategia 3: Híbrido (auto_arima + Optuna)
        logger.info(f"Utilizando método híbrido (auto_arima + Optuna) para búsqueda de hiperparámetros")

        # FASE 1: Búsqueda inicial con auto_arima
        _, initial_order, initial_seasonal_order = run_auto_arima(y_train, X_train, d, D, seasonal, config)

        # Evaluación inicial
        initial_rmse = float('inf')
        try:
            if is_sarimax:
                model = SARIMAX(
                    y_train,
                    exog=X_train,
                    order=initial_order,
                    seasonal_order=initial_seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False)

                # Predecir en validación con SARIMAX
                val_pred = result.forecast(steps=len(y_val), exog=X_val)
            else:
                model = ARIMA(y_train, order=initial_order)
                # ARIMA no acepta disp=False
                result = model.fit()

                # Predecir en validación con ARIMA
                val_pred = result.forecast(steps=len(y_val))

            initial_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            logger.info(f"RMSE inicial en validación: {initial_rmse:.4f}")

        except Exception as e:
            logger.warning(f"Error evaluando modelo inicial: {str(e)}")

        # FASE 2: Refinamiento con Optuna
        logger.info("Iniciando refinamiento con Optuna")

        # Configuración de CV temporal
        tscv = TimeSeriesSplit(n_splits=config.cv_splits, test_size=config.test_size)
        cv_splits_data = list(tscv.split(y_train))

        # Función objetivo de Optuna
        def objective(trial):
            # Sugerir hiperparámetros dentro de rangos acotados alrededor de los iniciales
            # Limitamos la búsqueda a ±1 de la solución de auto_arima
            p_min, p_max = max(0, initial_order[0] - 1), min(config.arima_max_p, initial_order[0] + 1)
            q_min, q_max = max(0, initial_order[2] - 1), min(config.arima_max_q, initial_order[2] + 1)

            p = trial.suggest_int("p", p_min, p_max)
            # d ya está fijado
            q = trial.suggest_int("q", q_min, q_max)

            if is_sarimax and initial_seasonal_order:
                P_min, P_max = max(0, initial_seasonal_order[0] - 1), min(
                    config.arima_max_P, initial_seasonal_order[0] + 1
                )
                Q_min, Q_max = max(0, initial_seasonal_order[2] - 1), min(
                    config.arima_max_Q, initial_seasonal_order[2] + 1
                )

                P = trial.suggest_int("P", P_min, P_max)
                # D ya está fijado
                Q = trial.suggest_int("Q", Q_min, Q_max)
                m = config.seasonal_period  # Fijo para días hábiles semanales
                seasonal_order = (P, D, Q, m)
            else:
                seasonal_order = None

            # Early-pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Validación cruzada con paralelización opcional
            if config.n_jobs != 1:
                errors = Parallel(n_jobs=config.n_jobs)(
                    delayed(evaluate_candidate_model)(
                        train_idx, val_idx, y_train, X_train, p, d, q, seasonal_order, is_sarimax
                    )
                    for train_idx, val_idx in cv_splits_data
                )
            else:
                errors = [
                    evaluate_candidate_model(train_idx, val_idx, y_train, X_train, p, d, q, seasonal_order, is_sarimax)
                    for train_idx, val_idx in cv_splits_data
                ]

            # Filtrar errores infinitos
            errors = [e for e in errors if e != float('inf')]

            # Si todos los modelos fallaron, penalizar
            if not errors:
                return float('inf')

            return float(np.mean(errors))

        # Crear y optimizar estudio de Optuna
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler(seed=config.random_seed),
        )

        study.optimize(objective, n_trials=config.max_trials, n_jobs=1)  # Parallelism inside objective

        # Crear orden óptimo
        best_params = study.best_params
        final_order = (best_params.get("p", initial_order[0]), d, best_params.get("q", initial_order[2]))

        if is_sarimax and initial_seasonal_order:
            final_seasonal_order = (
                best_params.get("P", initial_seasonal_order[0]),
                D,
                best_params.get("Q", initial_seasonal_order[2]),
                config.seasonal_period,  # Mantener período fijo
            )
        else:
            final_seasonal_order = None

        # Comparar con resultado inicial
        try:
            if is_sarimax:
                model = SARIMAX(
                    y_train,
                    exog=X_train,
                    order=final_order,
                    seasonal_order=final_seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False)

                # Predecir con SARIMAX
                val_pred = result.forecast(steps=len(y_val), exog=X_val)
            else:
                model = ARIMA(y_train, order=final_order)
                # ARIMA no acepta disp
                result = model.fit()

                # Predecir con ARIMA
                val_pred = result.forecast(steps=len(y_val))

            final_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

            improvement = ((initial_rmse - final_rmse) / initial_rmse) * 100 if initial_rmse != float('inf') else 0
            logger.info(f"RMSE final en validación: {final_rmse:.4f} (mejora: {improvement:.2f}%)")
            logger.info(f"Orden óptimo: {final_order}, seasonal_order: {final_seasonal_order}")

        except Exception as e:
            logger.warning(f"Error evaluando modelo final: {str(e)}")
            logger.info(f"Se mantiene modelo inicial: {initial_order}, {initial_seasonal_order}")
            final_order = initial_order
            final_seasonal_order = initial_seasonal_order

        # Crear diccionario completo de parámetros
        all_best_params = {"p": final_order[0], "d": final_order[1], "q": final_order[2]}

        if final_seasonal_order:
            all_best_params.update(
                {
                    "P": final_seasonal_order[0],
                    "D": final_seasonal_order[1],
                    "Q": final_seasonal_order[2],
                    "m": final_seasonal_order[3],
                }
            )

        return final_order, final_seasonal_order, all_best_params


# ====================================
# ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# ====================================


def train_final_model(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    exog_cols: Optional[List[str]],
    order: Tuple[int, int, int],
    seasonal_order: Optional[Tuple[int, int, int, int]],
    model_type: str,
) -> Tuple[Any, np.ndarray, ModelMetrics]:
    """
    Entrena el modelo final con la configuración óptima.

    Args:
        df: DataFrame completo
        df_test: DataFrame de test
        target_col: Columna objetivo
        exog_cols: Columnas exógenas
        order: Orden ARIMA (p,d,q)
        seasonal_order: Orden estacional (P,D,Q,m)
        model_type: Tipo de modelo

    Returns:
        tuple: (modelo_entrenado, predicciones_test, métricas)
    """
    logger = get_logger(__name__)

    # Separar datos
    y = df[target_col]
    X = df[exog_cols] if exog_cols else None
    y_test = df_test[target_col]
    X_test = df_test[exog_cols] if exog_cols else None

    # Crear y entrenar modelo
    try:
        if model_type == "ARIMA":
            model = ARIMA(y, order=order)
            # ARIMA no acepta disp=False
            result = model.fit()
        else:  # SARIMAX
            if seasonal_order:
                model = SARIMAX(
                    y,
                    exog=X,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
            else:
                model = SARIMAX(y, exog=X, order=order, enforce_stationarity=False, enforce_invertibility=False)
            # Solo SARIMAX acepta disp=False
            result = model.fit(disp=False)
    except Exception as e:
        logger.error(f"Error entrenando modelo final: {str(e)}")
        raise ValueError(f"No se pudo entrenar el modelo: {str(e)}")

    # Hacer predicciones en el test set
    try:
        if model_type == "ARIMA":
            test_pred = result.forecast(steps=len(df_test))
        else:
            test_pred = result.forecast(steps=len(df_test), exog=X_test)
    except Exception as e:
        logger.error(f"Error generando predicciones: {str(e)}")
        raise ValueError(f"No se pudieron generar predicciones: {str(e)}")

    # Calcular métricas
    rmse = sqrt(mean_squared_error(y_test, test_pred))
    mae = mean_absolute_error(y_test, test_pred)

    # Calcular MAPE evitando divisiones por cero
    try:
        if (y_test == 0).any():
            mape = np.nan
        else:
            mape = mean_absolute_percentage_error(y_test, test_pred) * 100
    except Exception:
        mape = np.nan

    # Calcular hit direction (porcentaje de aciertos en dirección)
    try:
        y_diff = y_test.diff().dropna()
        pred_diff = pd.Series(test_pred).diff().dropna()

        dir_y = np.sign(y_diff)
        dir_pred = np.sign(pred_diff.values[: len(dir_y)])

        # Evitar divisiones por cero si hay pocos datos
        hit_direction = np.mean(dir_y == dir_pred) * 100 if len(dir_y) > 0 else np.nan
    except Exception:
        hit_direction = np.nan

    # Calcular mejora vs naive (último valor conocido)
    try:
        naive_pred = np.full(len(y_test), y.iloc[-1])
        naive_rmse = sqrt(mean_squared_error(y_test, naive_pred))
    except Exception:
        naive_rmse = np.nan

    # Calcular mejora vs naive drift (tendencia lineal)
    try:
        if len(y) >= 2:
            last_diff = y.iloc[-1] - y.iloc[-2]
            naive_drift_pred = [y.iloc[-1] + i * last_diff for i in range(1, len(y_test) + 1)]
            naive_drift_rmse = sqrt(mean_squared_error(y_test, naive_drift_pred))
        else:
            naive_drift_rmse = np.nan
    except Exception:
        naive_drift_rmse = np.nan

    # Mejora porcentual
    try:
        if naive_rmse > 0:
            improvement_vs_naive = ((naive_rmse - rmse) / naive_rmse) * 100
        else:
            improvement_vs_naive = np.nan

        if naive_drift_rmse > 0 and not np.isnan(naive_drift_rmse):
            improvement_vs_drift = ((naive_drift_rmse - rmse) / naive_drift_rmse) * 100
        else:
            improvement_vs_drift = np.nan
    except Exception:
        improvement_vs_naive = np.nan
        improvement_vs_drift = np.nan

    # Registrar resultados
    metrics = ModelMetrics(
        rmse=rmse,
        mae=mae,
        mape=mape,
        hit_direction=hit_direction,
        naive_rmse=naive_rmse,
        naive_drift_rmse=naive_drift_rmse,
        improvement_vs_naive=improvement_vs_naive,
        improvement_vs_drift=improvement_vs_drift,
    )

    logger.info(f"Modelo final {model_type} - RMSE: {rmse:.4f}")
    logger.info(f"Hit Direction: {hit_direction:.2f}%")
    logger.info(f"Mejora vs. Naive: {improvement_vs_naive:.2f}%, vs. Drift: {improvement_vs_drift:.2f}%")

    return result, test_pred, metrics


def forecast_future(
    model: Any, df: pd.DataFrame, target_col: str, exog_cols: Optional[List[str]], model_type: str, steps: int = 21
) -> pd.DataFrame:
    """
    Genera pronóstico para los próximos días hábiles.

    Args:
        model: Modelo entrenado
        df: DataFrame de datos históricos
        target_col: Variable objetivo
        exog_cols: Variables exógenas
        model_type: Tipo de modelo
        steps: Horizonte de pronóstico en días hábiles

    Returns:
        DataFrame: Pronósticos con intervalos de confianza
    """
    logger = get_logger(__name__)

    try:
        # Generar fechas futuras (días hábiles) primero
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='B')
        logger.info(f"Generando pronóstico para {steps} pasos, desde {future_dates[0]} hasta {future_dates[-1]}")

        if model_type == "ARIMA":
            # Usar get_forecast en lugar de forecast directamente
            forecast_result = model.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean
            ci = forecast_result.conf_int(alpha=0.05)
            lower = ci.iloc[:, 0]
            upper = ci.iloc[:, 1]

            # Verificar que forecast tenga la longitud esperada
            if len(forecast) != steps:
                logger.warning(f"ADVERTENCIA: ARIMA devolvió {len(forecast)} pasos en lugar de {steps}")

        else:  # SARIMAX
            # Mejor manejo de exógenas futuras
            if exog_cols:
                # Método 1: Repetir último valor conocido
                last_exog = df[exog_cols].iloc[-1:].values
                X_future = np.repeat(last_exog, steps, axis=0)

                # Verificar forma correcta
                if X_future.shape[0] != steps or X_future.shape[1] != len(exog_cols):
                    logger.warning(
                        f"Forma incorrecta de X_future: {X_future.shape}, esperado: ({steps}, {len(exog_cols)})"
                    )
                    X_future = np.reshape(X_future, (steps, len(exog_cols)))

                logger.info(f"X_future shape: {X_future.shape}")
            else:
                X_future = None

            # Use in_sample=False para garantizar que sea un pronóstico real
            forecast_result = model.get_forecast(steps=steps, exog=X_future)
            forecast = forecast_result.predicted_mean
            ci = forecast_result.conf_int(alpha=0.05)
            lower = ci.iloc[:, 0]
            upper = ci.iloc[:, 1]

            logger.info(f"SARIMAX generó pronóstico con {len(forecast)} pasos")
    except Exception as e:
        logger.error(f"Error generando pronóstico: {str(e)}")
        # En caso de error, generar un forecast simple
        logger.warning("Generando pronóstico simplificado como fallback")

        # Utilizar el último valor + tendencia simple
        last_value = df[target_col].iloc[-1]
        trend = 0
        if len(df) > 5:  # Si hay suficientes datos para tendencia
            trend = (df[target_col].iloc[-1] - df[target_col].iloc[-5]) / 5

        forecast = np.array([last_value + trend * i for i in range(1, steps + 1)])
        lower = forecast * 0.95  # Simplificación de intervalos
        upper = forecast * 1.05

    # Generar fechas futuras (días hábiles)
    last_date = df.index[-1]
    future_dates = generate_forecast_dates(last_date, steps, freq='B')

    # Garantizar que todas las longitudes coincidan
    min_length = min(len(future_dates), len(forecast), len(lower), len(upper))

    # Crear DataFrame de pronóstico
    forecast_df = pd.DataFrame(
        {
            'date': future_dates[:min_length],
            'forecast': forecast[:min_length],
            'lower_95': lower[:min_length],
            'upper_95': upper[:min_length],
        }
    )

    return forecast_df


def generate_comprehensive_timeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    instrument: str,
    model_type: str,
    output_dir: Path,
) -> Path:
    """Crea una visualización clara que muestra todos los períodos con líneas divisorias."""

    # Verificar que está disponible matplotlib
    import matplotlib.pyplot as plt

    # Crear figura
    fig, ax = plt.subplots(figsize=(15, 8))

    # Graficar cada período con estilo diferente
    if train_df is not None and len(train_df) > 0:
        target_col = train_df.columns[0]  # Primera columna es el target
        ax.plot(train_df.index, train_df[target_col], color='blue', label='Entrenamiento', linewidth=1.5)

    if val_df is not None and len(val_df) > 0:
        target_col = val_df.columns[0]
        ax.plot(val_df.index, val_df[target_col], color='green', label='Validación', linewidth=1.5)

    if test_df is not None and len(test_df) > 0:
        target_col = test_df.columns[0]
        ax.plot(test_df.index, test_df[target_col], color='orange', label='Test', linewidth=1.5)

    # Verificar que forecast_df tenga la columna 'forecast'
    if forecast_df is not None and 'forecast' in forecast_df.columns and len(forecast_df) > 0:
        # Asegurar que forecast_df tenga un índice de datetime
        if 'date' in forecast_df.columns and not isinstance(forecast_df.index, pd.DatetimeIndex):
            forecast_index = pd.to_datetime(forecast_df['date'])
        else:
            forecast_index = forecast_df.index

        # Graficar pronóstico y bandas de confianza
        ax.plot(
            forecast_index,
            forecast_df['forecast'],
            color='red',
            linestyle='--',
            label='Pronóstico (21 días)',
            linewidth=2.0,
        )

        # Incluir bandas de confianza si están disponibles
        if 'lower_95' in forecast_df.columns and 'upper_95' in forecast_df.columns:
            ax.fill_between(
                forecast_index,
                forecast_df['lower_95'],
                forecast_df['upper_95'],
                color='red',
                alpha=0.2,
                label='Int. Confianza 95%',
            )

    # Añadir líneas verticales en las transiciones
    if val_df is not None and len(val_df) > 0:
        if train_df is not None and len(train_df) > 0:
            # Línea de fin de entrenamiento = inicio de validación
            transition_date = train_df.index[-1]
            ax.axvline(x=transition_date, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

            # Etiqueta visible
            ax.annotate(
                'FIN TRAIN',
                xy=(transition_date, ax.get_ylim()[0]),
                xytext=(5, 15),
                textcoords='offset points',
                color='gray',
                fontweight='bold',
                fontsize=10,
            )

    if test_df is not None and len(test_df) > 0:
        # Línea de fin de validación = inicio de test
        transition_date = val_df.index[-1] if val_df is not None and len(val_df) > 0 else None
        if transition_date is not None:
            ax.axvline(x=transition_date, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

            # Etiqueta visible
            ax.annotate(
                'FIN VALIDACIÓN',
                xy=(transition_date, ax.get_ylim()[0]),
                xytext=(5, 15),
                textcoords='offset points',
                color='black',
                fontweight='bold',
                fontsize=10,
            )

    if forecast_df is not None and len(forecast_df) > 0:
        # Línea de fin de test = inicio de pronóstico
        if test_df is not None and len(test_df) > 0:
            transition_date = test_df.index[-1]
            ax.axvline(x=transition_date, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8)

            # Etiqueta más visible
            ax.annotate(
                'INICIO PRONÓSTICO',
                xy=(transition_date, ax.get_ylim()[0]),
                xytext=(10, 30),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#E74C3C'),
                color='#E74C3C',
                fontweight='bold',
                fontsize=12,
            )

    # Configurar gráfico
    title = f"Línea de Tiempo Completa - {instrument} - {model_type}"
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Guardar
    output_path = output_dir / f"{instrument}_{model_type}_timeline.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


# ====================================
# FUNCIÓN PARA GENERAR VISUALIZACIONES
# ====================================


# Encuentra esta función en el código y modifícala así:
def generate_model_visualizations(
    model: Any,
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    exog_cols: Optional[List[str]],
    test_pred: np.ndarray,
    forecast_df: pd.DataFrame,
    metrics: ModelMetrics,
    model_type: str,
    instrument: str,
    output_dir: Path,
    df_train: Optional[pd.DataFrame] = None,
    df_val: Optional[pd.DataFrame] = None,
) -> Dict[str, Path]:
    """
    Genera visualizaciones para el modelo y las guarda en el directorio de salida.
    """
    # Verificar si el módulo de visualizaciones está disponible
    if 'plots' not in globals():
        return {}

    logger = get_logger(__name__, instrument=instrument, model_type=model_type)

    # Crear directorio para gráficos
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True, parents=True)

    # Rutas para los gráficos
    paths = {}

    try:
        # Series reales y predichas
        y_train = df[target_col]
        y_test = df_test[target_col]

        # Test predictions como Series para facilitar visualización
        test_pred_series = pd.Series(test_pred, index=df_test.index)

        # Determinar fechas de fin de entrenamiento y validación
        train_end_date = None
        val_end_date = None

        if df_train is not None and len(df_train) > 0:
            train_end_date = df_train.index[-1]

        if df_val is not None and len(df_val) > 0:
            val_end_date = df_val.index[-1]
        else:
            # Si no se proporciona df_val, podemos inferir la fecha de fin de validación
            # como la fecha justo antes del inicio de test
            if len(df_test) > 0:
                val_end_date = df_test.index[0] - pd.Timedelta(days=1)

        # 1. Gráfico de comparación real vs predicción
        comparison_path = charts_dir / f"{instrument}_{model_type}_comparison.png"
        plots.plot_series_comparison(
            real_series=y_test,
            pred_series=test_pred_series,
            title="Real vs Predicción",
            save_path=comparison_path,
            plot_diff=True,
            model_name=model_type,
            instrument=instrument,
        )
        paths['comparison'] = comparison_path
        logger.info(f"Gráfico de comparación guardado en {comparison_path}")

        # 2. Gráfico de pronóstico
        forecast_series = forecast_df['forecast']
        ci_lower = forecast_df['lower_95'] if 'lower_95' in forecast_df else None
        ci_upper = forecast_df['upper_95'] if 'upper_95' in forecast_df else None

        forecast_path = charts_dir / f"{instrument}_{model_type}_forecast.png"

        # SOLUCIÓN: Verificar qué parámetros acepta la función plot_forecast
        # Opción 1: Eliminar los parámetros problemáticos
        # Asegurarse de que y_val_pred exista
        y_val_pred = None
        if df_val is not None and len(df_val) > 0:
            try:
                if model_type.startswith("SARIMAX"):
                    y_val_pred = final_model.get_prediction(
                        start=df_val.index[0], end=df_val.index[-1], exog=df_val[exog_cols] if exog_cols else None
                    ).predicted_mean
                else:
                    y_val_pred = final_model.get_prediction(start=df_val.index[0], end=df_val.index[-1]).predicted_mean
            except Exception as e:
                logger.warning(f"No se pudieron obtener predicciones para validación: {str(e)}")
                y_val_pred = None

        # Combinar predicciones de validación y test
        if y_val_pred is not None:
            combined_pred = pd.concat(
                [pd.Series(y_val_pred, index=df_val.index), pd.Series(test_pred, index=df_test.index)]
            )
        else:
            combined_pred = pd.Series(test_pred, index=df_test.index)

        plots.plot_forecast(
            historical=y_train,
            pred_series=combined_pred,
            forecast=forecast_series,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            title="Pronóstico",
            save_path=forecast_path,
            instrument=instrument,
            model_name=model_type,
            train_end_date=df_train.index[-1] if df_train is not None and len(df_train) > 0 else None,
            val_end_date=df_val.index[-1] if df_val is not None and len(df_val) > 0 else None,
            inverse_transform=np.exp,  # Ajustar según la transformación que uses (np.exp o np.expm1)
        )

        # Opción 2 (alternativa): Si realmente necesitas marcar estas fechas, puedes
        # verificar la firma de la función y usar los argumentos correctos
        # Por ejemplo, si la función acepta 'vlines' o algo similar:
        """
        plots.plot_forecast(
            historical=y_train,
            forecast=forecast_series,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            title="Pronóstico",
            save_path=forecast_path,
            instrument=instrument,
            model_name=model_type,
            vlines=[train_end_date, val_end_date],
            vline_labels=["Fin Training", "Fin Validación"]
        )
        """

        paths['forecast'] = forecast_path
        logger.info(f"Gráfico de pronóstico guardado en {forecast_path}")

        # 3. Gráfico de diagnóstico (residuos)
        try:
            # Obtener residuos para diagnóstico
            residuals = pd.Series(y_test.values - test_pred, index=y_test.index)

            diagnostic_path = charts_dir / f"{instrument}_{model_type}_diagnostics.png"
            plots.plot_model_diagnostics(
                model_fit=model,
                residuals=residuals,
                title="Diagnóstico de Modelo",
                save_path=diagnostic_path,
                instrument=instrument,
                model_name=model_type,
            )
            paths['diagnostics'] = diagnostic_path
            logger.info(f"Gráfico de diagnóstico guardado en {diagnostic_path}")
        except Exception as e:
            logger.warning(f"Error generando diagnóstico: {str(e)}")

        # 4. Dashboard completo
        metrics_dict = metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics
        dashboard_path = charts_dir / f"{instrument}_{model_type}_dashboard.png"
        plots.create_dashboard(
            real_series=y_test,
            pred_series=test_pred_series,
            forecast_series=forecast_series,
            metrics=metrics_dict,
            title="Dashboard",
            save_path=dashboard_path,
            instrument=instrument,
            model_name=model_type,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            # También eliminamos train_end_date y val_end_date aquí
        )
        paths['dashboard'] = dashboard_path
        logger.info(f"Dashboard guardado en {dashboard_path}")

        # 5. Línea de tiempo completa (nuevo gráfico)
        if df_train is None or df_val is None:
            df_train, df_val, _ = split_data(df)

        timeline_path = generate_comprehensive_timeline(
            train_df=df_train,
            val_df=df_val,
            test_df=df_test,
            forecast_df=forecast_df,
            instrument=instrument,
            model_type=model_type,
            output_dir=charts_dir,
        )
        paths['timeline'] = timeline_path
        logger.info(f"Línea de tiempo guardada en {timeline_path}")

    except Exception as e:
        logger.error(f"Error generando visualizaciones: {str(e)}")
        logger.error(f"Traza detallada: {traceback.format_exc()}")

    return paths


def process_instrument(instrument: str, config: Config) -> List[Dict[str, Any]]:
    """
    Procesa un instrumento completo con todos sus modelos.

    Args:
        instrument: Nombre del instrumento (SP500, EURUSD, USDJPY)
        config: Configuración global

    Returns:
        list: Resultados formatados para salida
    """
    logger = get_logger(__name__, instrument=instrument)
    logger.info(f"Procesando instrumento: {instrument}")

    results = []

    # Crear directorio para este instrumento
    output_inst_dir = config.output_dir / instrument
    output_inst_dir.mkdir(exist_ok=True)

    # Para cada tipo de modelo
    for model_type in config.model_types:
        try:
            model_logger = get_logger(__name__, instrument=instrument, model_type=model_type)
            model_logger.info(f"Procesando {instrument} - {model_type}")

            # 1. Cargar datos
            model_suffix = model_type.replace("SARIMAX_", "") if model_type.startswith("SARIMAX") else model_type
            df = load_data(config, instrument, model_suffix)

            if df is None or df.empty:
                model_logger.error(f"No se pudieron cargar datos para {instrument} - {model_type}")
                continue

            # 2. Identificar target y exógenas
            target_col = df.columns[0]  # Primera columna es el target
            exog_cols = df.columns[1:].tolist() if len(df.columns) > 1 and model_type != "ARIMA" else None

            # 3. Dividir datos
            df_train, df_val, df_test = split_data(df)

            # 4. Verificar estacionariedad (solo la primera vez)
            if model_type == "SARIMAX_EXTENDED" and exog_cols:
                check_stationarity_for_variables(df, exog_cols, config.output_dir)

                # Test de causalidad de Granger
                test_granger_causality(df, target_col, exog_cols, output_dir=config.output_dir)

            # 5. Optimizar hiperparámetros
            order, seasonal_order, best_params = search_hyperparameters(
                df_train, df_val, target_col, exog_cols, config, model_type
            )

            # 6. Entrenar modelo final con todos los datos hasta la fecha de test
            final_model, test_pred, metrics = train_final_model(
                df.iloc[: -len(df_test)], df_test, target_col, exog_cols, order, seasonal_order, model_type
            )

            # 7. Análisis de estabilidad para modelos SARIMAX
            if model_type.startswith("SARIMAX") and exog_cols:
                analyze_coefficient_stability(
                    df,
                    target_col,
                    exog_cols,
                    f"{instrument}_{model_type}",
                    config.output_dir,
                    use_sarimax=True,  # Usar SARIMAX para mayor coherencia
                )

            # 8. Generar forecast futuro
            forecast_df = forecast_future(
                final_model, df, target_col, exog_cols, model_type, steps=config.forecast_horizon
            )

            # 9. Guardar modelo (usar joblib en lugar de pickle)
            model_file = config.models_dir / f"{instrument}_{model_type}.pkl"

            try:
                # Usar joblib para mejor rendimiento y compatibilidad
                dump(final_model, model_file)
                model_logger.info(f"Modelo guardado en {model_file}")
            except Exception as e:
                model_logger.error(f"Error guardando modelo: {str(e)}")

                # Fallback a pickle si joblib falla
                with open(model_file, 'wb') as f:
                    pickle.dump(final_model, f)
                model_logger.info(f"Modelo guardado con pickle en {model_file}")

            # 10. Guardar métricas
            metrics_file = output_inst_dir / f"{instrument}_{model_type}_metrics.json"
            try:
                with open(metrics_file, 'w') as f:
                    # Añadir información de hiperparámetros
                    full_metrics = {
                        "instrument": instrument,
                        "model_type": model_type,
                        "order": str(order),
                        "seasonal_order": str(seasonal_order),
                        "hyperparameters": best_params,
                        "metrics": metrics.to_dict(),
                    }
                    json.dump(full_metrics, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
                model_logger.info(f"Métricas guardadas en {metrics_file}")
            except Exception as e:
                model_logger.error(f"Error guardando métricas: {str(e)}")

            # 11. Guardar forecast
            forecast_file = output_inst_dir / f"{instrument}_{model_type}_forecast.csv"
            try:
                forecast_df.to_csv(forecast_file, index=False)
                model_logger.info(f"Forecast guardado en {forecast_file}")
            except Exception as e:
                model_logger.error(f"Error guardando forecast: {str(e)}")

            # 12. Generar visualizaciones
            if 'plots' in globals():
                try:
                    visualizations = generate_model_visualizations(
                        model=final_model,
                        df=df,
                        df_test=df_test,
                        target_col=target_col,
                        exog_cols=exog_cols,
                        test_pred=test_pred,
                        forecast_df=forecast_df,
                        metrics=metrics,
                        model_type=model_type,
                        instrument=instrument,
                        output_dir=output_inst_dir,
                        df_train=df_train,
                        df_val=df_val,
                    )

                    if visualizations:
                        model_logger.info(f"Visualizaciones generadas: {len(visualizations)}")
                except Exception as e:
                    model_logger.error(f"Error generando visualizaciones: {str(e)}")

            # 13. Generar registros para all_models_predictions.csv usando vectorización
            try:
                # Construir resultados usando listas de filas para evitar problemas de índices
                model_results = []

                # Sección Training (lista de diccionarios)
                for idx, row in df_train.iterrows():
                    model_results.append(
                        {
                            "date": idx,
                            "Valor_Real": float(row[target_col]),
                            "Valor_Predicho": float(row[target_col]),  # Para Training usamos valor real = predicho
                            "Modelo": model_type,
                            "Version": "1.0",
                            "RMSE": None,
                            "Hyperparámetros": json.dumps({"order": str(order), "seasonal_order": str(seasonal_order)}),
                            "Tipo_Mercado": instrument,
                            "Periodo": "Training",
                        }
                    )

                # Sección Evaluación - Solo si tenemos datos válidos
                try:
                    eval_pred = final_model.get_prediction(
                        start=df_val.index[0], end=df_val.index[-1], exog=df_val[exog_cols] if exog_cols else None
                    ).predicted_mean

                    # Asegurar que las longitudes coincidan
                    min_eval_len = min(len(df_val), len(eval_pred))

                    val_rmse = np.sqrt(mean_squared_error(df_val[target_col][:min_eval_len], eval_pred[:min_eval_len]))

                    for i in range(min_eval_len):
                        model_results.append(
                            {
                                "date": df_val.index[i],
                                "Valor_Real": float(df_val.iloc[i][target_col]),
                                "Valor_Predicho": float(eval_pred[i]),
                                "Modelo": model_type,
                                "Version": "1.0",
                                "RMSE": float(val_rmse),
                                "Hyperparámetros": json.dumps(
                                    {"order": str(order), "seasonal_order": str(seasonal_order)}
                                ),
                                "Tipo_Mercado": instrument,
                                "Periodo": "Evaluacion",
                            }
                        )
                except Exception as e:
                    model_logger.error(f"Error generando datos de evaluación: {str(e)}")

                # Sección Test - Aseguramos acceso por posición correcta a test_pred
                if isinstance(test_pred, pd.Series):
                    # Si es Series, reseteamos el índice para garantizar acceso [0]
                    test_pred = test_pred.reset_index(drop=True)
                else:
                    # Si ya es array u otro formato, convertir a lista para indexación segura
                    test_pred = np.array(test_pred).flatten()

                test_pred_len = len(test_pred)
                test_indices = df_test.index[:test_pred_len]
                test_real_values = df_test[target_col][:test_pred_len]

                model_logger.info(
                    f"Longitudes para Test: índices={len(test_indices)}, valores={len(test_real_values)}, predicciones={test_pred_len}"
                )

                # Verificación final y recorte si es necesario
                min_len = min(len(test_indices), len(test_real_values), test_pred_len)

                for i in range(min_len):
                    model_results.append(
                        {
                            "date": test_indices[i],
                            "Valor_Real": float(test_real_values.iloc[i]),
                            "Valor_Predicho": float(test_pred[i]),  # Ahora con indexación garantizada
                            "Modelo": model_type,
                            "Version": "1.0",
                            "RMSE": float(metrics.rmse),
                            "Hyperparámetros": json.dumps({"order": str(order), "seasonal_order": str(seasonal_order)}),
                            "Tipo_Mercado": instrument,
                            "Periodo": "Test",
                        }
                    )

                # Sección Forecast - También fila por fila
                for i in range(len(forecast_df)):
                    model_results.append(
                        {
                            "date": pd.Timestamp(forecast_df['date'].iloc[i]),
                            "Valor_Real": None,  # Valor futuro desconocido
                            "Valor_Predicho": float(forecast_df['forecast'].iloc[i]),
                            "Modelo": model_type,
                            "Version": "1.0",
                            "RMSE": None,
                            "Hyperparámetros": json.dumps({"order": str(order), "seasonal_order": str(seasonal_order)}),
                            "Tipo_Mercado": instrument,
                            "Periodo": "Forecast",
                        }
                    )

                # Agregar al resultado general
                results.extend(model_results)

                model_logger.info(f"Procesamiento completado para {instrument} - {model_type}")

            except Exception as e:
                logger.error(f"Error generando resultados para all_models_predictions.csv: {str(e)}")
                logger.error(f"Detalles del error: {traceback.format_exc()}")

            # Generar visualización compresiva con timeline
            try:
                generate_comprehensive_timeline(
                    train_df=df_train,
                    val_df=df_val,
                    test_df=df_test,
                    forecast_df=forecast_df,
                    instrument=instrument,
                    model_type=model_type,
                    output_dir=output_inst_dir,
                )
                model_logger.info(f"Línea de tiempo completa generada para {instrument} - {model_type}")
            except Exception as e:
                model_logger.error(f"Error generando línea de tiempo completa: {str(e)}")

        except Exception as e:
            logger.error(f"Error procesando {instrument} - {model_type}: {str(e)}")
            logger.error(f"Traza de error: {traceback.format_exc()}")

    return results


# ====================================
# FUNCIÓN PRINCIPAL
# ====================================


def main():
    """Función principal."""
    # Configurar argumentos
    import argparse

    parser = argparse.ArgumentParser(description="Entrenar modelos ARIMA/SARIMAX")
    parser.add_argument("--instruments", nargs="+", default=None, help="Lista de instrumentos a procesar")
    parser.add_argument(
        "--method",
        choices=["auto_arima", "optuna", "hybrid"],
        default=None,
        help="Método de optimización de hiperparámetros",
    )
    parser.add_argument("--trials", type=int, default=None, help="Número de trials para optimización (Optuna)")
    parser.add_argument("--config", type=str, default=None, help="Ruta al archivo de configuración YAML")
    parser.add_argument(
        "--jobs", type=int, default=None, help="Número de trabajos paralelos (-1 para usar todos los núcleos)"
    )
    parser.add_argument("--seasonal_period", type=int, default=None, help="Período estacional (días hábiles)")

    args = parser.parse_args()

    # Cargar configuración
    config = Config.from_yaml(args.config)

    # Sobrescribir configuración con argumentos de línea de comandos si se proporcionan
    if args.instruments:
        config.instruments = args.instruments
    if args.method:
        config.optimization_method = args.method
    if args.trials:
        config.max_trials = args.trials
    if args.jobs:
        config.n_jobs = args.jobs
    if args.seasonal_period:
        config.seasonal_period = args.seasonal_period

    # Configurar logging
    logger = setup_logging(config)

    # Crear directorios
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)

    # Fijar semillas para reproducibilidad
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Procesar cada instrumento
    logger.info(f"Iniciando entrenamiento de modelos con método: {config.optimization_method}")

    # Versión paralela o secuencial según configuración
    if config.n_jobs != 1 and len(config.instruments) > 1:
        # Procesamiento paralelo por instrumento
        n_jobs = config.n_jobs if config.n_jobs > 0 else None  # None = todos los núcleos
        logger.info(f"Procesando {len(config.instruments)} instrumentos en paralelo con {n_jobs} workers")
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(process_instrument)(instrument, config) for instrument in config.instruments
        )
        all_results = [item for sublist in results_list for item in sublist]
    else:
        # Procesamiento secuencial
        logger.info(f"Procesando {len(config.instruments)} instrumentos secuencialmente")
        all_results = []
        for instrument in config.instruments:
            instrument_results = process_instrument(instrument, config)
            all_results.extend(instrument_results)

    # Guardar resultados combinados
    if all_results:
        # Convertir a DataFrame
        results_df = pd.DataFrame(all_results)

        # Ordenar por fecha para mejor lectura
        if 'date' in results_df.columns:
            results_df['date'] = pd.to_datetime(results_df['date'])
            results_df = results_df.sort_values(['Tipo_Mercado', 'Modelo', 'date'])

        # Guardar archivo combinado
        output_file = config.output_dir / "all_models_predictions.csv"
        results_df.to_csv(output_file, index=False, float_format='%.6f')
        logger.info(f"Resultados combinados guardados en {output_file}")

    logger.info("Procesamiento completo")


if __name__ == "__main__":
    main()
