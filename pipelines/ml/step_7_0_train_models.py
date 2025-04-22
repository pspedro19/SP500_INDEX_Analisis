import logging
import os
import random
import json
import joblib
import pandas as pd
import numpy as np
import optuna
import glob
import time
from math import sqrt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt



# Importar configuraciones centralizadas
from config import (
    PROJECT_ROOT, MODELS_DIR, TRAINING_DIR, RESULTS_DIR, IMG_CHARTS_DIR,
    DATE_COL, LOCAL_REFINEMENT_DAYS, TRAIN_TEST_SPLIT_RATIO,
    FORECAST_HORIZON_1MONTH, FORECAST_HORIZON_3MONTHS, RANDOM_SEED,
    ensure_directories
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


# Importar funciones de visualización
from ml.utils.plots import (
    plot_real_vs_pred, plot_training_curves, plot_feature_importance
)

# ------------------------------
# CONFIGURACIÓN DE LOGGING
# ------------------------------
log_file = os.path.join(PROJECT_ROOT, "logs", f"train_models_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Asegurar que existen los directorios
ensure_directories()

# ------------------------------
# FUNCIONES AUXILIARES
# ------------------------------
def get_most_recent_file(directory, pattern='*.xlsx'):
    """
    Obtiene el archivo más reciente en un directorio con la extensión especificada.
    
    Args:
        directory (str): Ruta al directorio
        pattern (str): Patrón para buscar archivos
        
    Returns:
        str: Ruta completa al archivo más reciente
    """
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

# ------------------------------
# CONFIGURACIÓN GLOBAL
# ------------------------------
# Constantes de horizonte para alinear con pipeline de series temporales
FORECAST_HORIZON_1MONTH = 20  # Exactamente 20 días hábiles
FORECAST_HORIZON_3MONTHS = 60  # Exactamente 60 días hábiles
LOCAL_REFINEMENT_DAYS = 225  # Número de días para refinamiento local
TRAIN_TEST_SPLIT_RATIO = 0.8  # 80/20 para training/test en refinamiento local

SCALING_REQUIRED_MODELS = {
    "CatBoost": False,
    "LightGBM": False,
    "XGBoost": False,
    "MLP": True,
    "SVM": True
}

class Args:
    # Obtener el archivo de entrada más reciente de la carpeta 3_trainingdata
    input_dir = TRAINING_DIR
    input_file = get_most_recent_file(input_dir)
    
    # Directorios de salida
    output_dir = MODELS_DIR
    output_predictions = os.path.join(RESULTS_DIR, "all_models_predictions.csv")
    
    n_trials = 20
    cv_splits = 5
    gap = FORECAST_HORIZON_1MONTH  # Alineado con horizonte de 20 días (1MONTH)
    tipo_mercado = "S&P500"
    forecast_period = "1MONTH"  # Puede ser "1MONTH" o "3MONTHS"

args = Args()

# Asegurar que existen los directorios
os.makedirs(os.path.dirname(args.output_predictions), exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)

# Semilla para reproducibilidad
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if args.input_file:
    logging.info(f"Usando el archivo más reciente: {args.input_file}")
else:
    logging.error(f"No se encontraron archivos Excel en {args.input_dir}")

# -----------------------------------------------------------------
# FUNCIONES DE OBJETIVO (usando únicamente datos de Training)
# -----------------------------------------------------------------
def objective_catboost(trial, X, y):
    """Optimización de hiperparámetros para CatBoost."""
    start_time = time.perf_counter()
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.05, log=True)
    depth = trial.suggest_int("depth", 4, 10)
    iterations = trial.suggest_int("iterations", 500, 2000)
    
    # Usar el gap correcto según el horizonte de predicción
    tscv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    rmse_scores = []
    
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Asegurar que los índices se mantienen ordenados temporalmente
        train_idx = sorted(train_idx)
        val_idx = sorted(val_idx)
        
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = CatBoostRegressor(
            learning_rate=learning_rate,
            depth=depth,
            iterations=iterations,
            random_seed=RANDOM_SEED,
            verbose=0,
            use_best_model=True
        )
        model.fit(X_train_cv, y_train_cv, eval_set=(X_val_cv, y_val_cv), early_stopping_rounds=50)
        y_pred = model.predict(X_val_cv)
        rmse_scores.append(sqrt(mean_squared_error(y_val_cv, y_pred)))
        logging.debug(f"[CatBoost][fold {fold_i}] best_iteration={model.get_best_iteration()}")
    
    mean_rmse = np.mean(rmse_scores)
    elapsed_time = time.perf_counter() - start_time
    logging.info(f"[CatBoost] Trial {trial.number}: RMSE={mean_rmse:.4f}, Time={elapsed_time:.2f}s")
    return mean_rmse

def objective_lgbm(trial, X, y):
    """Optimización de hiperparámetros para LightGBM."""
    start_time = time.perf_counter()
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.05, log=True)
    max_depth = trial.suggest_int("max_depth", 4, 10)
    n_estimators = trial.suggest_int("n_estimators", 500, 2000)
    
    # Usar el gap correcto según el horizonte de predicción
    tscv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    rmse_scores = []
    
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Asegurar que los índices se mantienen ordenados temporalmente
        train_idx = sorted(train_idx)
        val_idx = sorted(val_idx)
        
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = lgb.LGBMRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=RANDOM_SEED
        )
        model.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], eval_metric="rmse",
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)])
        y_pred = model.predict(X_val_cv)
        rmse_scores.append(sqrt(mean_squared_error(y_val_cv, y_pred)))
        if hasattr(model, "best_iteration_"):
            logging.debug(f"[LightGBM][fold {fold_i}] best_iteration={model.best_iteration_}")
    
    mean_rmse = np.mean(rmse_scores)
    elapsed_time = time.perf_counter() - start_time
    logging.info(f"[LightGBM] Trial {trial.number}: RMSE={mean_rmse:.4f}, Time={elapsed_time:.2f}s")
    return mean_rmse

def objective_xgboost(trial, X, y):
    """Optimización de hiperparámetros para XGBoost."""
    start_time = time.perf_counter()
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.05, log=True)
    max_depth = trial.suggest_int("max_depth", 4, 10)
    n_estimators = trial.suggest_int("n_estimators", 500, 2000)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    
    # Usar el gap correcto según el horizonte de predicción
    tscv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    rmse_scores = []
    
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Asegurar que los índices se mantienen ordenados temporalmente
        train_idx = sorted(train_idx)
        val_idx = sorted(val_idx)
        
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv)
        dval   = xgb.DMatrix(X_val_cv, label=y_val_cv)
        params = {
            "objective": "reg:squarederror",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "seed": RANDOM_SEED
        }
        evals_result = {}
        xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=n_estimators,
                              evals=[(dval, "eval")], early_stopping_rounds=50,
                              evals_result=evals_result, verbose_eval=False)
        best_iter = xgb_model.best_iteration
        y_pred = xgb_model.predict(dval, iteration_range=(0, best_iter+1))
        rmse_scores.append(sqrt(mean_squared_error(y_val_cv, y_pred)))
        logging.debug(f"[XGBoost][fold {fold_i}] best_iter={best_iter}")
    
    mean_rmse = np.mean(rmse_scores)
    elapsed_time = time.perf_counter() - start_time
    logging.info(f"[XGBoost] Trial {trial.number}: RMSE={mean_rmse:.4f}, Time={elapsed_time:.2f}s")
    return mean_rmse

def objective_mlp(trial, X, y):
    """Optimización de hiperparámetros para MLP."""
    start_time = time.perf_counter()
    hidden_neurons = trial.suggest_int("hidden_neurons", 50, 200)
    hidden_layer_sizes = (hidden_neurons, hidden_neurons)
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
    max_iter = trial.suggest_int("max_iter", 200, 1000)
    
    # Usar el gap correcto según el horizonte de predicción
    tscv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    rmse_scores = []
    
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Asegurar que los índices se mantienen ordenados temporalmente
        train_idx = sorted(train_idx)
        val_idx = sorted(val_idx)
        
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             learning_rate_init=learning_rate_init,
                             max_iter=max_iter,
                             random_state=RANDOM_SEED,
                             early_stopping=True,
                             validation_fraction=0.2,
                             n_iter_no_change=25)
        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_val_cv)
        rmse_scores.append(sqrt(mean_squared_error(y_val_cv, y_pred)))
        logging.debug(f"[MLP][fold {fold_i}] n_iter_={model.n_iter_}")
    
    mean_rmse = np.mean(rmse_scores)
    elapsed_time = time.perf_counter() - start_time
    logging.info(f"[MLP] Trial {trial.number}: RMSE={mean_rmse:.4f}, Time={elapsed_time:.2f}s")
    return mean_rmse

def objective_svm(trial, X, y):
    """Optimización de hiperparámetros para SVM."""
    start_time = time.perf_counter()
    C = trial.suggest_float("C", 0.1, 10, log=True)
    epsilon = trial.suggest_float("epsilon", 0.01, 0.5)
    
    # Usar el gap correcto según el horizonte de predicción
    tscv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    rmse_scores = []
    
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Asegurar que los índices se mantienen ordenados temporalmente
        train_idx = sorted(train_idx)
        val_idx = sorted(val_idx)
        
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = SVR(C=C, epsilon=epsilon)
        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_val_cv)
        rmse_scores.append(sqrt(mean_squared_error(y_val_cv, y_pred)))
    
    mean_rmse = np.mean(rmse_scores)
    elapsed_time = time.perf_counter() - start_time
    logging.info(f"[SVM] Trial {trial.number}: RMSE={mean_rmse:.4f}, Time={elapsed_time:.2f}s")
    return mean_rmse

# -----------------------------------------------------------------
# FUNCIONES PARA FORECAST Y ENTRENAMIENTO EXTENDIDO
# -----------------------------------------------------------------
def forecast_future(model, last_row, forecast_horizon=FORECAST_HORIZON_1MONTH):
    """
    Genera predicciones para los próximos 'forecast_horizon' días.
    Se asume que las características permanecen constantes.
    
    Args:
        model: Modelo entrenado
        last_row: Última fila de features disponible
        forecast_horizon: Horizonte de predicción en días hábiles (20 para 1MONTH, 60 para 3MONTHS)
    
    Returns:
        Lista de predicciones para cada día del horizonte
    """
    future_predictions = []
    current_features = last_row.values.reshape(1, -1)
    for _ in range(forecast_horizon):
        pred = model.predict(current_features)[0]
        future_predictions.append(pred)
    return future_predictions

def refinamiento_local(model, X, y, n_days=LOCAL_REFINEMENT_DAYS):
    """
    Realiza refinamiento local del modelo usando los últimos n_days.
    
    Args:
        model: Modelo a refinar
        X: Features completos
        y: Target completo
        n_days: Número de días a usar (225 por defecto)
    
    Returns:
        Modelo refinado
    """
    if len(X) <= n_days:
        local_X = X.copy()
        local_y = y.copy()
    else:
        local_X = X.tail(n_days).copy()
        local_y = y.tail(n_days).copy()
    
    # Split 80/20 sin shuffle para mantener orden temporal
    train_size = int(len(local_X) * TRAIN_TEST_SPLIT_RATIO)
    
    X_train = local_X.iloc[:train_size]
    y_train = local_y.iloc[:train_size]
    
    if hasattr(model, 'fit'):
        model.fit(X_train, y_train)
    
    return model

def calcular_metricas_basicas(y_true, y_pred):
    """
    Calcula métricas básicas de regresión.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        
    Returns:
        dict: Diccionario con métricas calculadas
    """
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # SMAPE (Symmetric Mean Absolute Percentage Error)
    smape = 100 * np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'SMAPE': smape
    }

def optimize_and_train_extended(algo_name, objective_func, X_train, y_train, X_val, y_val, X_test, y_test, 
                               forecast_horizon=FORECAST_HORIZON_1MONTH, top_n=3):
    """
    Optimiza el modelo usando únicamente el conjunto de Training, y luego:
      - Genera las predicciones para cada período:
          • Training: se asigna que "Valor_Predicho" es igual al "Valor_Real".
          • Evaluación: se predice sobre X_val y se calcula el RMSE.
          • Test: se predice sobre X_test y se calcula el RMSE.
      - Genera un forecast para los siguientes días hábiles (con "Valor_Real" NaN).
      - Concatena la serie histórica completa (Training + Evaluación + Test) y el forecast.
      
    Args:
        algo_name (str): Nombre del algoritmo
        objective_func (callable): Función objetivo para optimización
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        X_test, y_test: Datos de test
        forecast_horizon (int): Horizonte de predicción
        top_n (int): Número de mejores trials a considerar
        
    Returns:
        DataFrame: DataFrame con las predicciones históricas y forecast
    """
    start_time = time.perf_counter()
    
    # Optimización usando Training
    logging.info(f"[{algo_name}] Iniciando optimización con {args.n_trials} trials")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_func(trial, X_train, y_train), n_trials=args.n_trials)
    logging.info(f"[{algo_name}] Mejor RMSE en CV (Training): {study.best_value:.4f}")
    logging.info(f"[{algo_name}] Mejores hiperparámetros: {study.best_params}")
    
    # Tiempos de optimización
    optimization_time = time.perf_counter() - start_time
    logging.info(f"[{algo_name}] Tiempo de optimización: {optimization_time:.2f}s")
    
    # Guardar curva de optimización
    trial_metrics = {
        'Trial': [t.number for t in study.trials],
        'RMSE': [t.value for t in study.trials],
        'Modelo': [algo_name] * len(study.trials)
    }
    trial_df = pd.DataFrame(trial_metrics)
    
    # Gráfico de evolución del RMSE por trial
    trials_chart_path = os.path.join(IMG_CHARTS_DIR, f"{algo_name.lower()}_trials_evolution.png")
    plt.figure(figsize=(10, 6))
    plt.plot(trial_df['Trial'], trial_df['RMSE'], marker='o', linestyle='-')
    plt.title(f"Evolución de RMSE en trials - {algo_name}")
    plt.xlabel("Trial #")
    plt.ylabel("RMSE")
    plt.grid(True, alpha=0.3)
    plt.savefig(trials_chart_path, dpi=300)
    plt.close()
    logging.info(f"[{algo_name}] Gráfico de evolución de trials guardado: {trials_chart_path}")
    
    # Seleccionar los top_n mejores trials
    top_trials = sorted(study.trials, key=lambda t: t.value)[:top_n]
    all_results = []
    
    for trial_idx, trial in enumerate(top_trials):
        trial_start_time = time.perf_counter()
        params = trial.params
        logging.info(f"[{algo_name}] Entrenando modelo final con parámetros: {params}")
        
        # Configuración del modelo según el algoritmo
        if algo_name == "CatBoost":
            model = CatBoostRegressor(
                learning_rate=params["learning_rate"],
                depth=params["depth"],
                iterations=params["iterations"],
                random_seed=RANDOM_SEED,
                verbose=0,
                use_best_model=False
            )
        elif algo_name == "LightGBM":
            model = lgb.LGBMRegressor(
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                random_state=RANDOM_SEED
            )
        elif algo_name == "XGBoost":
            model = xgb.XGBRegressor(
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                subsample=params["subsample"],
                random_state=RANDOM_SEED,
                verbosity=0
            )
        elif algo_name == "MLP":
            hidden_layer_sizes = (params["hidden_neurons"], params["hidden_neurons"])
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                learning_rate_init=params["learning_rate_init"],
                max_iter=params["max_iter"],
                random_state=RANDOM_SEED,
                early_stopping=False
            )
        elif algo_name == "SVM":
            model = SVR(C=params["C"], epsilon=params["epsilon"])
        else:
            logging.error(f"Algoritmo {algo_name} no soportado.")
            continue

        # Entrenamiento con datos de Training
        model_train_start = time.perf_counter()
        model.fit(X_train, y_train)
        model_train_time = time.perf_counter() - model_train_start
        logging.info(f"[{algo_name}] Modelo entrenado inicialmente en {model_train_time:.2f}s")
        
        # Refinamiento local con últimos 225 días
        refinement_start = time.perf_counter()
        refined_model = refinamiento_local(model, X_train, y_train, n_days=LOCAL_REFINEMENT_DAYS)
        refinement_time = time.perf_counter() - refinement_start
        logging.info(f"[{algo_name}] Modelo refinado localmente en {refinement_time:.2f}s")

        # Sección Training: predicción sobre datos de entrenamiento
        train_pred_start = time.perf_counter()
        train_preds = refined_model.predict(X_train)
        train_metrics = calcular_metricas_basicas(y_train, train_preds)
        
        df_train = pd.DataFrame({
            "date": X_train.index,
            "Valor_Real": y_train,
            "Valor_Predicho": train_preds,
            "Modelo": algo_name,
            "Version": f"Trial_{trial.number}",
            "RMSE": train_metrics['RMSE'],
            "MAE": train_metrics['MAE'],
            "R2": train_metrics['R2'],
            "SMAPE": train_metrics['SMAPE'],
            "Hyperparámetros": json.dumps(params),
            "Tipo_Mercado": args.tipo_mercado,
            "Periodo": "Training"
        })
        train_pred_time = time.perf_counter() - train_pred_start
        logging.info(f"[{algo_name}] Training predictions generadas en {train_pred_time:.2f}s (RMSE={train_metrics['RMSE']:.4f})")

        # Sección Evaluación: predecir sobre X_val
        eval_pred_start = time.perf_counter()
        pred_eval = refined_model.predict(X_val)
        eval_metrics = calcular_metricas_basicas(y_val, pred_eval)
        
        df_eval = pd.DataFrame({
            "date": X_val.index,
            "Valor_Real": y_val,
            "Valor_Predicho": pred_eval,
            "Modelo": algo_name,
            "Version": f"Trial_{trial.number}",
            "RMSE": eval_metrics['RMSE'],
            "MAE": eval_metrics['MAE'],
            "R2": eval_metrics['R2'],
            "SMAPE": eval_metrics['SMAPE'],
            "Hyperparámetros": json.dumps(params),
            "Tipo_Mercado": args.tipo_mercado,
            "Periodo": "Evaluacion"
        })
        eval_pred_time = time.perf_counter() - eval_pred_start
        logging.info(f"[{algo_name}] Predicciones en Evaluacion generadas en {eval_pred_time:.2f}s (RMSE={eval_metrics['RMSE']:.4f})")

        # Sección Test: predecir sobre X_test
        test_pred_start = time.perf_counter()
        pred_test = refined_model.predict(X_test)
        test_metrics = calcular_metricas_basicas(y_test, pred_test)
        
        df_test = pd.DataFrame({
            "date": X_test.index,
            "Valor_Real": y_test,
            "Valor_Predicho": pred_test,
            "Modelo": algo_name,
            "Version": f"Trial_{trial.number}",
            "RMSE": test_metrics['RMSE'],
            "MAE": test_metrics['MAE'],
            "R2": test_metrics['R2'],
            "SMAPE": test_metrics['SMAPE'],
            "Hyperparámetros": json.dumps(params),
            "Tipo_Mercado": args.tipo_mercado,
            "Periodo": "Test"
        })
        test_pred_time = time.perf_counter() - test_pred_start
        logging.info(f"[{algo_name}] Predicciones en Test generadas en {test_pred_time:.2f}s (RMSE={test_metrics['RMSE']:.4f})")

        # Guardar el modelo refinado
        save_model_start = time.perf_counter()
        os.makedirs(args.output_dir, exist_ok=True)
        model_filename = f"{algo_name.lower()}trial{trial.number}.pkl"
        model_path = os.path.join(args.output_dir, model_filename)
        joblib.dump(refined_model, model_path)
        
        # Guardar versiones adicionales para los mejores trials
        if trial_idx == 0:  # Mejor trial
            best_model_path = os.path.join(args.output_dir, f"{algo_name.lower()}_best.pkl")
            joblib.dump(refined_model, best_model_path)
            logging.info(f"[{algo_name}] Mejor modelo guardado como: {best_model_path}")
        
        save_model_time = time.perf_counter() - save_model_start
        logging.info(f"[{algo_name}] Modelo refinado guardado en {save_model_time:.2f}s: {model_path}")

        # Forecast: predecir los siguientes días hábiles (con Valor_Real NaN)
        forecast_start = time.perf_counter()
        last_row = X_test.iloc[-1]
        future_preds = forecast_future(refined_model, last_row, forecast_horizon)
        last_date = X_test.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')
        df_forecast = pd.DataFrame({
            "date": future_dates,
            "Valor_Real": [np.nan] * forecast_horizon,
            "Valor_Predicho": future_preds,
            "Modelo": algo_name,
            "Version": f"Trial_{trial.number}",
            "RMSE": [np.nan] * forecast_horizon,
            "MAE": [np.nan] * forecast_horizon,
            "R2": [np.nan] * forecast_horizon,
            "SMAPE": [np.nan] * forecast_horizon,
            "Hyperparámetros": json.dumps(params),
            "Tipo_Mercado": args.tipo_mercado,
            "Periodo": "Forecast"
        })
        forecast_time = time.perf_counter() - forecast_start
        logging.info(f"[{algo_name}] Forecast de {forecast_horizon} días generado en {forecast_time:.2f}s")

        # Concatenar la serie histórica completa y el forecast
        df_hist = pd.concat([df_train, df_eval, df_test], ignore_index=True)
        df_all = pd.concat([df_hist, df_forecast], ignore_index=True)
        
        # Generar visualización completa
        model_version = f"{algo_name}_Trial{trial.number}"
        chart_path = os.path.join(IMG_CHARTS_DIR, f"{algo_name.lower()}_trial{trial.number}_full.png")
        plot_real_vs_pred(
            df_all,
            title=f"Histórico y Forecast - {model_version}",
            model_name=algo_name,
            output_path=chart_path
        )
        logging.info(f"[{algo_name}] Gráfico completo guardado: {chart_path}")
        
        # Generar CSV específico para este trial
        trial_csv_path = os.path.join(RESULTS_DIR, f"{args.tipo_mercado.lower()}_{algo_name.lower()}_trial{trial.number}.csv")
        df_all.to_csv(trial_csv_path, index=False)
        logging.info(f"[{algo_name}] CSV para Trial {trial.number} guardado: {trial_csv_path}")
        
        all_results.append(df_all)
        
        # Feature importance si el modelo lo soporta
        if hasattr(refined_model, 'feature_importances_'):
            importance_chart_path = os.path.join(IMG_CHARTS_DIR, f"{algo_name.lower()}_trial{trial.number}_importance.png")
            feature_names = X_train.columns
            
            plot_feature_importance(
                refined_model.feature_importances_,
                feature_names,
                title=f"Feature Importance - {model_version}",
                top_n=20,
                model_name=algo_name,
                output_path=importance_chart_path
            )
            logging.info(f"[{algo_name}] Gráfico de importancia de features guardado: {importance_chart_path}")
        
        # Métricas para cada período
        logging.info(f"[{algo_name}] Trial {trial.number} métricas:")
        logging.info(f"  - Training: RMSE={train_metrics['RMSE']:.4f}, MAE={train_metrics['MAE']:.4f}, R2={train_metrics['R2']:.4f}")
        logging.info(f"  - Validación: RMSE={eval_metrics['RMSE']:.4f}, MAE={eval_metrics['MAE']:.4f}, R2={eval_metrics['R2']:.4f}")
        logging.info(f"  - Test: RMSE={test_metrics['RMSE']:.4f}, MAE={test_metrics['MAE']:.4f}, R2={test_metrics['R2']:.4f}")
        
        # Tiempo total del trial
        trial_time = time.perf_counter() - trial_start_time
        logging.info(f"[{algo_name}] Tiempo total para Trial {trial.number}: {trial_time:.2f}s")
    
    result_df = pd.concat(all_results, ignore_index=True)
    
    # Tiempo total del modelo
    total_time = time.perf_counter() - start_time
    logging.info(f"[{algo_name}] Procesamiento completo en {total_time:.2f}s")
    
    return result_df

# -----------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# -----------------------------------------------------------------
def run_training():
    """
    Ejecuta el proceso completo con mejoras para alinearlo con el pipeline de series temporales:
      - Lectura y preprocesado del dataset.
      - División en conjuntos de Training, Evaluacion y Test.
      - Optimización y entrenamiento para cada modelo y trial.
      - Refinamiento local con los últimos 225 días (como en el pipeline de inferencia).
      - Generación de la serie histórica completa y forecast.
      - Almacenamiento del modelo y generación del archivo CSV final.
    """
    # Tiempo total de ejecución
    total_start_time = time.perf_counter()
    
    # Asegurar que existen los directorios
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(IMG_CHARTS_DIR, exist_ok=True)
    
    # Determinar horizonte según forecast_period
    if args.forecast_period == "1MONTH":
        forecast_horizon = FORECAST_HORIZON_1MONTH
    elif args.forecast_period == "3MONTHS":
        forecast_horizon = FORECAST_HORIZON_3MONTHS
    else:
        forecast_horizon = FORECAST_HORIZON_1MONTH  # Default
    
    # Actualizar gap con el horizonte correcto
    args.gap = forecast_horizon
    logging.info(f"Usando horizonte de {forecast_horizon} días para forecast_period={args.forecast_period}")
    
    # Lectura y ordenamiento
    data_load_start = time.perf_counter()
    df = pd.read_excel(args.input_file)
    df.sort_values(by="date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logging.info("Datos leídos y ordenados por fecha.")
    
    # Limpieza de datos
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    logging.info("Se han imputado los valores NaN e inf (ffill y relleno con 0).")
    
    # Definición de target y features
    target_col = df.columns[-1]
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()
    
    # Manejo de la columna 'date'
    if "date" in X.columns:
        dates = X["date"]
        X.drop(columns=["date"], inplace=True)
        X.index = pd.to_datetime(dates)
    else:
        X.index = pd.to_datetime(X.index)
    
    # Eliminación de columnas de varianza cero
    zero_var_cols = [c for c in X.columns if X[c].std() == 0]
    if zero_var_cols:
        logging.warning(f"Eliminando columnas de varianza 0: {zero_var_cols}")
        X.drop(columns=zero_var_cols, inplace=True)
    
    # Escalado (aplicable para MLP y SVM)
    X_ = X.copy()
    y_ = y.copy()
    if any(model in SCALING_REQUIRED_MODELS and SCALING_REQUIRED_MODELS[model] for model in ["MLP", "SVM"]):
        scaler = StandardScaler()
        X_ = pd.DataFrame(scaler.fit_transform(X_), columns=X_.columns, index=X_.index)
    X_.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_.ffill(inplace=True)
    X_.fillna(0, inplace=True)
    y_.ffill(inplace=True)
    y_.fillna(0, inplace=True)
    if np.isinf(X_.values).any() or np.isnan(X_.values).any():
        logging.warning("Aún hay inf/NaN en X_ tras limpieza. Se detiene el proceso.")
        return
    if np.isinf(y_.values).any() or np.isnan(y_.values).any():
        logging.warning("Aún hay inf/NaN en y_ tras limpieza. Se detiene el proceso.")
        return
    
    data_load_time = time.perf_counter() - data_load_start
    logging.info(f"Tiempo de carga y preprocesamiento: {data_load_time:.2f}s")
    
    # División basada en años (comparable al pipeline de series temporales)
    split_start_time = time.perf_counter()
    
    # Opción 1: División por años, con corte en 2022
    if True:  # Activa esta opción para dividir por años
        logging.info("Realizando división temporal por años (Year<=2022 para train+val)")
        df_años = pd.DataFrame({
            'X': range(len(X_)),
            'date': X_.index,
            'year': X_.index.year
        })
        # Train+val: años <= 2022; Test: años > 2022
        train_val_idx = df_años[df_años['year'] <= 2022]['X'].values
        test_idx = df_años[df_años['year'] > 2022]['X'].values
        
        # Verificar si hay suficientes datos
        if len(train_val_idx) == 0 or len(test_idx) == 0:
            logging.warning("No hay suficientes datos para la división por años. Usando división porcentual.")
        else:
            # División adicional para train/val
            train_val_size = len(train_val_idx)
            val_size = min(forecast_horizon, int(train_val_size * 0.2))
            train_idx = train_val_idx[:-val_size]
            val_idx = train_val_idx[-val_size:]
            
            # Crear conjuntos finales
            X_train = X_.iloc[train_idx]
            y_train = y_.iloc[train_idx]
            X_eval = X_.iloc[val_idx]
            y_eval = y_.iloc[val_idx]
            X_test = X_.iloc[test_idx]
            y_test = y_.iloc[test_idx]
            
            logging.info(f"División por años: Train={len(X_train)} (years <= 2022), "
                         f"Val={len(X_eval)} (last {val_size} days of 2022), "
                         f"Test={len(X_test)} (years > 2022)")
    
    # Opción 2: División con horizonte fijo (original)
    if 'X_train' not in locals():
        logging.info("Usando división con horizonte fijo (original)")
        val_size = test_size = forecast_horizon
        total = len(X_)
        if total < (val_size + test_size + 1):
            logging.error("No hay suficientes datos para hacer el split.")
            return
        train_size = total - (val_size + test_size)
        X_train = X_.iloc[:train_size]
        y_train = y_.iloc[:train_size]
        X_eval = X_.iloc[train_size:train_size+val_size]
        y_eval = y_.iloc[train_size:train_size+val_size]
        X_test = X_.iloc[train_size+val_size:]
        y_test = y_.iloc[train_size+val_size:]
        logging.info(f"Split realizado: Training={len(X_train)}, "
                     f"Evaluacion ({val_size} días)={len(X_eval)}, "
                     f"Test ({test_size} días)={len(X_test)}")
    
    split_time = time.perf_counter() - split_start_time
    logging.info(f"Tiempo para split de datos: {split_time:.2f}s")
    
    # Definición de algoritmos a entrenar
    algorithms = [
        ("CatBoost", objective_catboost),
        ("LightGBM", objective_lgbm),
        ("XGBoost", objective_xgboost),
        ("MLP", objective_mlp),
        ("SVM", objective_svm)
    ]
    
    # Registra el tiempo de inicio para cada algoritmo
    algorithm_times = {}
    final_results = []
    
    for algo_name, obj_func in algorithms:
        algo_start_time = time.perf_counter()
        logging.info(f"=== Optimizando y entrenando {algo_name}... ===")
        
        result_df = optimize_and_train_extended(
            algo_name, obj_func, 
            X_train, y_train, X_eval, y_eval, X_test, y_test, 
            forecast_horizon=forecast_horizon, 
            top_n=3
        )
        
        final_results.append(result_df)
        algo_end_time = time.perf_counter()
        algorithm_times[algo_name] = algo_end_time - algo_start_time
        logging.info(f"=== Completado {algo_name} en {algorithm_times[algo_name]:.2f}s ===")
    
    # Guardar tiempos en un archivo JSON para referencia
    timings_file = os.path.join(RESULTS_DIR, "training_times.json")
    with open(timings_file, 'w') as f:
        json.dump(algorithm_times, f, indent=4)
    
    # Visualizar tiempos de entrenamiento
    plt.figure(figsize=(10, 6))
    plt.bar(algorithm_times.keys(), algorithm_times.values())
    plt.title("Tiempo de entrenamiento por algoritmo")
    plt.xlabel("Algoritmo")
    plt.ylabel("Tiempo (segundos)")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(IMG_CHARTS_DIR, "training_times.png"), dpi=300)
    plt.close()
    
    if final_results:
        predictions_df = pd.concat(final_results, ignore_index=True)
        # Guardar CSV con el histórico completo y el forecast
        columns_to_save = ["date", "Valor_Real", "Valor_Predicho", "Modelo", "Version", 
                           "Periodo", "RMSE", "MAE", "R2", "SMAPE", "Hyperparámetros", "Tipo_Mercado"]
        predictions_df.to_csv(args.output_predictions, index=False, float_format="%.6f")
        logging.info(f"Archivo final de predicciones guardado en {args.output_predictions}")
        
        # Gráfico comparativo de todos los modelos
        all_models_chart = os.path.join(IMG_CHARTS_DIR, "all_models_comparison.png")
        plot_real_vs_pred(
            predictions_df[predictions_df['Periodo'] != 'Forecast'],
            title="Comparación de todos los modelos",
            output_path=all_models_chart
        )
        logging.info(f"Gráfico comparativo de todos los modelos guardado: {all_models_chart}")
        
        # Gráfico comparativo de RMSE por modelo
        rmse_df = predictions_df[predictions_df['Periodo'] == 'Test'].groupby('Modelo')['RMSE'].mean().reset_index()
        rmse_df = rmse_df.sort_values('RMSE')
        
        plt.figure(figsize=(10, 6))
        plt.bar(rmse_df['Modelo'], rmse_df['RMSE'])
        plt.title("RMSE por modelo en conjunto de Test")
        plt.xlabel("Modelo")
        plt.ylabel("RMSE")
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_CHARTS_DIR, "rmse_comparison.png"), dpi=300)
        plt.close()
        
        # Generar informe resumen en formato CSV
        summary_metrics = []
        for modelo in predictions_df['Modelo'].unique():
            for periodo in ['Training', 'Evaluacion', 'Test']:
                df_filtered = predictions_df[(predictions_df['Modelo'] == modelo) & 
                                            (predictions_df['Periodo'] == periodo)]
                if not df_filtered.empty:
                    metrics = {
                        'Modelo': modelo,
                        'Periodo': periodo,
                        'RMSE_medio': df_filtered['RMSE'].mean(),
                        'MAE_medio': df_filtered['MAE'].mean(),
                        'R2_medio': df_filtered['R2'].mean(),
                        'SMAPE_medio': df_filtered['SMAPE'].mean(),
                        'Tiempo_Entrenamiento': algorithm_times.get(modelo, np.nan)
                    }
                    summary_metrics.append(metrics)
        
        summary_df = pd.DataFrame(summary_metrics)
        summary_file = os.path.join(RESULTS_DIR, "resumen_metricas.csv")
        summary_df.to_csv(summary_file, index=False)
        logging.info(f"Resumen de métricas guardado en: {summary_file}")
    else:
        logging.warning("No se han generado resultados finales debido a problemas en el proceso.")
    
    # Tiempo total de ejecución
    total_time = time.perf_counter() - total_start_time
    logging.info(f"Proceso completo terminado en {total_time:.2f}s")
    
    # Imprimir resumen final
    print(f"✅ Entrenamiento completado para {len(algorithms)} algoritmos")
    print(f"✅ Visualizaciones generadas en: {IMG_CHARTS_DIR}")
    print(f"✅ Modelos guardados en: {args.output_dir}")
    print(f"✅ Predicciones consolidadas en: {args.output_predictions}")
    print(f"✅ Tiempo total de ejecución: {total_time:.2f}s")

if __name__ == "__main__":
    run_training()