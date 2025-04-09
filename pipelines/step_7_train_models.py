
import logging
import os
import random
import json
import joblib
import pandas as pd
import numpy as np
import optuna
from math import sqrt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# ------------------------------
# CONFIGURACIÓN GLOBAL
# ------------------------------
SCALING_REQUIRED_MODELS = {
    "CatBoost": False,
    "LightGBM": False,
    "XGBoost": False,
    "MLP": True,
    "SVM": True
}

class Args:
    input_file = "Data/processed/EUR_final_FPI.xlsx"
    output_dir = "models/"
    output_predictions = "Data/final/all_models_predictions.csv"
    n_trials = 20
    cv_splits = 5
    gap = 20
    tipo_mercado = "EUR"

args = Args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
random.seed(42)
np.random.seed(42)

# -----------------------------------------------------------------
# FUNCIONES DE OBJETIVO (usando únicamente datos de Training)
# -----------------------------------------------------------------
def objective_catboost(trial, X, y):
    """Optimización de hiperparámetros para CatBoost."""
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.05, log=True)
    depth = trial.suggest_int("depth", 4, 10)
    iterations = trial.suggest_int("iterations", 500, 2000)
    
    tscv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    rmse_scores = []
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = CatBoostRegressor(
            learning_rate=learning_rate,
            depth=depth,
            iterations=iterations,
            random_seed=42,
            verbose=0,
            use_best_model=True
        )
        model.fit(X_train_cv, y_train_cv, eval_set=(X_val_cv, y_val_cv), early_stopping_rounds=50)
        y_pred = model.predict(X_val_cv)
        rmse_scores.append(sqrt(mean_squared_error(y_val_cv, y_pred)))
        logging.debug(f"[CatBoost][fold {fold_i}] best_iteration={model.get_best_iteration()}")
    return np.mean(rmse_scores)

def objective_lgbm(trial, X, y):
    """Optimización de hiperparámetros para LightGBM."""
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.05, log=True)
    max_depth = trial.suggest_int("max_depth", 4, 10)
    n_estimators = trial.suggest_int("n_estimators", 500, 2000)
    
    tscv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    rmse_scores = []
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = lgb.LGBMRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=42
        )
        model.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], eval_metric="rmse",
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)])
        y_pred = model.predict(X_val_cv)
        rmse_scores.append(sqrt(mean_squared_error(y_val_cv, y_pred)))
        if hasattr(model, "best_iteration_"):
            logging.debug(f"[LightGBM][fold {fold_i}] best_iteration={model.best_iteration_}")
    return np.mean(rmse_scores)

def objective_xgboost(trial, X, y):
    """Optimización de hiperparámetros para XGBoost."""
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.05, log=True)
    max_depth = trial.suggest_int("max_depth", 4, 10)
    n_estimators = trial.suggest_int("n_estimators", 500, 2000)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    
    tscv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    rmse_scores = []
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv)
        dval   = xgb.DMatrix(X_val_cv, label=y_val_cv)
        params = {
            "objective": "reg:squarederror",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "seed": 42
        }
        evals_result = {}
        xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=n_estimators,
                              evals=[(dval, "eval")], early_stopping_rounds=50,
                              evals_result=evals_result, verbose_eval=False)
        best_iter = xgb_model.best_iteration
        y_pred = xgb_model.predict(dval, iteration_range=(0, best_iter+1))
        rmse_scores.append(sqrt(mean_squared_error(y_val_cv, y_pred)))
        logging.debug(f"[XGBoost][fold {fold_i}] best_iter={best_iter}")
    return np.mean(rmse_scores)

def objective_mlp(trial, X, y):
    """Optimización de hiperparámetros para MLP."""
    hidden_neurons = trial.suggest_int("hidden_neurons", 50, 200)
    hidden_layer_sizes = (hidden_neurons, hidden_neurons)
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
    max_iter = trial.suggest_int("max_iter", 200, 1000)
    
    tscv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    rmse_scores = []
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             learning_rate_init=learning_rate_init,
                             max_iter=max_iter,
                             random_state=42,
                             early_stopping=True,
                             validation_fraction=0.2,
                             n_iter_no_change=25)
        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_val_cv)
        rmse_scores.append(sqrt(mean_squared_error(y_val_cv, y_pred)))
        logging.debug(f"[MLP][fold {fold_i}] n_iter_={model.n_iter_}")
    return np.mean(rmse_scores)

def objective_svm(trial, X, y):
    """Optimización de hiperparámetros para SVM."""
    C = trial.suggest_float("C", 0.1, 10, log=True)
    epsilon = trial.suggest_float("epsilon", 0.01, 0.5)
    
    tscv = TimeSeriesSplit(n_splits=args.cv_splits, gap=args.gap)
    rmse_scores = []
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        model = SVR(C=C, epsilon=epsilon)
        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_val_cv)
        rmse_scores.append(sqrt(mean_squared_error(y_val_cv, y_pred)))
    return np.mean(rmse_scores)

# -----------------------------------------------------------------
# FUNCIONES PARA FORECAST Y ENTRENAMIENTO EXTENDIDO
# -----------------------------------------------------------------
def forecast_future(model, last_row, forecast_horizon=21):
    """
    Genera predicciones para los próximos 'forecast_horizon' días.
    Se asume que las características permanecen constantes.
    """
    future_predictions = []
    current_features = last_row.values.reshape(1, -1)
    for _ in range(forecast_horizon):
        pred = model.predict(current_features)[0]
        future_predictions.append(pred)
    return future_predictions

def optimize_and_train_extended(algo_name, objective_func, X_train, y_train, X_val, y_val, X_test, y_test, forecast_horizon=21, top_n=3):
    """
    Optimiza el modelo usando únicamente el conjunto de Training, y luego:
      - Genera las predicciones para cada período:
          • Training: se asigna que "Valor_Predicho" es igual al "Valor_Real".
          • Evaluación: se predice sobre X_val y se calcula el RMSE.
          • Test: se predice sobre X_test y se calcula el RMSE.
      - Genera un forecast para los siguientes 21 días (con "Valor_Real" NaN).
      - Concatena la serie histórica completa (Training + Evaluación + Test) y el forecast.
    """
    # Optimización usando Training
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_func(trial, X_train, y_train), n_trials=args.n_trials)
    logging.info(f"[{algo_name}] Mejor RMSE en CV (Training): {study.best_value:.4f}")
    logging.info(f"[{algo_name}] Mejores hiperparámetros: {study.best_params}")
    top_trials = sorted(study.trials, key=lambda t: t.value)[:top_n]
    all_results = []
    
    for trial in top_trials:
        params = trial.params
        logging.info(f"[{algo_name}] Entrenando modelo final con parámetros: {params}")
        # Configuración del modelo según el algoritmo
        if algo_name == "CatBoost":
            model = CatBoostRegressor(
                learning_rate=params["learning_rate"],
                depth=params["depth"],
                iterations=params["iterations"],
                random_seed=42,
                verbose=0,
                use_best_model=False
            )
        elif algo_name == "LightGBM":
            model = lgb.LGBMRegressor(
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                random_state=42
            )
        elif algo_name == "XGBoost":
            model = xgb.XGBRegressor(
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                subsample=params["subsample"],
                random_state=42,
                verbosity=0
            )
        elif algo_name == "MLP":
            hidden_layer_sizes = (params["hidden_neurons"], params["hidden_neurons"])
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                learning_rate_init=params["learning_rate_init"],
                max_iter=params["max_iter"],
                random_state=42,
                early_stopping=False
            )
        elif algo_name == "SVM":
            model = SVR(C=params["C"], epsilon=params["epsilon"])
        else:
            logging.error(f"Algoritmo {algo_name} no soportado.")
            continue

        # Entrenamiento solo con datos de Training
        model.fit(X_train, y_train)
        logging.info(f"[{algo_name}] Modelo entrenado con datos de Training.")

        # Sección Training: (predicción = real)
        df_train = pd.DataFrame({
            "date": X_train.index,
            "Valor_Real": y_train,
            "Valor_Predicho": y_train,
            "Modelo": algo_name,
            "Version": f"Trial_{trial.number}",
            "RMSE": np.nan,
            "Hyperparámetros": json.dumps(params),
            "Tipo_Mercado": args.tipo_mercado,
            "Periodo": "Training"
        })

        # Sección Evaluación: predecir sobre X_val
        pred_eval = model.predict(X_val)
        rmse_eval = sqrt(mean_squared_error(y_val, pred_eval))
        df_eval = pd.DataFrame({
            "date": X_val.index,
            "Valor_Real": y_val,
            "Valor_Predicho": pred_eval,
            "Modelo": algo_name,
            "Version": f"Trial_{trial.number}",
            "RMSE": rmse_eval,
            "Hyperparámetros": json.dumps(params),
            "Tipo_Mercado": args.tipo_mercado,
            "Periodo": "Evaluacion"
        })
        logging.info(f"[{algo_name}] Predicciones en Evaluacion generadas (RMSE={rmse_eval:.4f}).")

        # Sección Test: predecir sobre X_test
        pred_test = model.predict(X_test)
        rmse_test = sqrt(mean_squared_error(y_test, pred_test))
        df_test = pd.DataFrame({
            "date": X_test.index,
            "Valor_Real": y_test,
            "Valor_Predicho": pred_test,
            "Modelo": algo_name,
            "Version": f"Trial_{trial.number}",
            "RMSE": rmse_test,
            "Hyperparámetros": json.dumps(params),
            "Tipo_Mercado": args.tipo_mercado,
            "Periodo": "Test"
        })
        logging.info(f"[{algo_name}] Predicciones en Test generadas (RMSE={rmse_test:.4f}).")

        # Guardar el modelo en el directorio principal y en la subcarpeta "best_models"
        os.makedirs(args.output_dir, exist_ok=True)
        model_filename = f"{algo_name.lower()}trial{trial.number}.pkl"
        joblib.dump(model, os.path.join(args.output_dir, model_filename))
        best_models_dir = os.path.join(args.output_dir, "best_models")
        os.makedirs(best_models_dir, exist_ok=True)
        joblib.dump(model, os.path.join(best_models_dir, model_filename))
        logging.info(f"[{algo_name}] Modelo guardado en '{model_filename}' para Trial_{trial.number}.")

        # Forecast: predecir 21 días siguientes (con Valor_Real NaN)
        last_row = X_test.iloc[-1]
        future_preds = forecast_future(model, last_row, forecast_horizon)
        last_date = X_test.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')
        df_forecast = pd.DataFrame({
            "date": future_dates,
            "Valor_Real": [np.nan] * forecast_horizon,
            "Valor_Predicho": future_preds,
            "Modelo": algo_name,
            "Version": f"Trial_{trial.number}",
            "RMSE": [np.nan] * forecast_horizon,
            "Hyperparámetros": json.dumps(params),
            "Tipo_Mercado": args.tipo_mercado,
            "Periodo": "Forecast"
        })
        logging.info(f"[{algo_name}] Forecast de {forecast_horizon} días generado.")

        # Concatenar la serie histórica completa y el forecast
        df_hist = pd.concat([df_train, df_eval, df_test], ignore_index=True)
        df_all = pd.concat([df_hist, df_forecast], ignore_index=True)
        all_results.append(df_all)
    
    return pd.concat(all_results, ignore_index=True)

# -----------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# -----------------------------------------------------------------
def run_training():
    """
    Ejecuta el proceso completo:
      - Lectura y preprocesado del dataset.
      - División en:
          • Training: Primeros T - 2*(21*n_meses) registros.
          • Evaluacion: Siguientes 21*n_meses registros.
          • Test: Últimos 21*n_meses registros.
      - Optimización y entrenamiento para cada modelo y trial.
      - Generación de la serie histórica completa (Training + Evaluacion + Test) y forecast (21 días).
      - Almacenamiento del modelo y generación del archivo CSV final.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Lectura y ordenamiento
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
    
    # División en Training, Evaluacion y Test
    n_meses = 1  # Configurable: 1 para 21 días, 2 para 42 días, etc.
    val_size = test_size = 21 * n_meses
    total = len(X_)
    if total < (val_size + test_size + 1):
        logging.error("No hay suficientes datos para hacer el split.")
        return
    train_size = total - (val_size + test_size)
    X_train = X_.iloc[:train_size]
    y_train = y_.iloc[:train_size]
    X_eval = X_.iloc[train_size:train_size+val_size]
    y_eval = y_.iloc[train_size:train_size+val_size]
    X_test = X_.iloc[train_size+val_size:train_size+val_size+test_size]
    y_test = y_.iloc[train_size+val_size:train_size+val_size+test_size]
    logging.info(f"Split realizado: Training={len(X_train)}, Evaluacion ({val_size} días)={len(X_eval)}, Test ({test_size} días)={len(X_test)}")
    
    # Definición de algoritmos a entrenar
    algorithms = [
        ("CatBoost", objective_catboost),
        ("LightGBM", objective_lgbm),
        ("XGBoost", objective_xgboost),
        ("MLP", objective_mlp),
        ("SVM", objective_svm)
    ]
    final_results = []
    for algo_name, obj_func in algorithms:
        logging.info(f"=== Optimizando y entrenando {algo_name}... ===")
        result_df = optimize_and_train_extended(algo_name, obj_func, X_train, y_train, X_eval, y_eval, X_test, y_test, forecast_horizon=21, top_n=3)
        final_results.append(result_df)
    
    if final_results:
        predictions_df = pd.concat(final_results, ignore_index=True)
        # Guardar CSV con el histórico completo y el forecast
        columns_to_save = ["date", "Valor_Real", "Valor_Predicho", "Modelo", "Version", "Periodo", "RMSE", "Hyperparámetros", "Tipo_Mercado"]
        predictions_df.to_csv(args.output_predictions, index=False, float_format="%.6f")
        logging.info(f"Archivo final de predicciones guardado en {args.output_predictions}")
    else:
        logging.warning("No se han generado resultados finales debido a problemas en el proceso.")

if __name__ == "__main__":
    run_training()