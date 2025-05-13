"""Configuración centralizada para todo el pipeline ML."""
import os
from pathlib import Path

# ================================
# ROOT DEL PROYECTO
# ================================
# Usamos os para compatibilidad con rutas absolutas
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
# Ruta como Path para operaciones con pathlib
ROOT = Path(PROJECT_ROOT)

# ================================
# RUTAS PRINCIPALES
# ================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "0_raw")
PREPROCESS_DIR = os.path.join(DATA_DIR, "1_preprocess")
TS_PREP_DIR = os.path.join(DATA_DIR, "1_preprocess_ts")
PROCESSED_DIR = os.path.join(DATA_DIR, "2_processed")
MODEL_INPUT_DIR = os.path.join(DATA_DIR, "2_model_input")
TS_TRAIN_DIR = os.path.join(DATA_DIR, "2_trainingdata_ts")
TRAINING_DIR = os.path.join(DATA_DIR, "3_trainingdata")
RESULTS_DIR = os.path.join(DATA_DIR, "4_results")
METRICS_DIR = os.path.join(DATA_DIR, "5_metrics")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Subdirectorios para artefactos
IMG_CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
METRICS_CHARTS_DIR = os.path.join(METRICS_DIR, "charts")
SUBPERIODS_CHARTS_DIR = os.path.join(METRICS_CHARTS_DIR, "subperiods")
CSV_REPORTS_DIR = os.path.join(REPORTS_DIR, "csv")

# ================================
# COLUMNAS COMUNES
# ================================
DATE_COL = "fecha"
ID_COL = "id"
TARGET_SUFFIX = "_Target"

# ================================
# HORIZONTES DE PRONÓSTICO
# ================================
FORECAST_HORIZON_1MONTH = 20   # 20 días hábiles
FORECAST_HORIZON_3MONTHS = 60  # 60 días hábiles
LOCAL_REFINEMENT_DAYS = 225    # Días para refinamiento local
TRAIN_TEST_SPLIT_RATIO = 0.8    # 80/20 split local

# ================================
# VALIDACIÓN TEMPORAL
# ================================
CV_SPLITS = 5
CV_GAP_1MONTH = FORECAST_HORIZON_1MONTH
CV_GAP_3MONTHS = FORECAST_HORIZON_3MONTHS

# ================================
# UMBRALES DE SELECCIÓN DE FEATURES
# ================================
CONSTANT_THRESHOLD = 0.95
CORR_THRESHOLD = 0.85
VIF_THRESHOLD = 10
FPI_THRESHOLD = 0.000001  # Ajustado según snippet

# ================================
# CONFIGURACIÓN GENERAL
# ================================
RANDOM_SEED = 42
SCORER = "neg_mean_squared_error"

# ================================
# PARÁMETROS CATBOOST POR DEFECTO
# ================================
CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "early_stopping_rounds": 50,
    "verbose": 0,
    "random_seed": RANDOM_SEED,
    "n_estimators": 500,
    "learning_rate": 0.01,
    "max_depth": 8
}

# ================================
# COLORES DE ALGORITMOS PARA VISUALIZACIÓN
# ================================
ALGORITHM_COLORS = {
    "CatBoost": "#e377c2",
    "LightGBM": "#7f7f7f",
    "XGBoost": "#bcbd22",
    "MLP": "#17becf",
    "SVM": "#1f77b4",
    "Ensemble": "#ff7f0e"
}

PERIODO_LABELS = {
    "Training": "Entrenamiento",
    "Validation": "Validación",
    "Test": "Test",
    "Forecast": "Pronóstico"
}

DATA_RAW = RAW_DIR
DATA_PREP = PREPROCESS_DIR
CSV_REPORTS = CSV_REPORTS_DIR
# ================================
# FUNCIONES AUXILIARES
# ================================
def ensure_directories():
    """Crea todos los directorios necesarios si no existen."""
    dirs = [
        RAW_DIR, PREPROCESS_DIR, TS_PREP_DIR, PROCESSED_DIR, MODEL_INPUT_DIR,
        TS_TRAIN_DIR, TRAINING_DIR, RESULTS_DIR, METRICS_DIR, LOG_DIR,
        REPORTS_DIR, MODELS_DIR, IMG_CHARTS_DIR, METRICS_CHARTS_DIR,
        CSV_REPORTS_DIR, SUBPERIODS_CHARTS_DIR
    ]
    for directory in dirs:
        # Soporta tanto str como Path
        Path(directory).mkdir(parents=True, exist_ok=True)