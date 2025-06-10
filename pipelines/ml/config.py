"""Configuración centralizada para todo el pipeline ML."""
from pathlib import Path

from sp500_analysis.config.settings import settings

# ================================
# ROOT DEL PROYECTO
# ================================
PROJECT_ROOT = settings.project_root
ROOT = settings.root

# ================================
# RUTAS PRINCIPALES
# ================================
DATA_DIR = settings.data_dir
RAW_DIR = settings.raw_dir
PREPROCESS_DIR = settings.preprocess_dir
TS_PREP_DIR = settings.ts_prep_dir
PROCESSED_DIR = settings.processed_dir
MODEL_INPUT_DIR = settings.model_input_dir
TS_TRAIN_DIR = settings.ts_train_dir
TRAINING_DIR = settings.training_dir
RESULTS_DIR = settings.results_dir
METRICS_DIR = settings.metrics_dir
LOG_DIR = settings.log_dir
REPORTS_DIR = settings.reports_dir
MODELS_DIR = settings.models_dir

# Subdirectorios para artefactos
IMG_CHARTS_DIR = settings.img_charts_dir
METRICS_CHARTS_DIR = settings.metrics_charts_dir
SUBPERIODS_CHARTS_DIR = settings.subperiods_charts_dir
CSV_REPORTS_DIR = settings.csv_reports_dir

# ================================
# COLUMNAS COMUNES
# ================================
DATE_COL = settings.date_col
ID_COL = settings.id_col
TARGET_SUFFIX = settings.target_suffix

# ================================
# HORIZONTES DE PRONÓSTICO
# ================================
FORECAST_HORIZON_1MONTH = settings.forecast_horizon_1month
FORECAST_HORIZON_3MONTHS = settings.forecast_horizon_3months
LOCAL_REFINEMENT_DAYS = settings.local_refinement_days
TRAIN_TEST_SPLIT_RATIO = settings.train_test_split_ratio

# ================================
# VALIDACIÓN TEMPORAL
# ================================
CV_SPLITS = settings.cv_splits
CV_GAP_1MONTH = settings.cv_gap_1month
CV_GAP_3MONTHS = settings.cv_gap_3months

# ================================
# UMBRALES DE SELECCIÓN DE FEATURES
# ================================
CONSTANT_THRESHOLD = settings.constant_threshold
CORR_THRESHOLD = settings.corr_threshold
VIF_THRESHOLD = settings.vif_threshold
FPI_THRESHOLD = settings.fpi_threshold

# ================================
# CONFIGURACIÓN GENERAL
# ================================
RANDOM_SEED = settings.random_seed
SCORER = settings.scorer

# ================================
# PARÁMETROS CATBOOST POR DEFECTO
# ================================
CATBOOST_PARAMS = settings.catboost_params

# ================================
# COLORES DE ALGORITMOS PARA VISUALIZACIÓN
# ================================
ALGORITHM_COLORS = settings.algorithm_colors

PERIODO_LABELS = settings.periodo_labels

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
