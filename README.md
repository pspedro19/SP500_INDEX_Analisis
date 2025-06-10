# 📊 SP500 Index Analysis - Data Transformation Pipeline

[![CI](https://github.com/pspedro19/SP500_INDEX_Analisis/actions/workflows/ci.yml/badge.svg)](https://github.com/pspedro19/SP500_INDEX_Analisis/actions/workflows/ci.yml)

Este proyecto aplica una serie de transformaciones modulares y secuenciales sobre datos macroeconómicos para generar datasets entrenables para modelos predictivos, orientados al análisis del índice S&P500.

El proceso inicia con `pipelines/ml/step_0_preprocess.py`, que se ejecuta por separado mediante `sp500 preprocess`.
Luego, `run_pipeline.py` orquesta los pasos 1–10 (incluido el 7.5) desde la carpeta `pipelines/`.

---

## 📁 Estructura del Proyecto

```
SP500_index_analysis/
│
├── data/
│   ├── raw/           # Datos crudos originales
│   ├── processed/     # Datos transformados intermedios
│   └── final/         # Datos finales listos para entrenamiento
│
├── models/            # Modelos entrenados (.pkl)
├── pipelines/         # Scripts de procesamiento por paso
├── logs/              # Logs de ejecución
├── outputs/           # Predicciones listas para visualización (ej. Power BI)
├── notebooks/         # Jupyter notebooks exploratorios
├── run_pipeline.py    # Orquestador principal del pipeline
├── .gitignore         # Exclusión de archivos sensibles
└── requirements.txt   # Dependencias del proyecto
```

---

## ▶️ Ejecución del pipeline

```bash
python run_pipeline.py
```

### Uso de la CLI

También puedes ejecutar los pasos principales mediante el comando `sp500`:

```bash
sp500 preprocess  # preprocesamiento de datos
sp500 train       # entrenamiento de modelos
sp500 infer       # generar predicciones
sp500 backtest   # ejecutar backtests
```
### Ejecutar pasos individualmente

Cada script puede ejecutarse por separado:
```bash
python pipelines/ml/step_0_preprocess.py          # Paso 0
python pipelines/ml/step_1_merge_excels.py       # Paso 1
python pipelines/ml/step_2_generate_categories.py # Paso 2
python pipelines/ml/step_3_clean_columns.py       # Paso 3
python pipelines/ml/step_4_transform_features.py  # Paso 4
python pipelines/ml/step_5_remove_relations.py    # Paso 5
python pipelines/ml/step_6_fpi_selection.py       # Paso 6
python src/sp500_analysis/application/model_training/trainer.py  # Paso 7
python pipelines/ml/step_7_5_ensemble.py          # Paso 7.5
python pipelines/ml/step_8_prepare_output.py      # Paso 8
python -m sp500_analysis.application.services.evaluation_service  # Paso 9
python -m sp500_analysis.application.services.inference_service   # Paso 10
```

---

## 🧩 Descripción de cada paso

### 🟣 Paso 0 - Preprocesamiento Inicial
**Script:** `pipelines/ml/step_0_preprocess.py`  
- Limpia estructuras base y normaliza nombres y fechas.  
- **Input:** Archivos `.xlsx` o `.csv` de `data/raw/`  
- **Output:** Archivos estandarizados a `data/processed/`

---

### 🔵 Paso 1 - Unión de Archivos
**Script:** `pipelines/ml/step_1_merge_excels.py`  
- Une múltiples archivos Excel en uno solo.  
- **Input:** Archivos macroeconómicos  
- **Output:** `MERGEDEXCELS.xlsx`

---

### 🔵 Paso 2 - Generación de Categorías
**Script:** `pipelines/ml/step_2_generate_categories.py`  
- Clasifica columnas en categorías económicas.  
- **Output:** `MERGEDEXCELS_CATEGORIZADO.xlsx`

---

### 🔵 Paso 3 - Limpieza de Nombres de Columnas
**Script:** `pipelines/ml/step_3_clean_columns.py`  
- Elimina redundancias y mejora la trazabilidad de variables.  
- **Output:** `MERGEDEXCELS_CATEGORIZADO_LIMPIO.xlsx`

---

### 🟠 Paso 4 - Transformaciones e Indicadores
**Script:** `pipelines/ml/step_4_transform_features.py`  
- Aplica indicadores técnicos como:
  - MoM, YoY, medias móviles, z-score, log-retornos, RSI, Bollinger Bands.  
- **Output:** Datos enriquecidos por categoría.

---

### 🟠 Paso 5 - Eliminación de Relaciones Redundantes
**Script:** `pipelines/ml/step_5_remove_relations.py`  
- Elimina multicolinealidad con VIF, y variables con alta correlación o baja varianza.  
- **Output:** Dataset reducido.

---

### 🟡 Paso 6 - Selección de Variables Relevantes (FPI)
**Script:** `pipelines/ml/step_6_fpi_selection.py`  
- Selecciona variables clave usando Feature Permutation Importance con CatBoost y validación temporal.  
- **Output:** `EUR_final_FPI.xlsx`

---

### 🟢 Paso 7 - Entrenamiento de Modelos
**Script:** `src/sp500_analysis/application/model_training/trainer.py`  
- Entrena y optimiza modelos CatBoost, LightGBM, XGBoost, MLP, SVM con Optuna.  
- **Output:**  
  - Modelos `.pkl` en `models/`  
  - Predicciones en `data/final/all_models_predictions.csv`

### 🟢 Paso 7.5 - Ensamble de Modelos
**Script:** `pipelines/ml/step_7_5_ensemble.py`
- Combina las predicciones de los modelos base con un enfoque greedy.
- **Output:** `ensemble_greedy.pkl` y `ensemble_info.json`

---

### 🟢 Paso 8 - Preparación de Resultados para Dashboard
**Script:** `pipelines/ml/step_8_prepare_output.py`  
- Convierte los resultados a formato `.csv` compatible con Power BI (formato español).  
- **Output:** `outputs/archivo_para_powerbi.csv`

### 🟢 Paso 9 - Backtest de Estrategias
**Servicio:** `EvaluationService` (`src/sp500_analysis/application/services/evaluation_service.py`)
- Evalúa el desempeño histórico de las predicciones.
- **Output:** métricas y gráficos en `metrics/`.

### 🟢 Paso 10 - Inferencia
**Servicio:** `InferenceService` (`src/sp500_analysis/application/services/inference_service.py`)
- Genera pronósticos usando los modelos entrenados.
- **Output:** `predictions_api.json` y visualizaciones de forecast.

---

## 🛡️ Seguridad y control de versiones

- Variables sensibles están definidas en `.env` (excluido con `.gitignore`)
- Los datos, modelos y logs están excluidos del versionado
- Usa ramas `feature/` o `refactor/` para nuevas funcionalidades

---

## ⚙️ Requisitos

```bash
pip install -r requirements.txt
```

---

## ✅ Integración continua

Este proyecto utiliza **GitHub Actions** para ejecutar automáticamente las
pruebas y las tareas de linting en cada commit. El flujo de trabajo principal se
encuentra en `.github/workflows/ci.yml`.

### Ejecutar pruebas localmente

Para lanzar la misma suite de pruebas que corre en CI, desde la raíz del
repositorio ejecuta:

```bash
make test
```

---

## 🧠 Autor

Proyecto desarrollado por Pedro para análisis avanzado del índice S&P500 usando datos macroeconómicos estructurados y modelado basado en aprendizaje automático.
