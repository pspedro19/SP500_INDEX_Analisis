# 📊 SP500 Index Analysis - Data Transformation Pipeline

[![CI](https://github.com/pspedro19/SP500_INDEX_Analisis/actions/workflows/ci.yml/badge.svg)](https://github.com/pspedro19/SP500_INDEX_Analisis/actions/workflows/ci.yml)

Este proyecto aplica una serie de transformaciones modulares y secuenciales sobre datos macroeconómicos para generar datasets entrenables para modelos predictivos, orientados al análisis del índice S&P500.

El proceso inicia con `pipelines/ml/00_step_preprocess.py`, que **no se ejecuta**
automáticamente al correr `run_pipeline.py`. Este paso debe lanzarse de forma
manual o a través del comando `sp500 preprocess`. Posteriormente,
`run_pipeline.py` orquesta los pasos 1–10 (incluido el 7.5) desde la carpeta
`pipelines/`.

---

## 📁 Estructura del Proyecto

```
SP500_index_analysis/
│
├── data/
│   ├── raw/        # Datos crudos originales (0_raw)
│   ├── interim/    # Transformaciones intermedias (1_preprocess*)
│   ├── features/   # Insumos para modelado (2_model_input)
│   ├── processed/  # Datos limpios o finales
│   └── samples/    # Subconjuntos de prueba
│   # Al ejecutar el pipeline se crearán `3_trainingdata/`,
│   # `4_results/` y `5_metrics/` en esta carpeta.
│
├── pipelines/      # Scripts de procesamiento por paso
├── notebooks/      # Jupyter notebooks exploratorios
├── logs/           # Registros de ejecución
├── run_pipeline.py # Orquestador principal del pipeline
├── .gitignore
└── requirements.txt
```

Las carpetas `models/` y `data/4_results/` se generan automáticamente
al ejecutar el pipeline y no están versionadas.

---

## ▶️ Ejecución del pipeline

```bash
python run_pipeline.py
```

### Uso de la CLI

Instala el paquete en modo editable y luego ejecuta los comandos `sp500` para
invocar los distintos servicios:

```bash
sp500 preprocess  # preprocesamiento de datos
sp500 train       # entrenamiento de modelos
sp500 infer       # generar predicciones
sp500 backtest   # ejecutar backtests
```

Para ver todas las opciones disponibles también puedes ejecutar:

```bash
sp500 --help
```
### Ejecutar pasos individualmente

Cada script puede ejecutarse por separado:
```bash
python pipelines/ml/00_step_preprocess.py                 # Paso 0
python pipelines/ml/01_step_merge_excels.py               # Paso 1
python pipelines/ml/02_step_generate_categories.py        # Paso 2
python pipelines/ml/03_step_clean_columns.py              # Paso 3
python pipelines/ml/04_step_transform_features.py         # Paso 4
python pipelines/ml/05_step_remove_relations.py           # Paso 5
python pipelines/ml/06_step_fpi_selection.py              # Paso 6
python pipelines/ml/06a_step_filtro_20days.py             # Paso 6a
python pipelines/ml/07_step_train_models.py               # Paso 7
python pipelines/ml/07a_step_apply_inverse_transform.py   # Paso 7a
python pipelines/ml/07b_step_ensemble.py                  # Paso 7b
python pipelines/ml/07c_step_Calculo_Valor_SP500.py       # Paso 7c
python pipelines/ml/07d_step_Transform_to_PowerBI.py      # Paso 7d
python pipelines/ml/08_step_prepare_output.py             # Paso 8
python pipelines/ml/09_step_backtest.py                   # Paso 9
python pipelines/ml/10_step_inference.py                  # Paso 10
```

---

## 🧩 Descripción de cada paso

### 🟣 Paso 0 - Preprocesamiento Inicial
**Script:** `pipelines/ml/00_step_preprocess.py`
- Limpia estructuras base y normaliza nombres y fechas.  
- **Input:** Archivos `.xlsx` o `.csv` de `data/raw/`  
- **Output:** Archivos estandarizados a `data/processed/`

---

### 🔵 Paso 1 - Unión de Archivos
**Script:** `pipelines/ml/01_step_merge_excels.py`
- Une múltiples archivos Excel en uno solo.  
- **Input:** Archivos macroeconómicos  
- **Output:** `MERGEDEXCELS.xlsx`

---

### 🔵 Paso 2 - Generación de Categorías
**Script:** `pipelines/ml/02_step_generate_categories.py`
- Clasifica columnas en categorías económicas.  
- **Output:** `MERGEDEXCELS_CATEGORIZADO.xlsx`

---

### 🔵 Paso 3 - Limpieza de Nombres de Columnas
**Script:** `pipelines/ml/03_step_clean_columns.py`
- Elimina redundancias y mejora la trazabilidad de variables.  
- **Output:** `MERGEDEXCELS_CATEGORIZADO_LIMPIO.xlsx`

---

### 🟠 Paso 4 - Transformaciones e Indicadores
**Script:** `pipelines/ml/04_step_transform_features.py`
- Aplica indicadores técnicos como:
  - MoM, YoY, medias móviles, z-score, log-retornos, RSI, Bollinger Bands.  
- **Output:** Datos enriquecidos por categoría.

---

### 🟠 Paso 5 - Eliminación de Relaciones Redundantes
**Script:** `pipelines/ml/05_step_remove_relations.py`
- Elimina multicolinealidad con VIF, y variables con alta correlación o baja varianza.  
- **Output:** Dataset reducido.

---

### 🟡 Paso 6 - Selección de Variables Relevantes (FPI)
**Script:** `pipelines/ml/06_step_fpi_selection.py`
- Selecciona variables clave usando Feature Permutation Importance con CatBoost y validación temporal.
- **Output:** `EUR_final_FPI.xlsx`

---

### 🟡 Paso 6a - Filtro de 20 días
**Script:** `pipelines/ml/06a_step_filtro_20days.py`
- Reduce el ruido eliminando filas muy cercanas en el tiempo.
- **Output:** Dataset filtrado.

---

### 🟢 Paso 7 - Entrenamiento de Modelos
**Script:** `pipelines/ml/07_step_train_models.py`
- Entrena y optimiza modelos CatBoost, LightGBM, XGBoost, MLP, SVM con Optuna.
- **Output:**
  - Modelos `.pkl` en `models/` (carpeta creada al ejecutar el pipeline)
  - Predicciones en `data/4_results/all_models_predictions.csv`

### 🟢 Paso 7a - Aplicar Transformación Inversa
**Script:** `pipelines/ml/07a_step_apply_inverse_transform.py`
- Restaura las predicciones a su escala original.
- **Output:** `predicciones_reales.csv`

### 🟢 Paso 7b - Ensamble de Modelos
**Script:** `pipelines/ml/07b_step_ensemble.py`
- Combina las predicciones de los modelos base con un enfoque greedy.
- **Output:** `ensemble_greedy.pkl` y `ensemble_info.json`

### 🟢 Paso 7c - Cálculo del Valor del S&P500
**Script:** `pipelines/ml/07c_step_Calculo_Valor_SP500.py`
- Convierte los retornos pronosticados en valores del índice.
- **Output:** `valor_sp500.csv`

### 🟢 Paso 7d - Formato compatible con Power BI
**Script:** `pipelines/ml/07d_step_Transform_to_PowerBI.py`
- Adapta los CSV al formato regional español para Power BI.
- **Output:** `data/4_results/archivo_powerbi_es.csv`

---

### 🟢 Paso 8 - Preparación de Resultados para Dashboard
**Script:** `pipelines/ml/08_step_prepare_output.py`
- Convierte los resultados a formato `.csv` compatible con Power BI (formato español).
- **Output:** `data/4_results/archivo_para_powerbi.csv`

### 🟢 Paso 9 - Backtest de Estrategias
**Script:** `pipelines/ml/09_step_backtest.py`
- Evalúa el desempeño histórico de las predicciones.
- **Output:** métricas y gráficos en `data/5_metrics/`.

### 🟢 Paso 10 - Inferencia
**Script:** `pipelines/ml/10_step_inference.py`
- Genera pronósticos usando los modelos entrenados.
- **Output:** `predictions_api.json` y visualizaciones de forecast.

---

## 🛡️ Seguridad y control de versiones

- Variables sensibles están definidas en `.env` (excluido con `.gitignore`)
- Los datos, los modelos y los logs generados se excluyen del versionado
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
encuentra en `.github/workflows/ci.yml` y ejecuta `pip install -e .[dev]`,
`ruff check .`, `black --check .` y `pytest`.

### Ejecutar pruebas localmente

Para lanzar la misma suite de pruebas que corre en CI, desde la raíz del
repositorio ejecuta:

```bash
make test
```

---

## 🧠 Autor

Proyecto desarrollado por Pedro para análisis avanzado del índice S&P500 usando datos macroeconómicos estructurados y modelado basado en aprendizaje automático.

## ❌ Scripts eliminados

El script `pipelines/ml/limpiar_ultimo_nan.py` se eliminó del repositorio al no formar parte del flujo de procesamiento actual.
