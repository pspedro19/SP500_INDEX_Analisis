# 📊 SP500 Index Analysis - Data Transformation Pipeline
[![CI](https://github.com/pspedro19/SP500_INDEX_Analisis/actions/workflows/ci.yml/badge.svg)](https://github.com/pspedro19/SP500_INDEX_Analisis/actions/workflows/ci.yml)

Este proyecto aplica una serie de transformaciones modulares y secuenciales sobre datos macroeconómicos para generar datasets entrenables para modelos predictivos, orientados al análisis del índice S&P500.

Puedes revisar la organización completa de carpetas y módulos en [docs/architecture.md](docs/architecture.md).

El proceso está completamente orquestado por el archivo `run_pipeline.py` que ejecuta paso a paso cada transformación ubicada en `src/pipelines/ml/`.

---

## 📁 Estructura del Proyecto

```
SP500_index_analysis/
│
├── data/               # Carpeta de datos (configurable por entorno)
├── src/
│   ├── config/         # Configuración y logging
│   ├── core/           # Utilidades de bajo nivel
│   └── pipelines/      # Pasos del pipeline de ML
├── models/             # Modelos entrenados
├── logs/               # Registros de ejecución
├── reports/            # Informes generados
├── notebooks/          # Jupyter notebooks
├── run_pipeline.py     # Orquestador principal del pipeline
├── tests/              # Pruebas unitarias e integración
├── pyproject.toml      # Metadatos y dependencias
└── requirement.txt     # Dependencias básicas
```

---

## ▶️ Ejecución del pipeline

1. Copia el archivo `.env.example` a `.env` y ajusta las rutas según tu entorno.
2. Instala las dependencias:

```bash
pip install -r requirement.txt
```

3. Ejecuta el pipeline:

```bash
python run_pipeline.py
```

---

## 🧩 Descripción de cada paso

### 🟣 Paso 0 - Preprocesamiento Inicial
**Script:** `src/pipelines/ml/step_0_preprocess.py`  
- Limpia estructuras base y normaliza nombres y fechas.  
- **Input:** Archivos `.xlsx` o `.csv` de `data/raw/`  
- **Output:** Archivos estandarizados a `data/processed/`

---

### 🔵 Paso 1 - Unión de Archivos
**Script:** `src/pipelines/ml/step_1_merge_excels.py`  
- Une múltiples archivos Excel en uno solo.  
- **Input:** Archivos macroeconómicos  
- **Output:** `MERGEDEXCELS.xlsx`

---

### 🔵 Paso 2 - Generación de Categorías
**Script:** `src/pipelines/ml/step_2_generate_categories.py`  
- Clasifica columnas en categorías económicas.  
- **Output:** `MERGEDEXCELS_CATEGORIZADO.xlsx`

---

### 🔵 Paso 3 - Limpieza de Nombres de Columnas
**Script:** `src/pipelines/ml/step_3_clean_columns.py`  
- Elimina redundancias y mejora la trazabilidad de variables.  
- **Output:** `MERGEDEXCELS_CATEGORIZADO_LIMPIO.xlsx`

---

### 🟠 Paso 4 - Transformaciones e Indicadores
**Script:** `src/pipelines/ml/step_4_transform_features.py`  
- Aplica indicadores técnicos como:
  - MoM, YoY, medias móviles, z-score, log-retornos, RSI, Bollinger Bands.  
- **Output:** Datos enriquecidos por categoría.

---

### 🟠 Paso 5 - Eliminación de Relaciones Redundantes
**Script:** `src/pipelines/ml/step_5_remove_relations.py`  
- Elimina multicolinealidad con VIF, y variables con alta correlación o baja varianza.  
- **Output:** Dataset reducido.

---

### 🟡 Paso 6 - Selección de Variables Relevantes (FPI)
**Script:** `src/pipelines/ml/step_6_fpi_selection.py`  
- Selecciona variables clave usando Feature Permutation Importance con CatBoost y validación temporal.  
- **Output:** `EUR_final_FPI.xlsx`

---

### 🟢 Paso 7 - Entrenamiento de Modelos
**Script:** `src/pipelines/ml/step_7_train_models.py`  
- Entrena y optimiza modelos CatBoost, LightGBM, XGBoost, MLP, SVM con Optuna.  
- **Output:**  
  - Modelos `.pkl` en `models/`  
  - Predicciones en `data/final/all_models_predictions.csv`

---

### 🟢 Paso 8 - Preparación de Resultados para Dashboard
**Script:** `src/pipelines/ml/step_8_prepare_output.py`  
- Convierte los resultados a formato `.csv` compatible con Power BI (formato español).  
- **Output:** `outputs/archivo_para_powerbi.csv`

---

## 🛡️ Seguridad y control de versiones

- Variables sensibles están definidas en `.env` (excluido con `.gitignore`)
- Los datos, modelos y logs están excluidos del versionado
- Usa ramas `feature/` o `refactor/` para nuevas funcionalidades

---

## ⚙️ Requisitos

```bash
pip install -r requirement.txt
# o
pip install -e .
```

---

## 🚦 Integración Continua

Las pruebas se ejecutan automáticamente mediante **GitHub Actions** en cada *push* o *pull request* hacia la rama `main`. El flujo está definido en [.github/workflows/ci.yml](.github/workflows/ci.yml) e instala las dependencias desde `requirement.txt` para luego ejecutar `pytest`.

---

## 🧠 Autor

Proyecto desarrollado por Pedro para análisis avanzado del índice S&P500 usando datos macroeconómicos estructurados y modelado basado en aprendizaje automático.
