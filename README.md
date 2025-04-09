# 📊 SP500 Index Analysis - Data Transformation Pipeline

Este proyecto aplica una serie de transformaciones modulares y secuenciales sobre datos macroeconómicos para generar datasets entrenables para modelos predictivos, orientados al análisis del índice S&P500.

El proceso está completamente orquestado por el archivo `run_pipeline.py` que ejecuta paso a paso cada transformación desde la carpeta `pipelines/`.

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

---

## 🧩 Descripción de cada paso

### 🟣 Paso 0 - Preprocesamiento Inicial
**Script:** `pipelines/step_0_preprocess.py`  
- Limpia estructuras base y normaliza nombres y fechas.  
- **Input:** Archivos `.xlsx` o `.csv` de `data/raw/`  
- **Output:** Archivos estandarizados a `data/processed/`

---

### 🔵 Paso 1 - Unión de Archivos
**Script:** `pipelines/step_1_merge_excels.py`  
- Une múltiples archivos Excel en uno solo.  
- **Input:** Archivos macroeconómicos  
- **Output:** `MERGEDEXCELS.xlsx`

---

### 🔵 Paso 2 - Generación de Categorías
**Script:** `pipelines/step_2_generate_categories.py`  
- Clasifica columnas en categorías económicas.  
- **Output:** `MERGEDEXCELS_CATEGORIZADO.xlsx`

---

### 🔵 Paso 3 - Limpieza de Nombres de Columnas
**Script:** `pipelines/step_3_clean_columns.py`  
- Elimina redundancias y mejora la trazabilidad de variables.  
- **Output:** `MERGEDEXCELS_CATEGORIZADO_LIMPIO.xlsx`

---

### 🟠 Paso 4 - Transformaciones e Indicadores
**Script:** `pipelines/step_4_transform_features.py`  
- Aplica indicadores técnicos como:
  - MoM, YoY, medias móviles, z-score, log-retornos, RSI, Bollinger Bands.  
- **Output:** Datos enriquecidos por categoría.

---

### 🟠 Paso 5 - Eliminación de Relaciones Redundantes
**Script:** `pipelines/step_5_remove_relations.py`  
- Elimina multicolinealidad con VIF, y variables con alta correlación o baja varianza.  
- **Output:** Dataset reducido.

---

### 🟡 Paso 6 - Selección de Variables Relevantes (FPI)
**Script:** `pipelines/step_6_fpi_selection.py`  
- Selecciona variables clave usando Feature Permutation Importance con CatBoost y validación temporal.  
- **Output:** `EUR_final_FPI.xlsx`

---

### 🟢 Paso 7 - Entrenamiento de Modelos
**Script:** `pipelines/step_7_train_models.py`  
- Entrena y optimiza modelos CatBoost, LightGBM, XGBoost, MLP, SVM con Optuna.  
- **Output:**  
  - Modelos `.pkl` en `models/`  
  - Predicciones en `data/final/all_models_predictions.csv`

---

### 🟢 Paso 8 - Preparación de Resultados para Dashboard
**Script:** `pipelines/step_8_prepare_output.py`  
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
pip install -r requirements.txt
```

---

## 🧠 Autor

Proyecto desarrollado por Stiven para análisis avanzado del índice S&P500 usando datos macroeconómicos estructurados y modelado basado en aprendizaje automático.
