# Arquitectura del Proyecto

Este documento describe brevemente los principales módulos del repositorio y el flujo general del pipeline de análisis.

## Módulos

- **`run_pipeline.py`**: orquestador principal que ejecuta secuencialmente los pasos del pipeline.
- **`src/config`**: clases de configuración y utilidades de logging. `ProjectConfig` carga rutas desde variables de entorno y crea los directorios requeridos.
- **`src/core`**: utilidades de bajo nivel, incluyendo la gestión de GPU (`GPUManager`) y compatibilidad con configuraciones previas.
- **`src/pipelines/ml`**: contiene los scripts de cada etapa del pipeline de machine learning. Cada archivo `step_X.py` implementa una fase específica del proceso.
- **`src/utils.py`**: funciones auxiliares utilizadas en varios puntos del proyecto.
- **`tests`**: pruebas unitarias y de integración que validan las funciones principales.

## Flujo principal

1. `ProjectConfig.from_env()` lee las variables de entorno (o valores por defecto) para establecer las rutas de trabajo.
2. `run_pipeline.py` inicializa el sistema de logging y asegura que todos los directorios existen.
3. Se ejecutan de forma secuencial los pasos definidos en `src/pipelines/ml`, por ejemplo:
   1. `step_1_merge_excels.py`
   2. `step_2_generate_categories.py`
   3. `step_3_clean_columns.py`
   4. `step_4_transform_features.py`
   5. `step_5_remove_relations.py`
   6. `step_6_fpi_selection.py`
   7. `step_7_0_train_models.py`
   8. `step_7_5_ensemble.py`
   9. `step_8_prepare_output.py`
   10. `step_9_backtest.py`
   11. `step_10_inference.py`
4. Al finalizar cada paso se registran los tiempos y se generan reportes en la carpeta configurada (`reports`).

El objetivo final es generar datasets y modelos listos para análisis del índice S&P500, manteniendo un control estricto de las rutas de datos y métricas mediante las variables de entorno.
