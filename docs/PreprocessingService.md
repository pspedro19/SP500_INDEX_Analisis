# PreprocessingService

`PreprocessingService` es un servicio de la capa de aplicación encargado de ejecutar el paso 0 del pipeline (`pipelines/ml/00_step_preprocess.py`).

Se registra en el contenedor de dependencias (`container.py`) y es invocado por la CLI mediante `sp500 preprocess`. Su función es preparar las carpetas de `data/` con los archivos normalizados que luego consume `run_pipeline.py` para orquestar los pasos 1--10.
