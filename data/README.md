# Datos

Esta carpeta almacena los diferentes estados de los datasets utilizados en el proyecto.

- `raw/` datos originales sin procesar.
- `interim/` transformaciones intermedias generadas durante el pipeline.
- `features/` insumos listos para alimentar los modelos.
- `processed/` resultados finales o datasets limpios.
- `samples/` pequeños subconjuntos para pruebas rápidas.

## Plantilla de diccionario de datos

| Columna | Descripción | Tipo | Ejemplo |
| ------- | ----------- | ---- | ------- |
| `fecha` | Fecha de la observación | `datetime` | `2024-01-01` |
| `valor` | Valor numérico asociado | `float` | `123.45` |

