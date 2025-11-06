## ETL (Bronze → Silver → Gold)

Pipeline de datos para construir señales limpias y features para el modelo.

### Capas
- **Bronze**: extracción/ingesta de lecturas crudas por `tag=`.
  - Código: `etl/capa_bronze/`
  - Configuración: `config_bronze.py` (TAGS, ventanas `START_TIME`/`END_TIME`, rutas)
  - Salida: `data/capa_bronze_v2/readings_v1/tag=.../*.parquet` (+ `_delta_log/`)

- **Silver**: limpieza, transformaciones y reglas de negocio.
  - Código: `etl/capa_silver/` (`transform_silver.py`, `rules_silver.py`, `classify_silver.py`, `save_silver.py`)
  - Salida: `data/capa_silver/preprocesamiento_silver/year=/month=/*.parquet`

- **Gold**: features térmicas/eléctricas, labels, validación y finalización.
  - Código: `etl/capa_gold/` (`features_thermal.py`, `features_electrical.py`, `labels_gold.py`, `pipeline_gold.py`, `finalize_gold.py`)
  - Salida: `data/capa_gold/features_transformador/...` y archivos agregados (`*.parquet`, `*.csv`)

### Ejecución
Desde la raíz del repo:
```bash
# Bronze
python etl/capa_bronze/main_bronze.py

# Silver
python etl/capa_silver/main_silver.py

# Gold
python etl/capa_gold/main_gold.py
```

### Parámetros clave (Bronze)
- `TAGS`: mapeo semántico a nombres de señales.
- `START_TIME` / `END_TIME`: ventana histórica de extracción.
- Directorios: detectados desde `REPO_ROOT`; crea si no existen.

### Estructura (resumen)
- `capa_bronze/`: `extract_bronze.py`, `storage_bronze.py`, `logger_bronze.py`, `main_bronze.py`, `requirements.txt`.
- `capa_silver/`: `bronze_silver.py`, `transform_silver.py`, `rules_silver.py`, `classify_silver.py`, `save_silver.py`, `report_silver.py`, `main_silver.py`, `config_silver_v2.py`.
- `capa_gold/`: `features_thermal.py`, `features_electrical.py`, `labels_gold.py`, `pipeline_gold.py`, `validate_gold.py`, `finalize_gold.py`, `main_gold.py`, `config_gold_v2.py`.

### Buenas prácticas
- Mantener consistencia de particiones (`year=/month=`) y tipos.
- Versionar cambios de reglas y features en `config_*` y `pipeline_*`.
- Validar integridad en cada salto de capa (archivos y conteos esperados).


