## Data

Repositorio de datasets del proyecto organizados por capas.

### Capas y rutas
- `capa_bronze_v2/readings_v1/`
  - Particionado por `tag=` (p. ej., `tag=temperatura_aceite`) con `_delta_log/`.
  - Formato: Parquet por señal.

- `capa_silver/preprocesamiento_silver/`
  - Particionado `year=/month=`.
  - Formato: Parquet consolidado y limpiado.

- `capa_gold/features_transformador/`
  - Carpetas: `features_complete/`, `features_train/`, `features_valid/` con `year=/month=`.
  - Exports: `transformer_features_*.parquet/csv/txt`.

- `processed/`
  - Documentos auxiliares (p. ej., `transformador_data_metadatos_tecnicos.txt`).

### Convenciones
- Particiones estilo Hive (`year=YYYY/month=M`).
- Parquet snappy por defecto.
- `_delta_log/` conserva historial de operaciones en algunas rutas.

### Línea de tiempo
- Los rangos temporales de extracción inicial se definen en `etl/capa_bronze/config_bronze.py` (`START_TIME`, `END_TIME`).

### Uso típico
1) Bronze: ingesta de señales en `readings_v1/tag=.../*.parquet`.
2) Silver: consolidación a `preprocesamiento_silver/year=/month=/*.parquet`.
3) Gold: construcción de features a `features_transformador/...` y exports.

