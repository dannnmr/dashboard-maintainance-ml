## Notebooks

Documentación general de notebooks para EDA, preprocesamiento, feature engineering y modelado.
NOTA: varios de estos scripts fueron la base para realizar pruebas piloto para consolidad lo que se tiene en el ETL

### Índice y propósito
- `analisis_exploratorio.ipynb`: EDA general inicial sobre datos de transformadores.
- `analisis_datos.ipynb`: Análisis exploratorio complementario y validaciones.
- `EDA_bronze_to_silver.ipynb`: Revisión de calidad y transformaciones entre Bronze → Silver.
- `eda_transformadores_electricos.ipynb`: EDA específico de variables eléctricas y su comportamiento.

- `pre-procesamiento-transformadores.ipynb`: Preprocesamiento y limpieza de señales (outliers, resampling, imputación).
- `feature_engineering.ipynb`: Generación de features generales (tendencias, ventanas, agregaciones).
- `feature_enginering_transformadores.ipynb`: Feature engineering específico por dominio (térmico/eléctrico).

- `modelado.ipynb`: Experimentos generales de modelos de anomalías.
- `modelado_transformadores.ipynb`: Modelado focalizado en transformadores (selección de variables, métricas).
- `modelado_transformadores copy.ipynb`: Variante/experimentos alternos del cuaderno anterior.

- `ML_LSTM_steps (2).ipynb`: Pipeline de LSTM (definición, entrenamiento, evaluación).
- `ML_LSTM_steps (2) copy.ipynb`: Variante del flujo anterior.
- `ML_TR (3).ipynb`: Experimentos adicionales (Transformers/TimeSeries) o comparativas.

- `entrenamiento.ipynb`: Entrenamiento del modelo base.
- `entrenamientoNuevo.ipynb`: Nueva ronda de entrenamiento con ajustes.
- `entrenamientoNuevo copy.ipynb`: Variante/backup de la configuración nueva.

### Orden sugerido de ejecución
1) EDA: `analisis_exploratorio.ipynb` → `eda_transformadores_electricos.ipynb` → `EDA_bronze_to_silver.ipynb`
2) Preprocesamiento: `pre-procesamiento-transformadores.ipynb`
3) Features: `feature_engineering.ipynb` → `feature_enginering_transformadores.ipynb`
4) Modelado clásico/base: `modelado.ipynb` → `modelado_transformadores.ipynb`
5) Modelos secuenciales/avanzados: `ML_LSTM_steps (2).ipynb` / `ML_TR (3).ipynb`
6) Entrenamiento final: `entrenamiento.ipynb` → `entrenamientoNuevo.ipynb`

### Entradas y salidas
- Entradas esperadas: datos en `data/capa_bronze_v2`, `data/capa_silver`, `data/capa_gold` (Parquet/CSV).
- Salidas posibles: gráficos, métricas, y artefactos intermedios; los artefactos finales del modelo viven en `backend/modelo/artifacts_anomalia/`.

### Recomendaciones de entorno
- Python 3.10+ con dependencias de análisis (pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow/keras si usa LSTM).
- Ejecutar notebooks desde la raíz del repo o configurar `PYTHONPATH` para importar utilidades si fuera necesario.
- Mantener kernels consistentes entre notebooks para reproducibilidad.

### Buenas prácticas
- Fijar semillas aleatorias antes de entrenar y al partir datos.
- Versionar datasets de entrenamiento y registrar metadatos del experimento (fecha, features, hyperparams).
- Exportar artefactos validados a `backend/modelo/artifacts_anomalia/` con versión en metadatos.


