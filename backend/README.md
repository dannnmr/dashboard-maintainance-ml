## Backend (FastAPI)

### Descripción
API para el ML Dashboard: autenticación de usuarios, ejecución y almacenamiento de predicciones, generación de alertas y reportes.

### Stack
- FastAPI
- SQLAlchemy (async/sync) + PostgreSQL
- JWT Auth

### Estructura
- `app/main.py`: arranque FastAPI, CORS y enrutadores (`/auth`, `/users`, `/predictions`, `/reports`, `/alerts`).
- `app/database.py`: configuración async/sync de SQLAlchemy y dependencias de sesión.
- `app/models.py`: `User`, `Prediction`, `Alert` y enums de estados/roles.
- `app/prediction_service.py`, `app/service.py`: lógica de predicción y persistencia.
- `app/model_loader.py`, `app/utils.py`: carga de modelo y utilidades.
- `modelo/artifacts_anomalia/`: artefactos del modelo (Keras, pickles, metadatos).

### Variables de entorno
# En este punto se define el esquema de la base de datos donde se almacenara
- `DATABASE_URL` (requerida en prod):
  - Async (por defecto): `postgresql+asyncpg://postgres:PASS@localhost:5432/db-proy-ml`
  - Sync derivada: `postgresql://postgres:PASS@localhost:5432/db-proy-ml`

Ejemplo PowerShell:
```powershell
$env:DATABASE_URL = "postgresql+asyncpg://postgres:YOUR_PASS@localhost:5432/db-proy-ml"
```

### Ejecutar local
1) Crear y migrar tablas automáticamente en arranque.
```bash
uvicorn app.main:app --reload --port 8000
```
2) Probar salud:
```bash
curl http://localhost:8000/health
```

### Autenticación
- `POST /auth/login` → devuelve `access_token` (Bearer). Usar en `Authorization: Bearer <token>`.

### Endpoints principales (resumen)
- `GET /health` → estado del servicio/modelo.
- `POST /predict/test` → predicción sin guardar en BD. Body: `{ records?: list[dict], gold_parquet_path?: str, limit_rows?: int }`.
- `POST /predict` → predicción con persistencia en BD (requiere JWT). Acepta `records` o `gold_parquet_path`.
- `POST /predict/gold-data` → predicción desde capa gold (requiere JWT).
- `GET /predictions/...` → historial, estadísticas y resultados de mantenimiento.
- `GET/POST /alerts/...` → gestión de alertas.
- `GET/POST /users/...` → gestión de usuarios.

### Pruebas rápidas
Script de prueba: `backend/test_database_predictions.py` (login, results, cache, historial).
```bash
python backend/test_database_predictions.py
```

### Seguridad y recomendaciones
- Restringir CORS (`allow_origins`) en producción.
- Gestionar credenciales vía secretos; activar SSL en PostgreSQL si aplica.
- Validar tamaño de `input_features`/`prediction_results` almacenados como JSON.

### Despliegue
- Configurar `DATABASE_URL`.
- Ejecutar con Uvicorn/Gunicorn y workers adecuados.
- Monitorear logs y pooling de SQLAlchemy.


