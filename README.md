## ML Dashboard - Proyecto General

### Descripción
Plataforma end-to-end para monitoreo de transformadores con pipeline ETL (bronze → silver → gold), modelo de anomalías, API con autenticación y un dashboard web.

### Estructura del repositorio
- `backend/` → API FastAPI, autenticación, predicciones, alertas, reportes.
- `etl/` → pipelines de datos (bronze, silver, gold).
- `data/` → datasets en capas (`capa_bronze_v2`, `capa_silver`, `capa_gold`, `processed`).
- `frontend/` → app Next.js (dashboard, predicciones, alertas, reportes, usuarios).

### Requisitos
- Python 3.10+
- PostgreSQL 14+
- Node.js 20+

### Variables de entorno
- Backend:
  - `DATABASE_URL` (requerida en prod): `postgresql+asyncpg://USER:PASS@HOST:5432/DBNAME`
- Frontend:
  - `NEXT_PUBLIC_API_URL` (por defecto `http://localhost:8000`)

PowerShell (dev):
```powershell
$env:DATABASE_URL = "postgresql+asyncpg://postgres:YOUR_PASS@localhost:5432/db-proy-ml"
$env:NEXT_PUBLIC_API_URL = "http://localhost:8000"
```

### Arranque rápido (dev)
1) Backend (FastAPI)
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```
2) Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev   # http://localhost:3000
```
3) Salud del backend
```bash
curl http://localhost:8000/health
```

### Pipeline de datos (ETL)
- Bronze → ingesta cruda por `tag=` en Parquet.
- Silver → limpieza y reglas; particiones `year=/month=`.
- Gold → features térmicas/eléctricas, labels y validación.

Ejecutar:
```bash
python etl/capa_bronze/main_bronze.py
python etl/capa_silver/main_silver.py
python etl/capa_gold/main_gold.py
```

### Datos
- `data/capa_bronze_v2/readings_v1/tag=.../*.parquet` (+ `_delta_log/`)
- `data/capa_silver/preprocesamiento_silver/year=/month=/*.parquet`
- `data/capa_gold/features_transformador/...` y exports `transformer_features_*`

### API (resumen)
- `POST /auth/login` → JWT.
- `GET /health` → estado.
- `POST /predict/test` → predicción sin guardar.
- `POST /predict` → predicción con guardado (records o `gold_parquet_path`).
- `POST /predict/gold-data` → predicción desde gold (guarda en BD).
- `GET /predictions/*`, `GET/POST /alerts/*`, `GET/POST /users/*`, `GET/POST /reports/*`.

### Frontend
- App Router (Next.js 15). Rutas: `dashboard`, `login`, `predicciones`, `alertas`, `reportes`, `transformadores`, `users`, `test-predictions`.
- `AuthContext` maneja sesión JWT. `src/lib/api.ts` centraliza Axios.

### Seguridad y despliegue
- Restringir CORS en producción.
- Gestionar secretos fuera del repo; activar SSL en PostgreSQL si aplica.
- Desplegar backend con Uvicorn/Gunicorn; frontend build estático.

### Documentación por módulo
- `backend/README.md`
- `etl/README.md`
- `data/README.md`
- `frontend/README.md`


