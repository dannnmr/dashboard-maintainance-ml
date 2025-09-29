# app/main.py
# FastAPI app that serves anomaly predictions.
# Comments in English per your preference.

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import HealthResponse, PredictRequest, PredictResponse, FeaturesResponse
from app.service import AnomalyService

MODEL_DIR = os.environ.get("MODEL_DIR", "modelo/artifacts_anomalia")

app = FastAPI(
    title="Predictive Maintenance Serving API",
    version="1.0.0",
    description="Serves anomaly predictions from AE+IForest ensemble."
)

# CORS (ajusta dominios de tu frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service singleton
service = AnomalyService(model_dir=MODEL_DIR)

@app.get("/health", response_model=HealthResponse)
def health():
    ok, details = service.healthcheck()
    return HealthResponse(status="ok" if ok else "error", details=details)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Expects either:
    - records: list[dict[str, float]] with feature-value pairs; OR
    - gold_parquet_path: a Parquet path (server-side) to batch-predict last N rows.
    """
    try:
        if req.records and len(req.records) > 0:
            df, preds = service.predict_from_records(req.records)
        elif req.gold_parquet_path:
            df, preds = service.predict_from_parquet(req.gold_parquet_path, req.limit_rows)
        else:
            raise HTTPException(status_code=400, detail="Provide either 'records' or 'gold_parquet_path'.")

        return PredictResponse(
            model_version=service.meta.get("model_version", "unknown"),
            feature_order=service.feature_columns,
            results=preds
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/features", response_model=FeaturesResponse)
def features():
    return FeaturesResponse(
        feature_order=service.feature_columns,
        model_version=service.meta.get("model_version", "unknown")
    )
