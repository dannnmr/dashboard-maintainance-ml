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

# ========== ENDPOINT PARA LEER RESULTADOS DE TU ETL ==========
@app.get("/maintenance/results")
def get_maintenance_results():
    """
    Lee los últimos datos generados por tu ETL y ejecuta inferencia.
    No necesita input del frontend - solo lee y procesa.
    """
    try:
        import sys
        from pathlib import Path
        import pandas as pd
        
        # Agregar el directorio modelo al path
        modelo_path = Path(__file__).parent.parent / "modelo"
        sys.path.append(str(modelo_path))
        
        from infer import load_artifacts, infer_from_last_24h
        
        # Cargar artefactos del modelo
        ae, scaler_ae, feature_cols, meta, medians = load_artifacts()
        
        # Leer los datos más recientes del ETL
        # Buscar el parquet más reciente en features_complete
        data_path = Path(__file__).parent.parent.parent / "data" / "capa_gold" / "features_transformador"
        
        # Opción 1: Leer del CSV más reciente si existe
        csv_files = list(data_path.glob("transformer_features_complete_*.csv"))
        if csv_files:
            latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
            print(f"📊 Leyendo datos de: {latest_csv.name}")
            df = pd.read_csv(latest_csv)
        else:
            # Opción 2: Leer del parquet más reciente
            parquet_files = list(data_path.glob("transformer_features_complete_*.parquet"))
            if parquet_files:
                latest_parquet = max(parquet_files, key=lambda x: x.stat().st_mtime)
                print(f"📊 Leyendo datos de: {latest_parquet.name}")
                df = pd.read_parquet(latest_parquet)
            else:
                raise HTTPException(status_code=404, detail="No se encontraron datos del ETL")
        
        print(f"📈 Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
        
        # Tomar las últimas 24 filas (LOOKBACK) para inferencia
        df_last_24 = df.tail(24)
        
        # Ejecutar tu función original sin modificaciones
        result = infer_from_last_24h(df_last_24)
        
        # Formatear respuesta según tu estructura especificada
        response = {
            "model_version": meta.get("model_version", "ae_lstm_v1"),
            "feature_order": feature_cols,
            "results": [{
                "index": len(df) - 1,  # Índice de la última fila
                "score": float(result["score"]),
                "label": "ANOMALY" if result["pred"] == 1 else "NORMAL"
            }],
            "data_info": {
                "total_rows": len(df),
                "last_24_rows_used": len(df_last_24),
                "threshold_used": result["operate_thr"],
                "timestamp": df.index[-1].isoformat() if hasattr(df.index, 'to_pydatetime') else "unknown"
            }
        }
        
        print(f"✅ Resultado: score={result['score']:.4f}, pred={result['pred']}")
        return response
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error procesando datos del ETL: {str(e)}")