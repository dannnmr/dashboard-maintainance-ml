# app/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import Session
import time
from datetime import datetime

from .database import async_engine, Base, get_sync_db
from .service import AnomalyService
from .schemas import (
    HealthResponse, 
    PredictRequest, 
    PredictResponse, 
    FeaturesResponse,
    PredictionCreate,
    PredictionUpdate,
    UserResponse
)
from .prediction_service import PredictionService
from .auth_routes import router as auth_router
from .user_routes import router as user_router
from .prediction_routes import router as prediction_router
from .simple_routes import router as simple_router
from .report_routes import router as report_router
from .auth import get_current_active_user

# Initialize FastAPI app
app = FastAPI(
    title="ML Dashboard API",
    description="API for ML Dashboard with user authentication and prediction storage.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["authentication"])
app.include_router(user_router, prefix="/users", tags=["users"])
app.include_router(prediction_router, prefix="/predictions", tags=["predictions"])
app.include_router(simple_router, tags=["simple"])
app.include_router(report_router, prefix="/reports", tags=["reports"])

# Initialize service
service = AnomalyService("modelo/artifacts_anomalia")

@app.on_event("startup")
async def startup_event():
    """Create database tables on startup"""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.get("/health", response_model=HealthResponse)
def health():
    ok, details = service.healthcheck()
    return HealthResponse(status="ok" if ok else "error", details=details)

@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    current_user: UserResponse = Depends(get_current_active_user),
    db: Session = Depends(get_sync_db)
):
    """
    Unified prediction endpoint that handles all types of predictions and stores them in the database.
    Expects either:
    - records: list[dict[str, float]] with feature-value pairs; OR
    - gold_parquet_path: a Parquet path (server-side) to batch-predict last N rows.
    """
    try:
        prediction_service = PredictionService(db)
        stored_predictions = []
        start_time = time.time()
        
        if req.records and len(req.records) > 0:
            # Single or batch prediction from records
            df, preds = service.predict_from_records(req.records)
            
            for i, record in enumerate(req.records):
                # Create prediction record
                prediction_data = PredictionCreate(
                    input_type="single" if len(req.records) == 1 else "batch",
                    input_source="records",
                    input_features=record,
                    input_metadata={
                        "batch_size": len(req.records),
                        "record_index": i,
                        "timestamp": datetime.now().isoformat()
                    },
                    model_version=service.meta.get("model_version", "unknown")
                )
                
                stored_prediction = prediction_service.create_prediction(
                    prediction_data, current_user.id
                )
                
                # Update with results
                pred_result = preds[i] if i < len(preds) else None
                if pred_result:
                    processing_time = int((time.time() - start_time) * 1000)
                    prediction_service.update_prediction(
                        stored_prediction.id,
                        PredictionUpdate(
                            prediction_results={
                                "score": pred_result["score"],
                                "label": pred_result["label"],
                                "horizon_shift": pred_result["horizon_shift"],
                                "predicted_future_state": pred_result["predicted_future_state"],
                                "prediction_time": pred_result["prediction_time"]
                            },
                            anomaly_detected=pred_result["label"] == "anomaly",
                            confidence_score=pred_result["score"],
                            anomaly_score=pred_result["score"],
                            status="completed",
                            processing_time_ms=processing_time
                        )
                    )
                    stored_predictions.append(stored_prediction)
                    
        elif req.gold_parquet_path:
            # Batch prediction from parquet file
            df, preds = service.predict_from_parquet(req.gold_parquet_path, req.limit_rows)
            
            for i, pred in enumerate(preds):
                # Create prediction record
                prediction_data = PredictionCreate(
                    input_type="parquet",
                    input_source="gold_data",
                    input_features={"batch_prediction": True, "index": i},
                    input_metadata={
                        "file_path": req.gold_parquet_path,
                        "limit_rows": req.limit_rows,
                        "batch_size": len(preds),
                        "record_index": i,
                        "timestamp": datetime.now().isoformat()
                    },
                    model_version=service.meta.get("model_version", "unknown")
                )
                
                stored_prediction = prediction_service.create_prediction(
                    prediction_data, current_user.id
                )
                
                # Update with results
                processing_time = int((time.time() - start_time) * 1000)
                prediction_service.update_prediction(
                    stored_prediction.id,
                    PredictionUpdate(
                        prediction_results={
                            "score": pred["score"],
                            "label": pred["label"],
                            "horizon_shift": pred["horizon_shift"],
                            "predicted_future_state": pred["predicted_future_state"],
                            "prediction_time": pred["prediction_time"]
                        },
                        anomaly_detected=pred["label"] == "anomaly",
                        confidence_score=pred["score"],
                        anomaly_score=pred["score"],
                        status="completed",
                        processing_time_ms=processing_time
                    )
                )
                stored_predictions.append(stored_prediction)
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
        feature_columns=service.feature_columns,
        model_version=service.meta.get("model_version", "unknown")
    )

@app.post("/predict/gold-data", response_model=PredictResponse)
def predict_from_gold_data(
    req: PredictRequest,
    current_user: UserResponse = Depends(get_current_active_user),
    db: Session = Depends(get_sync_db)
):
    """
    Endpoint específico para predecir desde datos de capa_gold.
    Acepta gold_parquet_path en el request body.
    Automatically stores predictions in the database for audit and analysis.
    """
    try:
        if not req.gold_parquet_path:
            raise HTTPException(status_code=400, detail="gold_parquet_path is required for this endpoint")
        
        df, preds = service.predict_from_parquet(req.gold_parquet_path, req.limit_rows)
        
        # Store predictions in the database
        prediction_service = PredictionService(db)
        stored_predictions = []
        
        for i, pred in enumerate(preds):
            # Create prediction record
            prediction_data = PredictionCreate(
                input_type="parquet",
                input_source="gold_data",
                input_features={"batch_prediction": True, "index": i},
                input_metadata={
                    "file_path": req.gold_parquet_path,
                    "limit_rows": req.limit_rows,
                    "batch_size": len(preds),
                    "record_index": i,
                    "timestamp": datetime.now().isoformat()
                },
                model_version=service.meta.get("model_version", "unknown")
            )
            
            stored_prediction = prediction_service.create_prediction(
                prediction_data, current_user.id
            )
            
            # Update with results
            prediction_service.update_prediction(
                stored_prediction.id,
                PredictionUpdate(
                    prediction_results={
                        "score": pred["score"],
                        "label": pred["label"],
                        "horizon_shift": pred["horizon_shift"],
                        "predicted_future_state": pred["predicted_future_state"],
                        "prediction_time": pred["prediction_time"]
                    },
                    anomaly_detected=pred["label"] == "anomaly",
                    confidence_score=pred["score"],
                    anomaly_score=pred["score"],
                    status="completed"
                )
            )
            stored_predictions.append(stored_prediction)

        return PredictResponse(
            model_version=service.meta.get("model_version", "unknown"),
            feature_order=service.feature_columns,
            results=preds
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))