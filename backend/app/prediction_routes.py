# app/prediction_routes.py
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import json

from .database import get_sync_db
from .models import Prediction, PredictionStatus
from .schemas import (
    PredictionCreate, 
    PredictionUpdate, 
    PredictionResponse,
    PredictResponse,
    PredictItem,
    FeaturesResponse
)
from .prediction_service import PredictionService
from .service import AnomalyService
from .auth import get_current_active_user
from .models import User

router = APIRouter()

# Initialize service
service = AnomalyService("modelo/artifacts_anomalia")

@router.get("/maintenance/results", response_model=PredictResponse)
async def get_maintenance_results(
    db: Session = Depends(get_sync_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Obtiene las últimas predicciones del transformador desde la base de datos.
    Si no hay predicciones recientes, ejecuta una nueva predicción y la guarda.
    """
    try:
        prediction_service = PredictionService(db)
        
        # Buscar la predicción más reciente del transformador
        latest_prediction = prediction_service.get_latest_prediction_by_equipment("TR01")
        
        # Si hay una predicción reciente (menos de 1 hora), devolverla
        if latest_prediction and latest_prediction.created_at:
            time_diff = datetime.utcnow() - latest_prediction.created_at
            if time_diff < timedelta(hours=1) and latest_prediction.status == "completed":
                # Deserializar los resultados
                results_data = json.loads(latest_prediction.prediction_results)
                return PredictResponse(**results_data)
        
        # Si no hay predicción reciente, ejecutar nueva predicción
        return await execute_and_save_new_prediction(db, prediction_service, current_user.id)
        
    except Exception as e:
        print(f"❌ Error en get_maintenance_results: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error obteniendo resultados: {str(e)}")

async def execute_and_save_new_prediction(
    db: Session, 
    prediction_service: PredictionService, 
    user_id: int
) -> PredictResponse:
    """
    Ejecuta una nueva predicción y la guarda en la base de datos
    """
    try:
        from pathlib import Path
        import pandas as pd
        
        # Leer los datos más recientes del ETL desde capa_gold
        data_path = Path(__file__).parent.parent.parent / "data" / "capa_gold" / "features_transformador"
        
        # Buscar archivos de datos más recientes
        csv_files = list(data_path.glob("transformer_features_complete_*.csv"))
        parquet_files = list(data_path.glob("transformer_features_complete_*.parquet"))
        
        if csv_files:
            latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
            print(f"📊 Leyendo datos de: {latest_file.name}")
            df = pd.read_csv(latest_file)
        elif parquet_files:
            latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
            print(f"📊 Leyendo datos de: {latest_file.name}")
            df = pd.read_parquet(latest_file)
        else:
            raise HTTPException(status_code=404, detail="No se encontraron datos del ETL en capa_gold")
        
        print(f"📈 Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
        
        # Ejecutar predicción
        df, preds = service.predict_from_parquet(str(latest_file), limit_rows=24)
        
        # Crear entrada en la base de datos
        prediction_data = PredictionCreate(
            equipment_id="TR01",
            equipment_name="Transformador Principal",
            input_type="parquet",
            input_source="gold_data",
            input_features=df.tail(24).to_dict('records'),
            input_metadata={
                "file_source": latest_file.name,
                "total_rows": len(df),
                "data_columns": list(df.columns)
            },
            model_version=service.meta.get("model_version", "ae_lstm_v1")
        )
        
        # Guardar predicción en base de datos
        db_prediction = prediction_service.create_prediction(prediction_data, user_id)
        
        # Actualizar con resultados
        prediction_results = {
            "model_version": service.meta.get("model_version", "ae_lstm_v1"),
            "feature_order": service.feature_columns,
            "results": preds,
            "data_info": {
                "total_rows": len(df),
                "file_source": latest_file.name,
                "threshold_used": service.meta.get("operate_thr", 0.6),
                "lookback_window": service.meta.get("lookback", 24),
                "horizon_shift": service.meta.get("horizon_shift", 360),
                "ae_only_mode": service.meta.get("operate_with_ae_only", True),
                "prediction_type": "future_state_prediction"
            }
        }
        
        # Actualizar la predicción con resultados
        update_data = PredictionUpdate(
            prediction_results=json.dumps(prediction_results),
            anomaly_detected=preds[0]["label"] == "ANOMALY" if preds else False,
            confidence_score=preds[0]["score"] if preds else 0.0,
            status="completed",
            execution_time_ms=int(1000),  # Aproximado
            model_metadata=json.dumps(service.meta)
        )
        
        prediction_service.update_prediction(db_prediction.id, update_data)
        
        print(f"✅ Predicción guardada en BD con ID: {db_prediction.id}")
        return PredictResponse(**prediction_results)
        
    except Exception as e:
        print(f"❌ Error ejecutando nueva predicción: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error ejecutando predicción: {str(e)}")

@router.get("/predictions/latest", response_model=PredictionResponse)
async def get_latest_prediction(
    equipment_id: str = "TR01",
    db: Session = Depends(get_sync_db),
    current_user: User = Depends(get_current_active_user)
):
    """Obtiene la predicción más reciente para un equipo específico"""
    prediction_service = PredictionService(db)
    prediction = prediction_service.get_latest_prediction_by_equipment(equipment_id)
    
    if not prediction:
        raise HTTPException(status_code=404, detail="No se encontraron predicciones")
    
    return prediction

@router.get("/predictions/history", response_model=List[PredictionResponse])
async def get_prediction_history(
    equipment_id: str = "TR01",
    limit: int = 10,
    db: Session = Depends(get_sync_db),
    current_user: User = Depends(get_current_active_user)
):
    """Obtiene el historial de predicciones para un equipo"""
    prediction_service = PredictionService(db)
    predictions = prediction_service.get_predictions_by_equipment(equipment_id, limit)
    return predictions

@router.post("/predictions/execute", response_model=PredictResponse)
async def execute_new_prediction(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_sync_db),
    current_user: User = Depends(get_current_active_user)
):
    """Fuerza la ejecución de una nueva predicción"""
    prediction_service = PredictionService(db)
    return await execute_and_save_new_prediction(db, prediction_service, current_user.id)

@router.get("/features", response_model=FeaturesResponse)
async def get_features():
    """Obtiene información sobre las características del modelo"""
    return FeaturesResponse(
        feature_order=service.feature_columns,
        model_version=service.meta.get("model_version", "ae_lstm_v1")
    )
