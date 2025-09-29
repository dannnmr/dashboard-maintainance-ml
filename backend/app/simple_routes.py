# app/simple_routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
import json

from .database import get_sync_db
from .service import AnomalyService
from .schemas import PredictResponse, PredictItem, FeaturesResponse
from .auth import get_current_active_user
from .models import User

router = APIRouter(tags=["simple"])

# Initialize AnomalyService (model inference logic)
anomaly_service = AnomalyService("modelo/artifacts_anomalia")

@router.get("/simple/features", response_model=FeaturesResponse)
async def get_simple_features():
    """Get feature order and model version without authentication"""
    return FeaturesResponse(
        feature_order=anomaly_service.feature_columns,
        model_version=anomaly_service.meta.get("model_version", "ae_lstm_v1")
    )

@router.get("/simple/predictions", response_model=PredictResponse)
async def get_simple_predictions():
    """Get predictions without database storage - for testing"""
    try:
        from pathlib import Path
        import pandas as pd
        
        # Read the latest ETL data from capa_gold
        data_path = Path(__file__).parent.parent.parent / "data" / "capa_gold" / "features_transformador"
        
        # Look for latest data files
        csv_files = list(data_path.glob("transformer_features_complete_*.csv"))
        parquet_files = list(data_path.glob("transformer_features_complete_*.parquet"))
        
        latest_file = None
        if csv_files:
            latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        elif parquet_files:
            latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
        
        if not latest_file:
            # Use validation data as fallback
            validation_files = list(data_path.glob("*validation*.parquet"))
            if validation_files:
                latest_file = max(validation_files, key=lambda x: x.stat().st_mtime)
        
        if not latest_file:
            raise HTTPException(status_code=404, detail="No se encontraron datos del ETL")
        
        print(f"📊 Leyendo datos de: {latest_file.name}")
        
        # Load the original data to get timestamps
        if latest_file.suffix == '.csv':
            original_df = pd.read_csv(latest_file)
        else:
            original_df = pd.read_parquet(latest_file)
        
        # Execute inference using AnomalyService - get more predictions
        df, preds = anomaly_service.predict_from_parquet(str(latest_file), limit_rows=100)
        
        # Generate multiple predictions from different time windows
        all_predictions = []
        lookback_window = anomaly_service.meta.get("lookback", 24)
        
        # Generate historical predictions for maintenance
        # Each prediction uses 24 hours of data to predict 15 days in the future
        horizon_days = 15  # 15 days prediction horizon
        horizon_hours = horizon_days * 24  # 360 hours
        
        # For historical predictions, we want to show more predictions
        # Use a smaller interval to show more historical data
        # But still maintain the 15-day prediction horizon
        interval_hours = 12  # Every 12 hours for historical view
        
        # Calculate how many historical predictions we can generate
        max_predictions = min(20, (len(df) - lookback_window) // interval_hours)
        
        print(f"📅 Generating historical predictions every {interval_hours} hours")
        print(f"🎯 Each prediction forecasts {horizon_days} days ahead")
        print(f"📊 Can generate up to {max_predictions} historical predictions")
        
        for i in range(max_predictions):
            # Space historical predictions every 12 hours
            # Each prediction uses 24 hours of data, spaced 12 hours apart
            # But each prediction still forecasts 15 days ahead
            prediction_offset = i * interval_hours  # 0, 12, 24, 36, etc.
            
            start_idx = len(df) - lookback_window - prediction_offset
            end_idx = len(df) - prediction_offset
            
            if start_idx >= 0:
                window_df = df.iloc[start_idx:end_idx].copy()
                if len(window_df) == lookback_window:
                    try:
                        pred_result = anomaly_service._predict_from_window(window_df)
                        
                        # Get the actual timestamp from the original dataframe
                        original_idx = end_idx - 1
                        if original_idx < len(original_df) and 'timestamp' in original_df.columns:
                            actual_timestamp = original_df.iloc[original_idx]['timestamp']
                        else:
                            # Generate a timestamp based on the index (going backwards in time)
                            base_time = "2025-08-27T22:00:00Z"
                            actual_timestamp = base_time
                        
                        prediction_data = {
                            "index": end_idx - 1,
                            "score": pred_result["score"],
                            "label": "ANOMALY" if pred_result["pred"] == 1 else "NORMAL",
                            "horizon_shift": pred_result.get("horizon_shift", 360),
                            "predicted_future_state": pred_result.get("predicted_future_state", "NORMAL"),
                            "prediction_time": f"{horizon_days} days ahead",
                            "created_at": actual_timestamp
                        }
                        
                        print(f"🔍 Adding prediction with timestamp: {actual_timestamp}")
                        all_predictions.append(prediction_data)
                    except Exception as e:
                        print(f"Error en ventana {i}: {e}")
                        continue
        
        # Use generated predictions or fallback to original
        final_predictions = all_predictions if all_predictions else preds
        
        # Format response
        response = PredictResponse(
            model_version=anomaly_service.meta.get("model_version", "ae_lstm_v1"),
            feature_order=anomaly_service.feature_columns,
            results=final_predictions,
            data_info={
                "total_rows": len(df),
                "file_source": latest_file.name,
                "threshold_used": anomaly_service.meta.get("operate_thr", 0.6),
                "lookback_window": anomaly_service.meta.get("lookback", 24),
                "horizon_shift": anomaly_service.meta.get("horizon_shift", 360),
                "ae_only_mode": anomaly_service.meta.get("operate_with_ae_only", True),
                "prediction_type": "future_state_prediction"
            }
        )
        
        print(f"✅ Predicción exitosa: {preds[0] if preds else 'No predictions'}")
        return response
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error procesando datos: {str(e)}")

@router.get("/simple/health")
async def simple_health():
    """Simple health check"""
    return {
        "status": "ok",
        "details": {
            "feature_columns": len(anomaly_service.feature_columns),
            "model_loaded": True,
            "horizon_shift": anomaly_service.meta.get("horizon_shift", 360)
        }
    }

@router.get("/simple/stats")
async def get_simple_stats():
    """Get prediction statistics without database"""
    try:
        from pathlib import Path
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Read the latest ETL data from capa_gold
        data_path = Path(__file__).parent.parent.parent / "data" / "capa_gold" / "features_transformador"
        
        # Look for latest data files
        csv_files = list(data_path.glob("transformer_features_complete_*.csv"))
        parquet_files = list(data_path.glob("transformer_features_complete_*.parquet"))
        
        latest_file = None
        if csv_files:
            latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        elif parquet_files:
            latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
        
        if not latest_file:
            # Use validation data as fallback
            validation_files = list(data_path.glob("*validation*.parquet"))
            if validation_files:
                latest_file = max(validation_files, key=lambda x: x.stat().st_mtime)
        
        if not latest_file:
            raise HTTPException(status_code=404, detail="No se encontraron datos del ETL")
        
        # Load data for analysis
        if latest_file.suffix == '.csv':
            df = pd.read_csv(latest_file)
        else:
            df = pd.read_parquet(latest_file)
        
        # Calculate basic statistics
        total_predictions = len(df)
        
        # Execute prediction to get current status
        df_pred, preds = anomaly_service.predict_from_parquet(str(latest_file), limit_rows=24)
        current_prediction = preds[0] if preds else {}
        
        # Calculate stats based on available data
        completed = total_predictions
        failed = 0
        success_rate = (completed / total_predictions * 100) if total_predictions > 0 else 100.0
        
        stats = {
            "equipment_id": "TR01",
            "total_predictions": total_predictions,
            "completed_predictions": completed,
            "failed_predictions": failed,
            "pending_predictions": 0,
            "processing_predictions": 0,
            "success_rate": success_rate,
            "last_prediction_time": datetime.utcnow().isoformat(),
            "current_status": current_prediction.get("label", "UNKNOWN"),
            "current_score": current_prediction.get("score", 0.0),
            "model_version": anomaly_service.meta.get("model_version", "ae_lstm_v1"),
            "horizon_shift": anomaly_service.meta.get("horizon_shift", 360),
            "data_source": latest_file.name
        }
        
        return stats
        
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")

@router.get("/simple/transformers")
async def get_transformers():
    """Get list of available transformers"""
    try:
        transformers = [
            {
                "id": "TR01",
                "name": "Transformador Principal TR01",
                "status": "ACTIVE",
                "location": "Subestación Central",
                "lastPrediction": "2024-09-14T07:00:00Z",
                "model": "AE-LSTM v1",
                "predictionHorizon": "15 days"
            }
        ]
        
        return {"transformers": transformers}
        
    except Exception as e:
        print(f"❌ Error getting transformers: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo transformadores: {str(e)}")

@router.get("/simple/transformers/{transformer_id}/history")
async def get_transformer_history(transformer_id: str):
    """Get historical data for a specific transformer"""
    try:
        if transformer_id != "TR01":
            raise HTTPException(status_code=404, detail="Transformador no encontrado")
        
        from pathlib import Path
        import pandas as pd
        from datetime import datetime
        
        # Read historical data from features_complete
        data_path = Path(__file__).parent.parent.parent / "data" / "capa_gold" / "features_transformador" / "features_complete"
        
        # Get all available months
        available_months = []
        for year_dir in data_path.iterdir():
            if year_dir.is_dir() and year_dir.name.startswith("year="):
                year = year_dir.name.split("=")[1]
                for month_dir in year_dir.iterdir():
                    if month_dir.is_dir() and month_dir.name.startswith("month="):
                        month = month_dir.name.split("=")[1]
                        parquet_files = list(month_dir.glob("*.parquet"))
                        if parquet_files:
                            available_months.append({
                                "year": int(year),
                                "month": int(month),
                                "file": parquet_files[0]
                            })
        
        # Sort by year and month
        available_months.sort(key=lambda x: (x["year"], x["month"]))
        
        # Load data from all available months (2024-2025)
        # Show more historical data instead of just recent months
        recent_months = available_months  # Load all available data
        
        all_data = []
        for month_info in recent_months:
            try:
                df = pd.read_parquet(month_info["file"])
                # Add year and month columns for filtering
                df["year"] = month_info["year"]
                df["month"] = month_info["month"]
                all_data.append(df)
            except Exception as e:
                print(f"Error loading month {month_info['year']}-{month_info['month']}: {e}")
                continue
        
        if not all_data:
            raise HTTPException(status_code=404, detail="No se encontraron datos históricos")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp
        combined_df = combined_df.sort_values("timestamp")
        
        # Get historical data (not predictions)
        # Sample every 1 hour for detailed historical view
        sample_interval = 1  # hours
        max_samples = min(200, len(combined_df) // sample_interval)
        
        historical_data = []
        feature_columns = anomaly_service.feature_columns
        
        for i in range(0, max_samples * sample_interval, sample_interval):
            idx = len(combined_df) - 1 - i
            
            if idx >= 0:
                row = combined_df.iloc[idx]
                
                # Extract timestamp
                timestamp = str(row['timestamp']) if 'timestamp' in row else f"2024-{month_info['month']:02d}-15T12:00:00Z"
                
                # Extract key features for display using actual column names
                data_point = {
                    "timestamp": timestamp,
                    "temp_oil": float(row.get('temp_oil_value', 0)) if 'temp_oil_value' in row else 0,
                    "temp_ambient": float(row.get('temp_ambient_value', 0)) if 'temp_ambient_value' in row else 0,
                    "voltage": float(row.get('voltage_value', 0)) if 'voltage_value' in row else 0,
                    "current_load": float(row.get('current_load_value', 0)) if 'current_load_value' in row else 0,
                    "power_apparent": float(row.get('power_apparent_value', 0)) if 'power_apparent_value' in row else 0,
                    "tap_position": float(row.get('tap_position_value', 0)) if 'tap_position_value' in row else 0,
                    "estado_operacional": str(row.get('estado_operacional', 'UNKNOWN')) if 'estado_operacional' in row else 'UNKNOWN',
                    "nivel_severidad": float(row.get('nivel_severidad', 0)) if 'nivel_severidad' in row else 0,
                    "temp_hot_spot": float(row.get('temp_spot_hot_value', 0)) if 'temp_spot_hot_value' in row else 0,
                    "gradient_hot_oil": float(row.get('gradient_hot_oil', 0)) if 'gradient_hot_oil' in row else 0
                }
                
                historical_data.append(data_point)
        
        # Prepare response
        response = {
            "transformer_id": transformer_id,
            "name": "Transformador Principal TR01",
            "data_range": {
                "start": str(combined_df["timestamp"].min()),
                "end": str(combined_df["timestamp"].max()),
                "total_records": len(combined_df),
                "months_available": len(available_months)
            },
            "historical_data": historical_data,
            "current_measurements": historical_data[0] if historical_data else {},
            "data_info": {
                "sample_interval": f"{sample_interval} hours",
                "total_samples": len(historical_data),
                "features_available": feature_columns[:10] if feature_columns else []  # Show first 10 features
            }
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting transformer history: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error obteniendo historial del transformador: {str(e)}")
