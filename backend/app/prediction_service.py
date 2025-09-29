# app/prediction_service.py
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from datetime import datetime, timedelta
import json
import time

from .models import Prediction, PredictionStatus
from .schemas import PredictionCreate, PredictionUpdate

class PredictionService:
    def __init__(self, db: Session):
        self.db = db

    def create_prediction(self, prediction_data: PredictionCreate, executed_by_user_id: int = None) -> Prediction:
        """Create a new prediction record"""
        db_prediction = Prediction(
            equipment_id=prediction_data.equipment_id,
            equipment_name=prediction_data.equipment_name,
            input_type=prediction_data.input_type,
            input_source=prediction_data.input_source,
            input_features=json.dumps(prediction_data.input_features),
            input_metadata=json.dumps(prediction_data.input_metadata) if prediction_data.input_metadata else None,
            model_version=prediction_data.model_version,
            executed_by_user_id=executed_by_user_id,
            status="pending"
        )
        self.db.add(db_prediction)
        self.db.commit()
        self.db.refresh(db_prediction)
        return db_prediction

    def update_prediction(
        self, 
        prediction_id: int, 
        prediction_update: PredictionUpdate
    ) -> Optional[Prediction]:
        """Update a prediction with results"""
        db_prediction = self.db.query(Prediction).filter(
            Prediction.id == prediction_id
        ).first()
        
        if not db_prediction:
            return None

        # Update fields
        if prediction_update.prediction_results is not None:
            db_prediction.prediction_results = json.dumps(prediction_update.prediction_results)
        if prediction_update.anomaly_detected is not None:
            db_prediction.anomaly_detected = prediction_update.anomaly_detected
        if prediction_update.confidence_score is not None:
            db_prediction.confidence_score = prediction_update.confidence_score
        if prediction_update.anomaly_score is not None:
            db_prediction.anomaly_score = prediction_update.anomaly_score
        if prediction_update.status is not None:
            db_prediction.status = PredictionStatus(prediction_update.status)
        if prediction_update.processing_time_ms is not None:
            db_prediction.processing_time_ms = prediction_update.processing_time_ms
        if prediction_update.error_message is not None:
            db_prediction.error_message = prediction_update.error_message

        db_prediction.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(db_prediction)
        return db_prediction

    def get_prediction(self, prediction_id: int) -> Optional[Prediction]:
        """Get a specific prediction by ID"""
        return self.db.query(Prediction).filter(
            Prediction.id == prediction_id
        ).first()

    def get_predictions_by_equipment(
        self, 
        equipment_id: str = "TR01", 
        skip: int = 0, 
        limit: int = 100,
        status: Optional[PredictionStatus] = None
    ) -> List[Prediction]:
        """Get predictions for a specific equipment (defaults to TR01)"""
        query = self.db.query(Prediction).filter(Prediction.equipment_id == equipment_id)
        
        if status:
            query = query.filter(Prediction.status == status)
        
        return query.order_by(desc(Prediction.created_at)).offset(skip).limit(limit).all()
    
    def get_latest_prediction(self, equipment_id: str = "TR01") -> Optional[Prediction]:
        """Get the most recent prediction for an equipment"""
        return self.db.query(Prediction).filter(
            Prediction.equipment_id == equipment_id
        ).order_by(desc(Prediction.created_at)).first()

    def get_all_predictions(
        self, 
        skip: int = 0, 
        limit: int = 100,
        status: Optional[PredictionStatus] = None
    ) -> List[Prediction]:
        """Get all predictions (admin only)"""
        query = self.db.query(Prediction)
        
        if status:
            query = query.filter(Prediction.status == status)
        
        return query.order_by(desc(Prediction.created_at)).offset(skip).limit(limit).all()

    def get_prediction_stats(self, equipment_id: str = "TR01") -> dict:
        """Get prediction statistics for an equipment"""
        query = self.db.query(Prediction).filter(Prediction.equipment_id == equipment_id)
        
        total = query.count()
        completed = query.filter(Prediction.status == "completed").count()
        failed = query.filter(Prediction.status == "failed").count()
        pending = query.filter(Prediction.status == "pending").count()
        processing = query.filter(Prediction.status == "processing").count()
        
        return {
            "equipment_id": equipment_id,
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "processing": processing,
            "success_rate": (completed / total * 100) if total > 0 else 0
        }

    def delete_prediction(self, prediction_id: int) -> bool:
        """Delete a prediction (admin only)"""
        db_prediction = self.db.query(Prediction).filter(
            Prediction.id == prediction_id
        ).first()
        
        if not db_prediction:
            return False
        
        self.db.delete(db_prediction)
        self.db.commit()
        return True

    def cleanup_old_predictions(self, days_old: int = 30) -> int:
        """Clean up predictions older than specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        deleted_count = self.db.query(Prediction).filter(
            Prediction.created_at < cutoff_date
        ).delete()
        self.db.commit()
        return deleted_count

    def get_latest_prediction_by_equipment(self, equipment_id: str) -> Optional[Prediction]:
        """Get the latest prediction for a specific equipment"""
        return self.db.query(Prediction).filter(
            Prediction.equipment_id == equipment_id
        ).order_by(desc(Prediction.created_at)).first()

    def get_predictions_by_equipment(self, equipment_id: str, limit: int = 10) -> List[Prediction]:
        """Get predictions for a specific equipment"""
        return self.db.query(Prediction).filter(
            Prediction.equipment_id == equipment_id
        ).order_by(desc(Prediction.created_at)).limit(limit).all()