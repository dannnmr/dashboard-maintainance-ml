# app/alert_routes.py
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from .database import get_db
from .schemas import (
    AlertResponse, 
    AlertCreate, 
    AlertUpdate, 
    AlertSummary,
    AlertSeverity,
    AlertStatus
)
from .auth import get_current_active_user, require_admin
from .models import User, Prediction
from .alert_service import AlertService

router = APIRouter(tags=["alerts"])

@router.get("/", response_model=List[AlertResponse])
async def get_alerts(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    severity: Optional[AlertSeverity] = Query(None),
    status: Optional[AlertStatus] = Query(None),
    equipment_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get alerts with optional filters."""
    alert_service = AlertService(db)
    alerts = await alert_service.get_alerts(
        skip=skip,
        limit=limit,
        severity=severity,
        status=status,
        equipment_id=equipment_id
    )
    return alerts

@router.get("/summary", response_model=AlertSummary)
async def get_alert_summary(
    equipment_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get alert summary statistics."""
    alert_service = AlertService(db)
    summary = await alert_service.get_alert_summary(equipment_id=equipment_id)
    return summary

@router.get("/active")
async def get_active_alerts(
    equipment_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get active alerts."""
    try:
        from sqlalchemy import select
        from .models import Alert, AlertStatus
        
        # Direct query without service (fetch and filter in Python to avoid enum casting issues)
        query = select(Alert)
        if equipment_id:
            query = query.where(Alert.equipment_id == equipment_id)
        query = query.order_by(Alert.created_at.desc())
        
        result = await db.execute(query)
        all_alerts = result.scalars().all()
        alerts = []
        for a in all_alerts:
            try:
                status_val = a.status.value if hasattr(a.status, "value") else str(a.status)
            except Exception:
                status_val = str(a.status)
            if (status_val or "").lower() == "active":
                alerts.append(a)
        
        # Convert to AlertResponse format (normalize enums to lowercase strings)
        alert_list = []
        for alert in alerts:
            alert_dict = {
                "id": alert.id,
                "equipment_id": alert.equipment_id,
                "equipment_name": alert.equipment_name,
                "title": alert.title,
                "message": alert.message,
                "severity": (alert.severity.value.lower() if hasattr(alert.severity, "value") else str(alert.severity).lower()),
                "status": (alert.status.value.lower() if hasattr(alert.status, "value") else str(alert.status).lower()),
                "alert_type": alert.alert_type,
                "source": alert.source,
                "comments": alert.comments,
                "validation_status": alert.validation_status,
                "prediction_id": alert.prediction_id,
                "anomaly_score": alert.anomaly_score,
                "confidence_score": alert.confidence_score,
                "acknowledged_by_user_id": alert.acknowledged_by_user_id,
                "resolved_by_user_id": alert.resolved_by_user_id,
                "created_at": alert.created_at,
                "acknowledged_at": alert.acknowledged_at,
                "resolved_at": alert.resolved_at,
                "updated_at": alert.updated_at
            }
            alert_list.append(alert_dict)
        
        return alert_list
    except Exception as e:
        print(f"Error in get_active_alerts endpoint: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get alert by ID."""
    alert_service = AlertService(db)
    alert = await alert_service.get_alert_by_id(alert_id)
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    return alert

@router.post("/", response_model=AlertResponse)
async def create_alert(
    alert_data: AlertCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Create a new alert (admin only)."""
    alert_service = AlertService(db)
    alert = await alert_service.create_alert(alert_data)
    return alert

@router.put("/{alert_id}", response_model=AlertResponse)
async def update_alert(
    alert_id: int,
    alert_data: AlertUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update alert status (acknowledge/resolve)."""
    alert_service = AlertService(db)
    alert = await alert_service.update_alert(alert_id, alert_data, current_user.id)
    return alert

@router.delete("/{alert_id}")
async def delete_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Delete alert (admin only - soft delete)."""
    alert_service = AlertService(db)
    success = await alert_service.delete_alert(alert_id)
    return {"message": "Alert resolved successfully"}

@router.post("/acknowledge/{alert_id}", response_model=AlertResponse)
async def acknowledge_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Acknowledge an alert."""
    alert_service = AlertService(db)
    alert_data = AlertUpdate(status=AlertStatus.ACKNOWLEDGED)
    alert = await alert_service.update_alert(alert_id, alert_data, current_user.id)
    return alert

@router.post("/resolve/{alert_id}", response_model=AlertResponse)
async def resolve_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Resolve an alert."""
    alert_service = AlertService(db)
    alert_data = AlertUpdate(status=AlertStatus.RESOLVED)
    alert = await alert_service.update_alert(alert_id, alert_data, current_user.id)
    return alert

@router.put("/{alert_id}/comments", response_model=AlertResponse)
async def update_alert_comments(
    alert_id: int,
    comments_data: AlertUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update alert comments and validation status (admin and tecnico only)."""
    if current_user.role == "viewer":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Viewers cannot update alert comments")
    
    alert_service = AlertService(db)
    alert = await alert_service.update_alert(alert_id, comments_data, current_user.id)
    return alert

@router.post("/cleanup")
async def cleanup_old_alerts(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Clean up old resolved alerts (admin only)."""
    alert_service = AlertService(db)
    count = await alert_service.cleanup_old_alerts(days)
    return {"message": f"Cleaned up {count} old alerts"}

@router.post("/generate-from-predictions")
async def generate_alerts_from_predictions(
    equipment_id: Optional[str] = Query("TR01"),
    hours_back: int = Query(24, ge=1, le=168),  # Max 1 week
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Generate alerts from recent predictions with anomalies (admin only)."""
    from .prediction_service import PredictionService
    from sqlalchemy.orm import Session
    from datetime import datetime, timedelta
    
    # Get sync session for prediction service
    sync_db = Session(db.bind)
    prediction_service = PredictionService(sync_db)
    
    # Get recent predictions with anomalies
    cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
    recent_predictions = sync_db.query(Prediction).filter(
        Prediction.equipment_id == equipment_id,
        Prediction.anomaly_detected == True,
        Prediction.status == "completed",
        Prediction.created_at >= cutoff_time
    ).all()
    
    alert_service = AlertService(db)
    generated_count = 0
    
    for prediction in recent_predictions:
        try:
            alert = await alert_service.generate_alert_from_prediction(prediction)
            if alert:
                generated_count += 1
        except Exception as e:
            print(f"Error generating alert for prediction {prediction.id}: {e}")
    
    sync_db.close()
    return {"message": f"Generated {generated_count} alerts from {len(recent_predictions)} predictions"}
