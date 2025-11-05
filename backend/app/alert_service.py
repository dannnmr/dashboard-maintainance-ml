# app/alert_service.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from .models import Alert, Prediction, AlertSeverity, AlertStatus
from .schemas import AlertCreate, AlertUpdate, AlertSummary

class AlertService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_alert(self, alert_data: AlertCreate) -> Alert:
        """Create a new alert."""
        try:
            db_alert = Alert(
                equipment_id=alert_data.equipment_id,
                equipment_name=alert_data.equipment_name,
                title=alert_data.title,
                message=alert_data.message,
                severity=alert_data.severity,
                alert_type=alert_data.alert_type,
                source=alert_data.source,
                prediction_id=alert_data.prediction_id,
                anomaly_score=alert_data.anomaly_score,
                confidence_score=alert_data.confidence_score,
                status=AlertStatus.ACTIVE
            )
            
            self.db.add(db_alert)
            await self.db.commit()
            await self.db.refresh(db_alert)
            return db_alert
        except IntegrityError:
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Error creating alert"
            )

    async def get_alert_by_id(self, alert_id: int) -> Optional[Alert]:
        """Get alert by ID."""
        result = await self.db.execute(select(Alert).where(Alert.id == alert_id))
        return result.scalar_one_or_none()

    async def get_alerts(
        self, 
        skip: int = 0, 
        limit: int = 100,
        severity: Optional[AlertSeverity] = None,
        status: Optional[AlertStatus] = None,
        equipment_id: Optional[str] = None
    ) -> List[Alert]:
        """Get alerts with filters."""
        query = select(Alert)
        
        # Apply filters
        conditions = []
        if severity:
            conditions.append(Alert.severity == severity)
        if status:
            conditions.append(Alert.status == status)
        if equipment_id:
            conditions.append(Alert.equipment_id == equipment_id)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.order_by(Alert.created_at.desc()).offset(skip).limit(limit)
        
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_active_alerts(self, equipment_id: Optional[str] = None) -> List[Alert]:
        """Get active alerts."""
        try:
            query = select(Alert).where(Alert.status == AlertStatus.ACTIVE)
            
            if equipment_id:
                query = query.where(Alert.equipment_id == equipment_id)
            
            query = query.order_by(Alert.created_at.desc())
            
            result = await self.db.execute(query)
            alerts = result.scalars().all()
            
            # Convert to list to ensure proper serialization
            return list(alerts)
        except Exception as e:
            print(f"Error in get_active_alerts: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return []

    async def get_alert_summary(self, equipment_id: Optional[str] = None) -> AlertSummary:
        """Get alert summary statistics."""
        base_query = select(Alert)
        if equipment_id:
            base_query = base_query.where(Alert.equipment_id == equipment_id)
        
        # Total alerts
        total_result = await self.db.execute(select(func.count(Alert.id)).select_from(base_query.subquery()))
        total_alerts = total_result.scalar() or 0
        
        # Alerts by severity
        critical_result = await self.db.execute(
            select(func.count(Alert.id)).where(
                and_(Alert.severity == AlertSeverity.CRITICAL, 
                     base_query.where().where(Alert.equipment_id == equipment_id) if equipment_id else True)
            )
        )
        critical_alerts = critical_result.scalar() or 0
        
        warning_result = await self.db.execute(
            select(func.count(Alert.id)).where(
                and_(Alert.severity == AlertSeverity.WARNING,
                     base_query.where().where(Alert.equipment_id == equipment_id) if equipment_id else True)
            )
        )
        warning_alerts = warning_result.scalar() or 0
        
        info_result = await self.db.execute(
            select(func.count(Alert.id)).where(
                and_(Alert.severity == AlertSeverity.INFO,
                     base_query.where().where(Alert.equipment_id == equipment_id) if equipment_id else True)
            )
        )
        info_alerts = info_result.scalar() or 0
        
        # Alerts by status
        active_result = await self.db.execute(
            select(func.count(Alert.id)).where(
                and_(Alert.status == AlertStatus.ACTIVE,
                     base_query.where().where(Alert.equipment_id == equipment_id) if equipment_id else True)
            )
        )
        active_alerts = active_result.scalar() or 0
        
        acknowledged_result = await self.db.execute(
            select(func.count(Alert.id)).where(
                and_(Alert.status == AlertStatus.ACKNOWLEDGED,
                     base_query.where().where(Alert.equipment_id == equipment_id) if equipment_id else True)
            )
        )
        acknowledged_alerts = acknowledged_result.scalar() or 0
        
        resolved_result = await self.db.execute(
            select(func.count(Alert.id)).where(
                and_(Alert.status == AlertStatus.RESOLVED,
                     base_query.where().where(Alert.equipment_id == equipment_id) if equipment_id else True)
            )
        )
        resolved_alerts = resolved_result.scalar() or 0
        
        return AlertSummary(
            total_alerts=total_alerts,
            critical_alerts=critical_alerts,
            warning_alerts=warning_alerts,
            info_alerts=info_alerts,
            active_alerts=active_alerts,
            acknowledged_alerts=acknowledged_alerts,
            resolved_alerts=resolved_alerts
        )

    async def update_alert(self, alert_id: int, alert_data: AlertUpdate, user_id: int) -> Alert:
        """Update alert status."""
        alert = await self.get_alert_by_id(alert_id)
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        # Update fields
        if alert_data.status is not None:
            alert.status = alert_data.status
            
            # Set timestamps based on status
            if alert_data.status == AlertStatus.ACKNOWLEDGED:
                alert.acknowledged_at = datetime.utcnow()
                alert.acknowledged_by_user_id = user_id
            elif alert_data.status == AlertStatus.RESOLVED:
                alert.resolved_at = datetime.utcnow()
                alert.resolved_by_user_id = user_id
        
        if alert_data.message is not None:
            alert.message = alert_data.message
        
        if alert_data.comments is not None:
            alert.comments = alert_data.comments
        
        if alert_data.validation_status is not None:
            alert.validation_status = alert_data.validation_status
        
        alert.updated_at = datetime.utcnow()
        
        try:
            await self.db.commit()
            await self.db.refresh(alert)
            return alert
        except IntegrityError:
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Error updating alert"
            )

    async def delete_alert(self, alert_id: int) -> bool:
        """Delete alert (soft delete by marking as resolved)."""
        alert = await self.get_alert_by_id(alert_id)
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        await self.db.commit()
        return True

    async def generate_alert_from_prediction(self, prediction: Prediction) -> Optional[Alert]:
        """Generate alert from prediction if anomaly is detected."""
        if not prediction.anomaly_detected:
            return None
        
        # Determine severity based on anomaly score and confidence
        severity = AlertSeverity.INFO
        if prediction.anomaly_score and prediction.confidence_score:
            if prediction.anomaly_score > 0.8 and prediction.confidence_score > 0.7:
                severity = AlertSeverity.CRITICAL
            elif prediction.anomaly_score > 0.6 and prediction.confidence_score > 0.5:
                severity = AlertSeverity.WARNING
            else:
                severity = AlertSeverity.INFO
        
        # Create alert title and message
        title = f"Anomalía Detectada en {prediction.equipment_name}"
        message = f"Se ha detectado una anomalía con score {prediction.anomaly_score:.3f} y confianza {prediction.confidence_score:.3f}. "
        
        if severity == AlertSeverity.CRITICAL:
            message += "Se requiere atención inmediata."
        elif severity == AlertSeverity.WARNING:
            message += "Se recomienda monitoreo cercano."
        else:
            message += "Se recomienda revisión rutinaria."
        
        # Check if similar alert already exists (avoid duplicates)
        existing_alert = await self.db.execute(
            select(Alert).where(
                and_(
                    Alert.prediction_id == prediction.id,
                    Alert.status == AlertStatus.ACTIVE,
                    Alert.created_at > datetime.utcnow() - timedelta(hours=1)
                )
            )
        )
        
        if existing_alert.scalar_one_or_none():
            return None  # Don't create duplicate alert
        
        # Create new alert
        alert_data = AlertCreate(
            equipment_id=prediction.equipment_id,
            equipment_name=prediction.equipment_name,
            title=title,
            message=message,
            severity=severity,
            prediction_id=prediction.id,
            anomaly_score=prediction.anomaly_score,
            confidence_score=prediction.confidence_score
        )
        
        return await self.create_alert(alert_data)

    async def cleanup_old_alerts(self, days: int = 30) -> int:
        """Clean up old resolved alerts."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = await self.db.execute(
            select(Alert).where(
                and_(
                    Alert.status == AlertStatus.RESOLVED,
                    Alert.resolved_at < cutoff_date
                )
            )
        )
        
        old_alerts = result.scalars().all()
        count = len(old_alerts)
        
        for alert in old_alerts:
            await self.db.delete(alert)
        
        await self.db.commit()
        return count
