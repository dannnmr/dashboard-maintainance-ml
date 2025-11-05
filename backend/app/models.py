from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    VIEWER = "viewer"
    TECNICO = "tecnico"

class PredictionStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class AlertSeverity(str, enum.Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

class AlertStatus(str, enum.Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(SQLEnum(UserRole, name="userrole"), default="VIEWER")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    
    # Equipment identification (TR01 is the only transformer)
    equipment_id = Column(String(50), nullable=False, default="TR01", index=True)
    equipment_name = Column(String(100), nullable=True, default="Transformador Principal")
    
    # Input Data
    input_type = Column(String(20), nullable=False)  # 'single', 'batch', 'parquet'
    input_source = Column(String(20), nullable=False)  # 'records', 'gold_data', 'manual'
    input_features = Column(Text, nullable=False)  # JSON string of input features
    input_metadata = Column(Text, nullable=True)  # JSON string of additional metadata
    
    # Prediction Results
    prediction_results = Column(Text, nullable=True)  # JSON string of complete results
    anomaly_detected = Column(Boolean, nullable=True)  # True/False
    confidence_score = Column(Float, nullable=True)  # 0.0-1.0
    anomaly_score = Column(Float, nullable=True)  # Anomaly score
    
    # System Metadata
    status = Column(String(20), default="pending")
    model_version = Column(String(50), nullable=True)
    execution_time_ms = Column(Integer, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    model_metadata = Column(Text, nullable=True)  # JSON string of model metadata
    
    # Audit trail (who executed the prediction)
    executed_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    executed_by_user = relationship("User", foreign_keys=[executed_by_user_id])

# Add the relationship to User model (executed predictions)
User.executed_predictions = relationship("Prediction", foreign_keys="Prediction.executed_by_user_id")

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    
    # Equipment identification
    equipment_id = Column(String(50), nullable=False, default="TR01", index=True)
    equipment_name = Column(String(100), nullable=True, default="Transformador Principal")
    
    # Alert details
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    severity = Column(SQLEnum(AlertSeverity, name="alertseverity"), nullable=False)
    status = Column(SQLEnum(AlertStatus, name="alertstatus"), default="active")
    
    # Prediction reference
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)
    anomaly_score = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Alert metadata
    alert_type = Column(String(50), nullable=False, default="anomaly_detection")
    source = Column(String(50), nullable=False, default="ml_model")
    
    # Comments and observations
    comments = Column(Text, nullable=True)  # User comments/observations
    validation_status = Column(String(20), nullable=True)  # 'validated', 'false_positive', 'investigating'
    
    # User who acknowledged/resolved
    acknowledged_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    resolved_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships (commented out to avoid serialization issues)
    # prediction = relationship("Prediction", foreign_keys=[prediction_id])
    # acknowledged_by_user = relationship("User", foreign_keys=[acknowledged_by_user_id])
    # resolved_by_user = relationship("User", foreign_keys=[resolved_by_user_id])