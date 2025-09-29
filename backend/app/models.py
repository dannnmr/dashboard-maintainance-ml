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
    status = Column(SQLEnum(PredictionStatus, name="prediction_status"), default="pending")
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