# app/schemas.py
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, EmailStr
from typing import List
from datetime import datetime
from .models import UserRole, PredictionStatus, AlertSeverity, AlertStatus

class FeaturesResponse(BaseModel):
    feature_order: List[str]
    model_version: str

class HealthResponse(BaseModel):
    status: str
    details: Dict[str, Any]

class PredictRequest(BaseModel):
    # Option A: JSON records posted by the frontend
    records: Optional[List[Dict[str, float]]] = Field(default=None)
    # Option B: Server-side batch from a Gold parquet (useful for dashboards)
    gold_parquet_path: Optional[str] = Field(default=None)
    limit_rows: int = Field(default=200)

class PredictItem(BaseModel):
    index: int
    score: float
    label: str
    horizon_shift: int = 12
    predicted_future_state: str = "NORMAL"
    prediction_time: str = "12 hours ahead"
    created_at: Optional[str] = None
    # Optional: attach original features if needed by the frontend
    # features: Dict[str, float] | None = None

class PredictResponse(BaseModel):
    model_version: str
    feature_order: List[str]
    results: List[PredictItem]
    data_info: Optional[Dict[str, Any]] = None

# Authentication schemas
class UserBase(BaseModel):
    email: EmailStr
    username: str
    role: UserRole = UserRole.VIEWER

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, description="New password must be at least 8 characters")

# Prediction schemas
class PredictionBase(BaseModel):
    equipment_id: str = "TR01"  # Default to the only transformer
    equipment_name: Optional[str] = "Transformador Principal"
    input_type: str  # 'single', 'batch', 'parquet'
    input_source: str  # 'records', 'gold_data', 'manual'
    input_features: List[Dict[str, Any]]
    input_metadata: Optional[Dict[str, Any]] = None
    model_version: Optional[str] = None

class PredictionCreate(PredictionBase):
    pass

class PredictionUpdate(BaseModel):
    prediction_results: Optional[str] = None  # JSON string
    anomaly_detected: Optional[bool] = None
    confidence_score: Optional[float] = None
    anomaly_score: Optional[float] = None
    status: Optional[PredictionStatus] = None
    execution_time_ms: Optional[int] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    model_metadata: Optional[str] = None  # JSON string

    class Config:
        from_attributes = True

class PredictionResponse(PredictionBase):
    id: int
    prediction_results: Optional[Dict[str, Any]] = None
    anomaly_detected: Optional[bool] = None
    confidence_score: Optional[float] = None
    anomaly_score: Optional[float] = None
    status: PredictionStatus
    execution_time_ms: Optional[int] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    executed_by_user_id: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class PredictionRequest(BaseModel):
    equipment_id: str = "TR01"  # Default to the only transformer
    input_type: str = "single"
    input_source: str = "records"
    input_features: Dict[str, Any]
    input_metadata: Optional[Dict[str, Any]] = None
    model_version: Optional[str] = None

# Alert schemas
class AlertBase(BaseModel):
    equipment_id: str = "TR01"
    equipment_name: Optional[str] = "Transformador Principal"
    title: str
    message: str
    severity: AlertSeverity
    alert_type: str = "anomaly_detection"
    source: str = "ml_model"
    comments: Optional[str] = None
    validation_status: Optional[str] = None

class AlertCreate(AlertBase):
    prediction_id: Optional[int] = None
    anomaly_score: Optional[float] = None
    confidence_score: Optional[float] = None

class AlertUpdate(BaseModel):
    status: Optional[AlertStatus] = None
    message: Optional[str] = None
    comments: Optional[str] = None
    validation_status: Optional[str] = None

class AlertResponse(AlertBase):
    id: int
    status: AlertStatus
    prediction_id: Optional[int] = None
    anomaly_score: Optional[float] = None
    confidence_score: Optional[float] = None
    acknowledged_by_user_id: Optional[int] = None
    resolved_by_user_id: Optional[int] = None
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class AlertSummary(BaseModel):
    total_alerts: int
    critical_alerts: int
    warning_alerts: int
    info_alerts: int
    active_alerts: int
    acknowledged_alerts: int
    resolved_alerts: int
