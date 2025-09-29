# app/schemas.py
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from typing import List

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
    # Optional: attach original features if needed by the frontend
    # features: Dict[str, float] | None = None

class PredictResponse(BaseModel):
    model_version: str
    feature_order: List[str]
    results: List[PredictItem]
