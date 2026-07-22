from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import datetime

class PredictionResponse(BaseModel):
    scan_id: str
    patient_id: str
    patient_name: str
    age: int
    gender: str
    scan_type: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    spatial_metrics: Dict[str, Any]
    original_image_url: str
    heatmap_image_url: str
    overlay_image_url: str
    report_text: str
    risk_level: str
    report_source: str
    pdf_report_url: str
    created_at: str

class AnalyticsResponse(BaseModel):
    total_scans: int
    class_distribution: Dict[str, int]
    average_confidence: float
    high_risk_cases: int

class ScanHistoryResponse(BaseModel):
    scans: List[PredictionResponse]
    total: int

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    database_status: str
    timestamp: str
