from sqlalchemy.orm import Session
from sqlalchemy import func
import json
import uuid
from typing import List, Dict, Any, Optional

from backend.database.models import ScanRecord

def create_scan_record(
    db: Session,
    patient_id: str,
    patient_name: str,
    age: int,
    gender: str,
    scan_type: str,
    original_image_path: str,
    heatmap_image_path: str,
    overlay_image_path: str,
    predicted_class: str,
    confidence: float,
    probabilities: Dict[str, float],
    spatial_metrics: Dict[str, Any],
    report_text: str,
    risk_level: str,
    report_source: str
) -> ScanRecord:
    """
    Creates and persists a new scan diagnostic record in database.
    """
    scan_id = f"SCAN-{uuid.uuid4().hex[:8].upper()}"
    
    db_record = ScanRecord(
        scan_id=scan_id,
        patient_id=patient_id or "P-MED-1001",
        patient_name=patient_name or "Anonymous Patient",
        age=age or 45,
        gender=gender or "Unspecified",
        scan_type=scan_type or "Chest X-Ray PA View",
        original_image_path=original_image_path,
        heatmap_image_path=heatmap_image_path,
        overlay_image_path=overlay_image_path,
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities_json=json.dumps(probabilities),
        spatial_metrics_json=json.dumps(spatial_metrics),
        activation_coverage_pct=spatial_metrics.get("activation_coverage_pct", 0.0),
        severity_score=spatial_metrics.get("severity_score", 0.0),
        report_text=report_text,
        risk_level=risk_level,
        report_source=report_source
    )
    
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record

def get_scan_by_id(db: Session, scan_id: str) -> Optional[ScanRecord]:
    return db.query(ScanRecord).filter(ScanRecord.scan_id == scan_id).first()

def get_scan_by_pk(db: Session, record_id: int) -> Optional[ScanRecord]:
    return db.query(ScanRecord).filter(ScanRecord.id == record_id).first()

def get_all_scans(
    db: Session,
    skip: int = 0,
    limit: int = 50,
    search: Optional[str] = None,
    filter_class: Optional[str] = None
) -> List[ScanRecord]:
    query = db.query(ScanRecord)
    
    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            (ScanRecord.patient_id.like(search_pattern)) |
            (ScanRecord.patient_name.like(search_pattern)) |
            (ScanRecord.scan_id.like(search_pattern))
        )
        
    if filter_class and filter_class != "All":
        query = query.filter(ScanRecord.predicted_class == filter_class)
        
    return query.order_by(ScanRecord.created_at.desc()).offset(skip).limit(limit).all()

def get_analytics_summary(db: Session) -> Dict[str, Any]:
    """
    Computes aggregate stats and analytics for the platform.
    """
    total_scans = db.query(func.count(ScanRecord.id)).scalar() or 0
    
    # Class breakdown
    class_counts = db.query(
        ScanRecord.predicted_class, func.count(ScanRecord.id)
    ).group_by(ScanRecord.predicted_class).all()
    
    class_distribution = {c_name: count for c_name, count in class_counts}
    
    # Average confidence
    avg_conf = db.query(func.avg(ScanRecord.confidence)).scalar() or 0.0
    
    # High risk count
    high_risk_count = db.query(func.count(ScanRecord.id)).filter(ScanRecord.risk_level == "High Risk").scalar() or 0
    
    return {
        "total_scans": total_scans,
        "class_distribution": class_distribution,
        "average_confidence": round(float(avg_conf), 4),
        "high_risk_cases": high_risk_count
    }
