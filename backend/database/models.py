from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import declarative_base
import datetime

Base = declarative_base()

class ScanRecord(Base):
    """
    SQLAlchemy Model for storing Medical Diagnostics, Patient Data, XAI metrics, and Reports.
    """
    __tablename__ = "scan_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    scan_id = Column(String, unique=True, index=True, nullable=False)
    patient_id = Column(String, index=True, default="P-MED-1001")
    patient_name = Column(String, default="Anonymous Patient")
    age = Column(Integer, default=45)
    gender = Column(String, default="Unspecified")
    scan_type = Column(String, default="Chest X-Ray PA View")
    
    # Image file paths
    original_image_path = Column(String, nullable=False)
    heatmap_image_path = Column(String, nullable=True)
    overlay_image_path = Column(String, nullable=True)
    
    # Prediction metrics
    predicted_class = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    probabilities_json = Column(Text, nullable=False)
    
    # XAI Grad-CAM metrics
    spatial_metrics_json = Column(Text, nullable=True)
    activation_coverage_pct = Column(Float, default=0.0)
    severity_score = Column(Float, default=0.0)
    
    # Clinical Report
    report_text = Column(Text, nullable=True)
    risk_level = Column(String, default="Low Risk")
    report_source = Column(String, default="AI Clinical Logic Engine")
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        import json
        return {
            "id": self.id,
            "scan_id": self.scan_id,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "age": self.age,
            "gender": self.gender,
            "scan_type": self.scan_type,
            "original_image_path": self.original_image_path,
            "heatmap_image_path": self.heatmap_image_path,
            "overlay_image_path": self.overlay_image_path,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "probabilities": json.loads(self.probabilities_json) if self.probabilities_json else {},
            "spatial_metrics": json.loads(self.spatial_metrics_json) if self.spatial_metrics_json else {},
            "activation_coverage_pct": self.activation_coverage_pct,
            "severity_score": self.severity_score,
            "report_text": self.report_text,
            "risk_level": self.risk_level,
            "report_source": self.report_source,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else ""
        }
