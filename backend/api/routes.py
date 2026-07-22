from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

from backend.database.connection import get_db
from backend.database import crud
from backend.models.classifier import get_model_manager
from backend.xai.gradcam import GradCAM
from backend.llm.report_generator import get_report_generator
from backend.utils.pdf_generator import generate_scan_pdf_report
from backend.config import UPLOADS_DIR, HEATMAPS_DIR, REPORTS_DIR, BASE_DIR
from backend.api.schemas import PredictionResponse, AnalyticsResponse, ScanHistoryResponse, HealthResponse

router = APIRouter(prefix="/api/v1", tags=["Medical Intelligence"])

@router.post("/predict", response_model=PredictionResponse)
async def predict_medical_image(
    file: UploadFile = File(...),
    patient_id: str = Form("P-MED-1001"),
    patient_name: str = Form("Anonymous Patient"),
    age: int = Form(45),
    gender: str = Form("Unspecified"),
    scan_type: str = Form("Chest X-Ray PA View"),
    db: Session = Depends(get_db)
):
    """
    Main Diagnostic Endpoint:
    1. Uploads and saves Chest X-Ray image.
    2. Runs PyTorch model inference for pathology prediction.
    3. Generates Explainable AI (Grad-CAM) heatmap & alpha overlay.
    4. Generates AI Clinical Radiology Report (Gemini API / Logic engine).
    5. Saves full record to SQLite Database.
    6. Generates printable PDF report.
    """
    # 1. Save uploaded image
    file_ext = os.path.splitext(file.filename)[1] or ".png"
    unique_filename = f"scan_{uuid.uuid4().hex[:10]}{file_ext}"
    saved_image_path = UPLOADS_DIR / unique_filename

    with open(saved_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. DL Model Inference
    try:
        model_mgr = get_model_manager()
        pred_result = model_mgr.predict(str(saved_image_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # 3. Grad-CAM XAI Generation
    try:
        gradcam = GradCAM(model_mgr.model, model_mgr.model.get_target_layer())
        heatmap_2d, spatial_metrics = gradcam.generate_heatmap(
            pred_result["tensor"],
            pred_result["class_index"]
        )
        
        prefix = f"gcam_{uuid.uuid4().hex[:8]}"
        vis_paths = gradcam.save_visualization(pred_result["pil_image"], heatmap_2d, prefix)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(e)}")

    # 4. LLM Medical Report Generation
    patient_info = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "age": age,
        "gender": gender,
        "scan_type": scan_type
    }
    
    try:
        llm_gen = get_report_generator()
        report_data = llm_gen.generate_report(
            pred_result["predicted_class"],
            pred_result["confidence"],
            pred_result["probabilities"],
            spatial_metrics,
            patient_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

    # 5. Persist to Database
    rel_orig_path = f"/storage/uploads/{unique_filename}"
    rel_heatmap_path = f"/storage/heatmaps/{vis_paths['heatmap_filename']}"
    rel_overlay_path = f"/storage/heatmaps/{vis_paths['overlay_filename']}"

    db_record = crud.create_scan_record(
        db=db,
        patient_id=patient_id,
        patient_name=patient_name,
        age=age,
        gender=gender,
        scan_type=scan_type,
        original_image_path=rel_orig_path,
        heatmap_image_path=rel_heatmap_path,
        overlay_image_path=rel_overlay_path,
        predicted_class=pred_result["predicted_class"],
        confidence=pred_result["confidence"],
        probabilities=pred_result["probabilities"],
        spatial_metrics=spatial_metrics,
        report_text=report_data["full_report_text"],
        risk_level=report_data["risk_level"],
        report_source=report_data["source"]
    )

    scan_dict = db_record.to_dict()

    # 6. Generate PDF Clinical Report
    full_orig_path = str(saved_image_path)
    full_overlay_path = vis_paths["overlay_path"]
    scan_dict_for_pdf = dict(scan_dict)
    scan_dict_for_pdf["original_image_path"] = full_orig_path
    scan_dict_for_pdf["overlay_image_path"] = full_overlay_path

    try:
        pdf_path = generate_scan_pdf_report(scan_dict_for_pdf)
        pdf_filename = os.path.basename(pdf_path)
        pdf_url = f"/api/v1/report/pdf/download/{pdf_filename}"
    except Exception as e:
        pdf_url = ""
        print(f"[Routes] PDF generation warning: {e}")

    return {
        "scan_id": scan_dict["scan_id"],
        "patient_id": scan_dict["patient_id"],
        "patient_name": scan_dict["patient_name"],
        "age": scan_dict["age"],
        "gender": scan_dict["gender"],
        "scan_type": scan_dict["scan_type"],
        "predicted_class": scan_dict["predicted_class"],
        "confidence": scan_dict["confidence"],
        "probabilities": scan_dict["probabilities"],
        "spatial_metrics": scan_dict["spatial_metrics"],
        "original_image_url": scan_dict["original_image_path"],
        "heatmap_image_url": scan_dict["heatmap_image_path"],
        "overlay_image_url": scan_dict["overlay_image_path"],
        "report_text": scan_dict["report_text"],
        "risk_level": scan_dict["risk_level"],
        "report_source": scan_dict["report_source"],
        "pdf_report_url": pdf_url,
        "created_at": scan_dict["created_at"]
    }

@router.get("/history", response_model=ScanHistoryResponse)
def get_diagnostic_history(
    skip: int = 0,
    limit: int = 50,
    search: Optional[str] = None,
    filter_class: Optional[str] = None,
    db: Session = Depends(get_db)
):
    scans = crud.get_all_scans(db, skip=skip, limit=limit, search=search, filter_class=filter_class)
    formatted_scans = []
    
    for s in scans:
        d = s.to_dict()
        pdf_file = f"{d['scan_id']}_clinical_report.pdf"
        pdf_url = f"/api/v1/report/pdf/download/{pdf_file}" if os.path.exists(REPORTS_DIR / pdf_file) else ""
        
        formatted_scans.append({
            "scan_id": d["scan_id"],
            "patient_id": d["patient_id"],
            "patient_name": d["patient_name"],
            "age": d["age"],
            "gender": d["gender"],
            "scan_type": d["scan_type"],
            "predicted_class": d["predicted_class"],
            "confidence": d["confidence"],
            "probabilities": d["probabilities"],
            "spatial_metrics": d["spatial_metrics"],
            "original_image_url": d["original_image_path"],
            "heatmap_image_url": d["heatmap_image_path"],
            "overlay_image_url": d["overlay_image_path"],
            "report_text": d["report_text"],
            "risk_level": d["risk_level"],
            "report_source": d["report_source"],
            "pdf_report_url": pdf_url,
            "created_at": d["created_at"]
        })

    return {
        "scans": formatted_scans,
        "total": len(formatted_scans)
    }

@router.get("/history/{scan_id}", response_model=PredictionResponse)
def get_scan_details(scan_id: str, db: Session = Depends(get_db)):
    scan = crud.get_scan_by_id(db, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan record not found")
        
    d = scan.to_dict()
    pdf_file = f"{d['scan_id']}_clinical_report.pdf"
    pdf_url = f"/api/v1/report/pdf/download/{pdf_file}" if os.path.exists(REPORTS_DIR / pdf_file) else ""

    return {
        "scan_id": d["scan_id"],
        "patient_id": d["patient_id"],
        "patient_name": d["patient_name"],
        "age": d["age"],
        "gender": d["gender"],
        "scan_type": d["scan_type"],
        "predicted_class": d["predicted_class"],
        "confidence": d["confidence"],
        "probabilities": d["probabilities"],
        "spatial_metrics": d["spatial_metrics"],
        "original_image_url": d["original_image_path"],
        "heatmap_image_url": d["heatmap_image_path"],
        "overlay_image_url": d["overlay_image_path"],
        "report_text": d["report_text"],
        "risk_level": d["risk_level"],
        "report_source": d["report_source"],
        "pdf_report_url": pdf_url,
        "created_at": d["created_at"]
    }

@router.get("/analytics", response_model=AnalyticsResponse)
def get_analytics(db: Session = Depends(get_db)):
    return crud.get_analytics_summary(db)

@router.get("/report/pdf/download/{pdf_name}")
def download_pdf_report(pdf_name: str):
    pdf_path = REPORTS_DIR / pdf_name
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF report file not found")
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=pdf_name
    )

@router.get("/project-report")
def get_project_report():
    report_path = REPORTS_DIR / "Advanced_AI_Medical_Intelligence_Platform_Report.pdf"
    if not report_path.exists():
        from backend.utils.pdf_generator import generate_project_technical_report
        generate_project_technical_report(str(report_path))
    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename="Advanced_AI_Medical_Intelligence_Platform_Report.pdf"
    )

@router.get("/health", response_model=HealthResponse)
def health_check():
    import datetime
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model_loaded": True,
        "database_status": "connected",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
