import os
from pathlib import Path
from typing import Dict, Any
import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable, PageBreak

from backend.config import REPORTS_DIR

def generate_scan_pdf_report(scan_data: Dict[str, Any], output_pdf_path: str = None) -> str:
    """
    Generates a clinical diagnostic PDF report for an individual patient scan using ReportLab.
    """
    if output_pdf_path is None:
        scan_id = scan_data.get("scan_id", "SCAN-000")
        output_pdf_path = str(REPORTS_DIR / f"{scan_id}_clinical_report.pdf")

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        alignment=0,
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'SubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=15
    )

    heading2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155')
    )

    disclaimer_style = ParagraphStyle(
        'DisclaimerCustom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#64748b')
    )

    elements = []

    # Header
    elements.append(Paragraph("ADVANCED AI MEDICAL INTELLIGENCE PLATFORM", title_style))
    elements.append(Paragraph(f"Clinical Radiology Report & Diagnostic Interpretation | Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563eb'), spaceAfter=15))

    # Patient Metadata Table
    patient_table_data = [
        [
            Paragraph("<b>Patient ID:</b>", body_style), Paragraph(str(scan_data.get("patient_id", "N/A")), body_style),
            Paragraph("<b>Patient Name:</b>", body_style), Paragraph(str(scan_data.get("patient_name", "N/A")), body_style)
        ],
        [
            Paragraph("<b>Age / Gender:</b>", body_style), Paragraph(f"{scan_data.get('age', 'N/A')} yrs / {scan_data.get('gender', 'N/A')}", body_style),
            Paragraph("<b>Scan Modality:</b>", body_style), Paragraph(str(scan_data.get("scan_type", "Chest X-Ray")), body_style)
        ],
        [
            Paragraph("<b>Scan ID:</b>", body_style), Paragraph(str(scan_data.get("scan_id", "N/A")), body_style),
            Paragraph("<b>Risk Stratification:</b>", body_style), Paragraph(f"<font color='#dc2626'><b>{scan_data.get('risk_level', 'N/A')}</b></font>", body_style)
        ]
    ]

    t_patient = Table(patient_table_data, colWidths=[100, 160, 110, 170])
    t_patient.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(t_patient)
    elements.append(Spacer(1, 15))

    # AI Diagnostic Result Badge Box
    pred_class = scan_data.get("predicted_class", "Normal")
    conf = scan_data.get("confidence", 0.0) * 100
    bg_color = colors.HexColor('#dcfce7') if pred_class == "Normal" else colors.HexColor('#fee2e2')
    text_color = '#15803d' if pred_class == "Normal" else '#b91c1c'

    diag_text = f"<b>PRIMARY AI DIAGNOSIS: {pred_class.upper()}</b> &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; <b>Model Confidence: {conf:.2f}%</b>"
    p_diag = Paragraph(f"<font size=12 color='{text_color}'>{diag_text}</font>", body_style)

    t_diag = Table([[p_diag]], colWidths=[540])
    t_diag.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_color),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor(text_color)),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(t_diag)
    elements.append(Spacer(1, 15))

    # Visual Evidence: Original Image vs Grad-CAM Overlay
    elements.append(Paragraph("Visual Evidence & Explainable AI (Grad-CAM)", heading2_style))
    
    orig_path = scan_data.get("original_image_path")
    overlay_path = scan_data.get("overlay_image_path")
    
    img_cells = []
    if orig_path and os.path.exists(orig_path):
        img_cells.append(RLImage(orig_path, width=240, height=240))
    else:
        img_cells.append(Paragraph("Original X-Ray Unavailable", body_style))
        
    if overlay_path and os.path.exists(overlay_path):
        img_cells.append(RLImage(overlay_path, width=240, height=240))
    else:
        img_cells.append(Paragraph("Grad-CAM Overlay Unavailable", body_style))

    t_images = Table([[img_cells[0], img_cells[1]]], colWidths=[270, 270])
    t_images.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(t_images)
    elements.append(Paragraph("<font size=8 color='#64748b'>Left: Original Chest X-Ray | Right: Grad-CAM Feature Attribution Heatmap (Red/Warm indicates focal diagnostic triggers)</font>", disclaimer_style))
    elements.append(Spacer(1, 15))

    # Detailed Clinical Report
    elements.append(Paragraph("Detailed Radiological Findings & Clinical Impression", heading2_style))
    report_text = scan_data.get("report_text", "No detailed report text available.")
    report_text_formatted = report_text.replace('\n', '<br/>')
    elements.append(Paragraph(f"<font size=9>{report_text_formatted}</font>", body_style))
    elements.append(Spacer(1, 15))

    # Disclaimer Footer
    elements.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#cbd5e1'), spaceAfter=8))
    elements.append(Paragraph("<b>MEDICAL DISCLAIMER:</b> This automated medical intelligence report is generated by a Deep Learning and Explainable AI pipeline designed for clinical decision support. This document does not constitute a final medical diagnosis and must be reviewed by a qualified healthcare professional.", disclaimer_style))

    doc.build(elements)
    print(f"[PDFGenerator] Saved clinical report PDF to {output_pdf_path}")
    return output_pdf_path


def generate_project_technical_report(output_pdf_path: str = None) -> str:
    """
    Generates the formal assignment submission PDF report (Advanced_AI_Medical_Intelligence_Platform_Report.pdf).
    Fulfills submission requirement for SN Matrix Software Pvt. Ltd.
    """
    if output_pdf_path is None:
        output_pdf_path = str(REPORTS_DIR / "Advanced_AI_Medical_Intelligence_Platform_Report.pdf")

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'ProjectTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#0f172a'),
        alignment=1,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        'ProjectSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2563eb'),
        alignment=1,
        spaceAfter=20
    )

    h1_style = ParagraphStyle(
        'H1Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=14,
        spaceAfter=8
    )

    h2_style = ParagraphStyle(
        'H2Custom',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0284c7'),
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'ProjectBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    code_style = ParagraphStyle(
        'ProjectCode',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#0f172a')
    )

    elements = []

    # Title Banner
    elements.append(Paragraph("ADVANCED AI MEDICAL INTELLIGENCE PLATFORM", title_style))
    elements.append(Paragraph("Technical Evaluation Project Assignment Report | SN Matrix Software Pvt. Ltd.", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2563eb'), spaceAfter=15))

    # Candidate Meta Info
    meta_table_data = [
        [Paragraph("<b>Candidate Name:</b>", body_style), Paragraph("Vamshi Cherukuri", body_style), Paragraph("<b>Position:</b>", body_style), Paragraph("AI/ML Engineer", body_style)],
        [Paragraph("<b>Submission Date:</b>", body_style), Paragraph(datetime.datetime.now().strftime('%Y-%m-%d'), body_style), Paragraph("<b>Target Organization:</b>", body_style), Paragraph("SN Matrix Software Pvt. Ltd.", body_style)]
    ]
    t_meta = Table(meta_table_data, colWidths=[110, 150, 110, 160])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(t_meta)
    elements.append(Spacer(1, 15))

    # 1. Executive Summary
    elements.append(Paragraph("1. Executive Summary", h1_style))
    elements.append(Paragraph(
        "This project presents an end-to-end clinical AI intelligence system built to assist medical practitioners in automated diagnostic screening of chest radiographs (X-Rays). "
        "The platform unifies PyTorch Deep Learning pathology classification, Grad-CAM visual Explainable AI (XAI), Large Language Model (LLM) radiology report generation, FastAPI REST web services, SQLAlchemy database tracking, and a glassmorphic Web Application UI.",
        body_style
    ))

    # 2. System Architecture & Tech Stack
    elements.append(Paragraph("2. System Architecture & Technology Stack", h1_style))
    arch_matrix = [
        [Paragraph("<b>Layer</b>", body_style), Paragraph("<b>Technology Stack</b>", body_style), Paragraph("<b>Key Responsibilities</b>", body_style)],
        [Paragraph("Deep Learning", body_style), Paragraph("PyTorch / Torchvision (DenseNet121)", body_style), Paragraph("Chest X-Ray pathology multi-class prediction (Normal, Pneumonia, COVID-19)", body_style)],
        [Paragraph("Explainable AI", body_style), Paragraph("Grad-CAM / OpenCV / NumPy", body_style), Paragraph("Convolutional gradient heatmap activation map generation & alpha overlay", body_style)],
        [Paragraph("LLM Reporter", body_style), Paragraph("Google Gemini 1.5 Flash / Clinical Logic", body_style), Paragraph("Structured medical report generation with diagnostic findings & recommendations", body_style)],
        [Paragraph("REST API", body_style), Paragraph("FastAPI / Uvicorn / Pydantic", body_style), Paragraph("High-performance async API endpoints with OpenAPI interactive documentation", body_style)],
        [Paragraph("Database", body_style), Paragraph("SQLAlchemy / SQLite", body_style), Paragraph("Persistent audit logging of patient metadata, predictions, heatmaps, and reports", body_style)],
        [Paragraph("Web Application", body_style), Paragraph("Vanilla Modern HTML5 / CSS3 / ES6 JS", body_style), Paragraph("Glassmorphic UI, side-by-side interactive opacity slider, and live history dashboard", body_style)]
    ]
    t_arch = Table(arch_matrix, colWidths=[90, 180, 260])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(t_arch)
    elements.append(Spacer(1, 10))

    # Page Break for clean multi-page document layout
    elements.append(PageBreak())

    # 3. Deep Learning & Explainable AI (Grad-CAM)
    elements.append(Paragraph("3. Deep Learning Model & Explainable AI Engine", h1_style))
    elements.append(Paragraph(
        "<b>Model Design:</b> The system utilizes a DenseNet121 backbone fine-tuned for thoracic medical imaging. "
        "DenseNet's dense connectivity promotes feature reuse, reducing vanishing gradients and making it exceptionally suited for subtle radiological pattern extraction.<br/>"
        "<b>Grad-CAM Formulation:</b> Feature attributions are computed at the final dense block layer (<code>features.denseblock4.denselayer16</code>). "
        "Gradients of the target class score \\( y^c \\) with respect to feature activation maps \\( A^k \\) are global-average pooled to obtain neuron weights \\( \\alpha_k^c \\):<br/>"
        "<code>&nbsp;&nbsp;&nbsp;&nbsp;&alpha;_k^c = (1/Z) &Sigma;_i &Sigma;_j (&part;y^c / &part;A_{i,j}^k)</code><br/>"
        "The resulting coarse heatmap is generated by combining weighted activations passed through a ReLU nonlinearity.",
        body_style
    ))

    # 4. LLM Medical Report Generation
    elements.append(Paragraph("4. LLM Medical Report Generation Pipeline", h1_style))
    elements.append(Paragraph(
        "The platform incorporates Google Gemini 1.5 Flash API for automated radiology report generation. "
        "The LLM receives structured inputs combining patient demographics, model confidence metrics, class probability distributions, and Grad-CAM spatial coverage indicators. "
        "If an API key is unconfigured, the system automatically falls back to an offline Medical Logic Generation Engine ensuring 100% operational uptime.",
        body_style
    ))

    # 5. REST API Specifications
    elements.append(Paragraph("5. REST API Specifications & OpenAPI Endpoints", h1_style))
    api_specs = [
        [Paragraph("<b>Endpoint</b>", body_style), Paragraph("<b>Method</b>", body_style), Paragraph("<b>Description</b>", body_style)],
        [Paragraph("/api/v1/predict", code_style), Paragraph("POST", body_style), Paragraph("Upload medical image -> Returns prediction, Grad-CAM heatmap, DB record, and LLM report", body_style)],
        [Paragraph("/api/v1/gradcam", code_style), Paragraph("POST", body_style), Paragraph("Generate custom Grad-CAM heatmap visualization with adjustable alpha overlay", body_style)],
        [Paragraph("/api/v1/report/pdf/{scan_id}", code_style), Paragraph("GET", body_style), Paragraph("Generate and download printable PDF clinical radiology report", body_style)],
        [Paragraph("/api/v1/history", code_style), Paragraph("GET", body_style), Paragraph("Retrieve searchable/filterable database history of patient diagnostic records", body_style)],
        [Paragraph("/api/v1/analytics", code_style), Paragraph("GET", body_style), Paragraph("Fetch aggregate platform statistics, class breakdowns, and model confidence metrics", body_style)],
        [Paragraph("/api/v1/health", code_style), Paragraph("GET", body_style), Paragraph("System status, model readiness, and memory diagnostic checks", body_style)]
    ]
    t_api = Table(api_specs, colWidths=[140, 50, 340])
    t_api.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0284c7')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(t_api)
    elements.append(Spacer(1, 10))

    # 6. Database Schema & Persistence
    elements.append(Paragraph("6. Database Schema Design", h1_style))
    elements.append(Paragraph(
        "The relational database schema is modeled in SQLAlchemy and stored in SQLite/PostgreSQL. "
        "The <code>scan_records</code> table tracks <code>scan_id</code>, <code>patient_id</code>, patient demographics, image file references, probability vectors, spatial metrics, report text, risk stratification, and timestamps.",
        body_style
    ))

    # 7. Deployment & Setup
    elements.append(Paragraph("7. Docker & Local Deployment Setup", h1_style))
    elements.append(Paragraph(
        "<b>Containerized Launch:</b> <code>docker-compose up --build</code><br/>"
        "<b>Local Python Execution:</b> <code>python train_and_setup.py</code> followed by <code>python run.py</code><br/>"
        "<b>Interactive Web UI:</b> Accessible at <code>http://localhost:8000/</code><br/>"
        "<b>Swagger API Docs:</b> Accessible at <code>http://localhost:8000/docs</code>",
        body_style
    ))

    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=15, spaceAfter=10))
    elements.append(Paragraph("Report Generated automatically by Advanced AI Medical Intelligence Platform | Candidate: Vamshi Cherukuri", code_style))

    doc.build(elements)
    print(f"[PDFGenerator] Generated Technical Project Submission Report at {output_pdf_path}")
    return output_pdf_path
