# Advanced AI Medical Intelligence Platform - Complete Technical Documentation

> **GitHub Repository:** [https://github.com/anirudhcherukuri/AI_Medical_platform](https://github.com/anirudhcherukuri/AI_Medical_platform)  
> **Target Application:** End-to-End Thoracic Chest Radiograph Diagnostics, Explainable AI (Grad-CAM), and LLM Radiology Report Platform.

---

## 📖 Table of Contents
1. [Project Overview & Objectives](#1-project-overview--objectives)
2. [Full System Architecture](#2-full-system-architecture)
3. [Pre-Inference Validation Layer & Anatomical Verification](#3-pre-inference-validation-layer--anatomical-verification)
4. [Deep Learning Model Architecture & Inference](#4-deep-learning-model-architecture--inference)
5. [Explainable AI (Grad-CAM) Engine](#5-explainable-ai-grad-cam-engine)
6. [LLM Radiology Report Synthesis & Fallback Engine](#6-llm-radiology-report-synthesis--fallback-engine)
7. [Database Schema & SQLAlchemy Audit Trail](#7-database-schema--sqlalchemy-audit-trail)
8. [ReportLab PDF Report Generation](#8-reportlab-pdf-report-generation)
9. [Web Application & UI/UX Studio](#9-web-application--uiux-studio)
10. [API Specifications & OpenAPI Endpoints](#10-api-specifications--openapi-endpoints)
11. [Installation, Setup, & Execution Guide](#11-installation-setup--execution-guide)
12. [Testing & Verification Suite](#12-testing--verification-suite)

---

## 1. Project Overview & Objectives

The **Advanced AI Medical Intelligence Platform** is a full-stack, production-grade medical artificial intelligence application. It assists radiologists, clinical technicians, and medical practitioners by providing automated diagnostic decision support for **Chest Radiographs (X-Rays)**.

### Core Objectives:
- **Thoracic Pathology Classification:** Classify input radiographs into 3 distinct diagnostic categories: `Normal`, `Pneumonia`, and `COVID-19`.
- **Anatomical Validation:** Reject invalid file formats, corrupted files, non-medical photos, and non-thoracic radiographs (Abdominal X-Rays, Hand X-Rays, CT Scans) before running deep learning inference.
- **Explainable AI (XAI):** Generate pixel-level feature attribution heatmaps using **Grad-CAM**, overlaid on original radiographs with interactive opacity controls and spatial metrics (affected coverage %, peak intensity location, severity score).
- **LLM Radiology Report Synthesis:** Automatically synthesize structured radiology evaluation reports using **Google Gemini 1.5 Flash API**, backed by an offline **Clinical Logic Engine** for zero-downtime fallback.
- **Audit Logging & PDF Reporting:** Store all diagnostic evaluations in an **SQLite** database and compile printable, professional PDF radiology reports using **ReportLab**.

---

## 2. Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                Modern Web Application Studio                                    │
│       (Glassmorphism UI | Drag & Drop Uploader | Grad-CAM Opacity Slider | Records | Analytics)   │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │ REST API (JSON / Multipart)
┌────────────────────────────────────────────────▼────────────────────────────────────────────────┐
│                                     FastAPI Backend Server                                      │
│   ├── /api/v1/predict   ├── /api/v1/history           ├── /api/v1/analytics                       │
│   ├── /api/v1/health    └── /api/v1/model/info        └── /docs (Swagger OpenAPI UI)              │
└───────┬────────────────────────────────────────┬────────────────────────────────────────┬───────┘
        │                                        │                                        │
┌───────▼───────────────────────┐ ┌──────────────▼───────────────┐ ┌──────────────▼───────────────┐
│   0. Validation Layer         │ │   1. PyTorch Deep Learning   │ │   2. Explainable AI Engine    │
│ (Extension, MIME, Size,       │ │   (DenseNet121 Backbone,     │ │   (Grad-CAM Feature           │
│  Integrity, Chest X-Ray Check)│ │    1024->512->256 Classifier) │ │    Activation Heatmaps)       │
└───────────────────────────────┘ └──────────────┬───────────────┘ └──────────────┬───────────────┘
                                                 │                                │
┌────────────────────────────────────────────────▼────────────────────────────────▼───────────────┐
│   3. LLM Radiology Synthesizer         4. ReportLab PDF Engine          5. Database Layer       │
│   (Gemini API / Offline Logic)         (Clinical Radiology PDFs)        (SQLAlchemy & SQLite)   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Pre-Inference Validation Layer & Anatomical Verification

Located in `backend/utils/validation.py`, the validation layer guarantees that the system only processes valid Thoracic Chest Radiographs.

### Validation Sequence:
1. **File Extension & MIME Type Check (`validate_file`)**:
   - Allowed extensions: `.png`, `.jpg`, `.jpeg`.
   - Allowed MIME types: `image/png`, `image/jpeg`.
   - Maximum upload size: **15MB**.
2. **Image Integrity Verification (`validate_image`)**:
   - Reads image bytes and calls `PIL.Image.verify()`.
   - Rejects empty (0-byte) or corrupted files with HTTP 400.
3. **Anatomical Chest X-Ray Verification (`validate_chest_xray`)**:
   - **Color Variance Check:** Computes RGB channel variance $\text{mean}(|R-G| + |G-B|)$. If $>5.0$, rejects standard color photographs (cats, landscapes, documents).
   - **Structural Background Void Check:** Evaluates black background ratio ($<25$ intensity). If $>70\%$, rejects extremity/hand/foot X-Rays.
   - **Circular CT FOV Check:** Analyzes corner intensity versus central region to identify circular field-of-view axial Head/Brain CT Scans.
   - **Bilateral Thoracic Pulmonary Lung Cavity Verification:** Analyzes upper-mid thoracic lung zones versus central mediastinum/spine contrast. If upper dark lung cavities are absent (e.g. in Abdominal X-Rays), rejects the image with HTTP 400.

---

## 4. Deep Learning Model Architecture & Inference

Located in `backend/models/classifier.py`:

- **Backbone:** PyTorch `DenseNet121` pretrained on ImageNet. Dense connectivity allows feature propagation and feature reuse across dense blocks.
- **Target Convolutional Layer:** `densenet.features.denseblock4.denselayer16`.
- **Custom Classification Head:**
  ```python
  Sequential(
      BatchNorm1d(1024), Dropout(0.5),
      Linear(1024, 512), ReLU(inplace=True),
      BatchNorm1d(512), Dropout(0.5),
      Linear(512, 256), ReLU(inplace=True),
      Dropout(0.5),
      Linear(256, 3) # ['COVID-19', 'Normal', 'Pneumonia']
  )
  ```
- **Inference Pipeline:**
  - Input image converted to RGB and resized to $224 \times 224$.
  - Normalized using ImageNet mean (`[0.485, 0.456, 0.406]`) and std (`[0.229, 0.224, 0.225]`).
  - Evaluated under `@torch.no_grad()`. Softmax produces normalized class probabilities.

---

## 5. Explainable AI (Grad-CAM) Engine

Located in `backend/xai/gradcam.py`:

- **Grad-CAM Mathematical Formulation:**  
  Computes gradients of target class logit score $y^c$ with respect to feature activation maps $A^k$ of the final dense layer:
  $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
  The coarse class activation heatmap is generated via:
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$
- **Visualization:** Normalized heatmaps are color-mapped using OpenCV `COLORMAP_JET` and alpha-blended ($50\%$) with the original X-Ray.
- **Spatial Metrics Extracted:**
  - **Affected Area Coverage %:** Percentage of lung pixels with intensity $>0.5$.
  - **Peak Location:** Normalized $(X, Y)$ coordinates of maximum activation.
  - **Severity Score:** Average intensity across high-activation regions ($>0.3$).

---

## 6. LLM Radiology Report Synthesis & Fallback Engine

Located in `backend/llm/report_generator.py`:

- **Primary Engine:** Google Gemini 1.5 Flash API (`google-genai` / `google-generativeai` SDK).
- **Prompt Structure:** Combines patient demographics, model confidence, class probabilities, and Grad-CAM spatial metrics to synthesize:
  1. Clinical Impression
  2. Radiological Findings
  3. Explainable AI (Grad-CAM) Interpretation
  4. Recommended Clinical Follow-up Plan
- **Offline Fallback Engine:** If an API key is missing or network fails, an offline **Clinical Logic Engine** automatically generates structured radiological reports, guaranteeing 100% platform availability.

---

## 7. Database Schema & SQLAlchemy Audit Trail

Located in `backend/database/models.py` and `crud.py`:

Stored in `storage/medical_intelligence.db` (SQLite):

```sql
CREATE TABLE scan_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id VARCHAR UNIQUE INDEX,
    patient_id VARCHAR INDEX,
    patient_name VARCHAR,
    age INTEGER,
    gender VARCHAR,
    scan_type VARCHAR,
    original_image_path VARCHAR,
    heatmap_image_path VARCHAR,
    overlay_image_path VARCHAR,
    predicted_class VARCHAR INDEX,
    confidence FLOAT,
    probabilities_json TEXT,
    spatial_metrics_json TEXT,
    activation_coverage_pct FLOAT,
    severity_score FLOAT,
    report_text TEXT,
    risk_level VARCHAR,
    report_source VARCHAR,
    created_at DATETIME INDEX
);
```

---

## 8. ReportLab PDF Report Generation

Located in `backend/utils/pdf_generator.py`:

- Generates individual patient **Clinical Radiology Reports** containing:
  - Header banner & candidate/institution metadata
  - Patient demographic table
  - Primary AI Diagnosis badge & confidence rating
  - Side-by-side Visual Evidence (Original Radiograph vs Grad-CAM Fused Overlay)
  - Detailed radiological findings & clinical plan
  - Medical decision support disclaimers

---

## 9. Web Application & UI/UX Studio

Located in `frontend/`:
- **Diagnostic Studio Tab:** Drag-and-drop X-Ray file uploader, sample preset buttons (`Normal`, `Pneumonia`, `COVID-19`), side-by-side interactive opacity slider, and live probability breakdown progress bars.
- **Glassmorphism Unsupported Image Warning Modal:** Replaces native browser alerts. Displays explicit warning instructions whenever an unsupported or non-Chest X-Ray image is uploaded.
- **Patient Records Tab:** Searchable and filterable diagnostic history audit table.
- **Platform Analytics Tab:** System-wide metrics, class distribution bars, and model training performance cards (training curves, confusion matrix, ROC curves).

---

## 10. API Specifications & OpenAPI Endpoints

Accessible via Swagger UI at `http://localhost:8000/docs`:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/v1/predict` | `POST` | Upload X-Ray -> Validation -> DenseNet -> Grad-CAM -> Gemini -> DB -> PDF |
| `/api/v1/history` | `GET` | Retrieve searchable/filterable patient audit history records |
| `/api/v1/history/{scan_id}` | `GET` | Fetch specific scan record details |
| `/api/v1/analytics` | `GET` | Platform statistics, class distribution, and average confidence |
| `/api/v1/model/info` | `GET` | DenseNet model metadata, accuracy metrics, and training charts |
| `/api/v1/report/pdf/download/{pdf_name}` | `GET` | Download printable PDF clinical report |
| `/api/v1/project-report` | `GET` | Download technical project submission report |
| `/api/v1/health` | `GET` | System status, database connection, and model readiness check |

---

## 11. Installation, Setup, & Execution Guide

### Option 1: Local Python Execution

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/anirudhcherukuri/AI_Medical_platform.git
   cd AI_Medical_platform
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize System (Train Model, Seed DB, Generate PDF Report):**
   ```bash
   python train_and_setup.py
   ```

4. **Launch Application Server:**
   ```bash
   python run.py
   ```

5. **Access Interfaces:**
   - Web Studio: `http://localhost:8000/`
   - Swagger API Docs: `http://localhost:8000/docs`

---

## 12. Testing & Verification Suite

Run automated unit tests using `pytest`:
```bash
python -m pytest tests/ -v
```

### Verification Matrix:
- `Chest X-Ray` $\rightarrow$ **ACCEPTED** (DenseNet & Grad-CAM pipeline executed)
- `Abdominal X-Ray` $\rightarrow$ **REJECTED** (HTTP 400 - Glassmorphic Warning Modal)
- `Hand X-Ray` $\rightarrow$ **REJECTED** (HTTP 400 - Glassmorphic Warning Modal)
- `CT Scan` $\rightarrow$ **REJECTED** (HTTP 400 - Glassmorphic Warning Modal)
- `Random Photo` $\rightarrow$ **REJECTED** (HTTP 400 - Glassmorphic Warning Modal)
- `PDF File` $\rightarrow$ **REJECTED** (HTTP 400 - Glassmorphic Warning Modal)
- `Corrupted File` $\rightarrow$ **REJECTED** (HTTP 400 - Glassmorphic Warning Modal)
