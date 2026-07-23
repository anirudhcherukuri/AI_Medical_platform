# Advanced AI Medical Intelligence Platform 🏥🤖

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Repository:** [https://github.com/anirudhcherukuri/AI_Medical_platform](https://github.com/anirudhcherukuri/AI_Medical_platform)  
> **Technical Evaluation Project Assignment** for the position of **AI/ML Engineer** at **SN Matrix Software Pvt. Ltd.**

---

## 📌 Executive Summary

The **Advanced AI Medical Intelligence Platform** is a production-grade, end-to-end medical AI application designed to assist radiologists and healthcare professionals in screening Chest Radiographs (X-Rays). 

It combines **Pre-Inference Anatomical Validation** (restricting uploads strictly to valid Thoracic Chest X-Rays), **Deep Learning** (PyTorch DenseNet121), **Explainable AI** (Grad-CAM feature localization), **Large Language Models** (Google Gemini 1.5 Flash API with offline clinical logic fallback), **FastAPI REST APIs**, **SQLAlchemy Database tracking**, **ReportLab PDF Generation**, and an interactive glassmorphic **Web Application Studio**.

---

## 🚀 Key Features & Capabilities

1. **Pre-Inference Validation & Anatomical Verification**:
   - Validates file extensions (`.png`, `.jpg`, `.jpeg`), MIME types (`image/png`, `image/jpeg`), file size (<15MB), and image integrity (`Image.verify()`).
   - Computer vision anatomical verification engine evaluating color variance, background ratio, circular CT FOV, and bilateral thoracic lung cavity structures.
   - Immediately halts pipeline and presents a glassmorphic warning modal on non-chest images (Abdominal X-Rays, Hand X-Rays, CT Scans, Random Photos, PDFs, or Corrupted Files).

2. **Deep Learning Medical Imaging Classifier (`PyTorch`)**:
   - DenseNet121 architecture fine-tuned for multi-class thoracic pathology detection (`Normal`, `Pneumonia`, `COVID-19`).
   - Automated training pipeline (`train_and_setup.py`) calibrating weights with synthetic X-Ray data.

2. **Explainable AI (Grad-CAM / Grad-CAM++)**:
   - Gradient-weighted Class Activation Mapping hooked to final dense convolutional blocks (`denseblock4`).
   - Generates normalized 2D heatmaps, JET color mapping, and adjustable alpha blended overlays highlighting focal pulmonary opacities.
   - Computes spatial localization metrics (affected area coverage %, peak intensity coordinates, severity score).

3. **LLM-Assisted Radiology Report Synthesis**:
   - Integrates with **Google Gemini 1.5 Flash API** to generate clinical-grade radiology reports.
   - Fallback offline Medical Logic Engine for standalone operation without API keys.

4. **REST API Backend (`FastAPI`)**:
   - Async endpoints for image prediction, Grad-CAM extraction, PDF report generation, audit history querying, and aggregate platform analytics.
   - Auto-generated OpenAPI / Swagger documentation (`/docs`).

5. **Database Audit Trail (`SQLAlchemy + SQLite/PostgreSQL`)**:
   - Stores patient demographics, prediction vectors, spatial metrics, image file paths, risk levels, and generated report text.

6. **Interactive Web Application (`Vanilla JS + HTML5 + CSS3 Glassmorphism`)**:
   - Dark-mode diagnostic studio featuring drag-and-drop X-Ray uploader, side-by-side Grad-CAM opacity sliders, patient history browser, and live analytics dashboard.

7. **PDF Clinical & Technical Project Reports (`ReportLab`)**:
   - Automated PDF report generator creating individual patient clinical reports and the full assignment technical submission report (`Advanced_AI_Medical_Intelligence_Platform_Report.pdf`).

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Modern Web Application (Frontend)                      │
│   (Interactive Image Upload | Grad-CAM Opacity Slider | LLM Report | Analytics)  │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ REST API (JSON / Multipart)
┌────────────────────────────────────────▼────────────────────────────────────────┐
│                              FastAPI Backend Server                             │
│   ├── /api/v1/predict   ├── /api/v1/gradcam   ├── /api/v1/history                 │
│   ├── /api/v1/report    ├── /api/v1/analytics └── /docs (Swagger API Docs)        │
└───────┬────────────────────────┬────────────────────────┬───────────────────────┘
        │                        │                        │
┌───────▼──────────────┐ ┌───────▼──────────────┐ ┌───────▼──────────────┐
│  Deep Learning Engine │ │   Explainable AI    │ │ LLM Medical Reporter │
│  (PyTorch ResNet50/  │ │ (Grad-CAM Layer      │ │ (Gemini API /        │
│   DenseNet121)       │ │  Heatmap Generator)  │ │  Clinical Prompt)    │
└───────┬──────────────┘ └───────┬──────────────┘ └───────┬──────────────┘
        │                        │                        │
┌───────▼────────────────────────▼────────────────────────▼───────────────────────┐
│                          Database Layer (SQLAlchemy & SQLite)                   │
│   (Patient Metadata | Diagnostic Results | Heatmap Storage | Full Clinical History) │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 Tech Stack Breakdown

- **Language**: Python 3.11
- **Deep Learning**: PyTorch, Torchvision, NumPy, OpenCV, Pillow
- **Explainable AI**: Custom PyTorch Grad-CAM Hook Engine
- **LLM Integration**: Google Gemini API (`google-genai` / `google-generativeai`)
- **Web Backend**: FastAPI, Uvicorn, Pydantic, Python-Multipart
- **Database**: SQLAlchemy, SQLite
- **PDF Reporting**: ReportLab
- **Testing**: PyTest
- **Containerization**: Docker, Docker-Compose

---

## 🛠️ Quickstart Setup & Installation

### Option 1: Local Python Execution (Recommended)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/vamshicherukuri/AI_Medical_Platform.git
   cd AI_Medical_Platform
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize System (Train Model, Seed DB, Generate PDF Report)**:
   ```bash
   python train_and_setup.py
   ```

4. **Launch Application Server**:
   ```bash
   python run.py
   ```

5. **Access Application**:
   - **Web Application Studio**: `http://localhost:8000/`
   - **Interactive API Documentation**: `http://localhost:8000/docs`
   - **Download PDF Project Report**: `http://localhost:8000/api/v1/project-report`

---

### Option 2: Docker Containerization

1. **Build and Run via Docker Compose**:
   ```bash
   docker-compose up --build
   ```

2. Access the Web Application at `http://localhost:8000/`.

---

## 🧪 Running Unit Tests

To run the automated `pytest` test suite:
```bash
python -m pytest tests/ -v
```

Expected Output:
```
tests/test_api.py::test_health_endpoint PASSED                           [ 20%]
tests/test_api.py::test_analytics_endpoint PASSED                        [ 40%]
tests/test_api.py::test_history_endpoint PASSED                          [ 60%]
tests/test_model.py::test_model_initialization PASSED                    [ 80%]
tests/test_model.py::test_inference_and_gradcam PASSED                   [100%]
======================= 5 passed in 25.65s =======================
```

---

## 📂 Project Repository Structure

```
d:/AI_Medical_Platform/
├── backend/
│   ├── api/             # FastAPI REST endpoints & Pydantic schemas
│   ├── database/        # SQLAlchemy DB connection, models, CRUD
│   ├── llm/             # Gemini LLM medical report generator
│   ├── models/          # PyTorch DenseNet121 medical classifier
│   ├── xai/             # Grad-CAM Explainable AI engine
│   ├── utils/           # ReportLab PDF generator
│   ├── config.py        # Centralized paths and configuration
│   └── main.py          # FastAPI application server
├── frontend/
│   ├── css/styles.css   # Dark glassmorphism design system
│   ├── js/app.js        # Frontend client logic & REST calls
│   └── index.html       # Web studio layout
├── storage/             # Generated heatmaps, uploads, reports, models
├── tests/               # PyTest unit test suite
├── Dockerfile           # Docker container file
├── docker-compose.yml   # Multi-container orchestration
├── requirements.txt     # Pinned Python dependencies
├── train_and_setup.py   # 1-click dataset build, training & setup
├── run.py               # 1-click server launch script
└── README.md            # Repository documentation
```

---

## 📋 Evaluation Criteria Mapping

| Evaluation Criteria | Implementation Detail |
| :--- | :--- |
| **Deep Learning Performance** | PyTorch DenseNet121 with custom linear head fine-tuned for 3-class Chest X-Ray diagnosis. |
| **Explainable AI (Grad-CAM)** | Layer hook activation extraction, JET colormap, alpha overlay, and spatial metrics. |
| **LLM Integration** | Gemini 1.5 Flash API with offline clinical logic fallback engine. |
| **API Development** | RESTful FastAPI server with OpenAPI Swagger UI at `/docs`. |
| **Database Design** | SQLAlchemy ORM tracking patient records, scan IDs, image paths, probabilities, and reports. |
| **Web Application** | Modern responsive glassmorphic UI with drag & drop uploader, opacity slider, history, and analytics. |
| **Documentation & Report** | Automated PDF project report generator and clean README. |
| **Deployment** | Dockerfile and `docker-compose.yml` for instant production deployment. |

---

## 📄 License & Contact

Developed by **Vamshi Cherukuri** for technical evaluation at **SN Matrix Software Pvt. Ltd.**  
Email: [vamshicheukuri@gmail.com](mailto:vamshicheukuri@gmail.com)
