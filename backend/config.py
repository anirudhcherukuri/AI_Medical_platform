import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Storage Directories
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
HEATMAPS_DIR = STORAGE_DIR / "heatmaps"
REPORTS_DIR = STORAGE_DIR / "reports"
MODELS_DIR = STORAGE_DIR / "models"
SAMPLES_DIR = STORAGE_DIR / "samples"

# Ensure all directories exist
for directory in [STORAGE_DIR, UPLOADS_DIR, HEATMAPS_DIR, REPORTS_DIR, MODELS_DIR, SAMPLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Settings
MODEL_SAVE_PATH = MODELS_DIR / "chest_xray_densenet121.pth"
NUM_CLASSES = 3
CLASS_NAMES = ["Normal", "Pneumonia", "COVID-19"]

# Database Settings
DATABASE_URL = f"sqlite:///{STORAGE_DIR}/medical_intelligence.db"

# LLM & External API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PREFERRED_LLM_MODEL = "gemini-1.5-flash"

# Server Configuration
HOST = "0.0.0.0"
PORT = 8000
DEBUG = True
