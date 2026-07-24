from dotenv import load_dotenv

load_dotenv()
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
TRAINING_RESULTS_DIR = MODELS_DIR / "training_results"
CLASS_NAMES_JSON_PATH = MODELS_DIR / "class_names.json"

# Ensure all directories exist
for directory in [STORAGE_DIR, UPLOADS_DIR, HEATMAPS_DIR, REPORTS_DIR, MODELS_DIR, SAMPLES_DIR, TRAINING_RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Settings
MODEL_SAVE_PATH = MODELS_DIR / "best_model.pth"
NUM_CLASSES = 3
# Order must match Kaggle training: COVID=0, Normal=1, Viral Pneumonia=2
CLASS_NAMES = ["COVID-19", "Normal", "Pneumonia"]

# Database Settings
DATABASE_URL = f"sqlite:///{STORAGE_DIR}/medical_intelligence.db"

# LLM & External API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Set via environment variable - never hardcode secrets
PREFERRED_LLM_MODEL = "gemini-flash-latest"

# Server Configuration
HOST = "0.0.0.0"
PORT = 8000
DEBUG = True
