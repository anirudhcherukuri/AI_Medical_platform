from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse
import os
from pathlib import Path

from backend.config import BASE_DIR, STORAGE_DIR, HOST, PORT
from backend.database.connection import init_db
from backend.api.routes import router as api_router
from backend.models.classifier import get_model_manager

app = FastAPI(
    title="Advanced AI Medical Intelligence Platform",
    description="End-to-end Medical Image Diagnostics, Grad-CAM Explainable AI, and LLM Radiology Report System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup Event: Initialize DB and Pre-load PyTorch Model
@app.on_event("startup")
def startup_event():
    init_db()
    print("[Main] Initializing PyTorch Deep Learning Model Manager...")
    get_model_manager()
    print("[Main] Server startup complete. Ready to receive diagnostic requests.")

# Include API Router
app.include_router(api_router)

# Mount Static File Directories
frontend_dir = BASE_DIR / "frontend"

if STORAGE_DIR.exists():
    app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")

if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/")
def serve_index():
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)
