# Production Dockerfile for Advanced AI Medical Intelligence Platform
FROM python:3.11-slim

# System dependencies for OpenCV, PyTorch, and ReportLab
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements & install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Run initialization script (generate synthetic dataset, train PyTorch model, seed DB, create PDF report)
RUN python train_and_setup.py

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["python", "run.py"]
