# Production Dockerfile for Advanced AI Medical Intelligence Platform

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required directories (if they don't exist)
RUN mkdir -p \
    storage/models \
    storage/uploads \
    storage/heatmaps \
    storage/reports \
    storage/samples

# Expose FastAPI port
EXPOSE 8000

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start the application
CMD ["python", "run.py"]
