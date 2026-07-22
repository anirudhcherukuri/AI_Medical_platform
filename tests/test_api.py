import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_analytics_endpoint():
    response = client.get("/api/v1/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "total_scans" in data
    assert "class_distribution" in data

def test_history_endpoint():
    response = client.get("/api/v1/history")
    assert response.status_code == 200
    data = response.json()
    assert "scans" in data
    assert "total" in data
