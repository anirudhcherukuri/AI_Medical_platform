import pytest
import torch
import numpy as np
from PIL import Image
from backend.models.classifier import MedicalChestXRayClassifier, MedicalModelManager
from backend.xai.gradcam import GradCAM
from backend.config import SAMPLES_DIR

def test_model_initialization():
    model = MedicalChestXRayClassifier(num_classes=3, pretrained=False)
    assert model is not None
    assert model.get_target_layer() is not None

def test_inference_and_gradcam():
    manager = MedicalModelManager()
    sample_image_path = str(SAMPLES_DIR / "sample_normal.jpg")
    
    # Test Prediction
    result = manager.predict(sample_image_path)
    assert "predicted_class" in result
    assert result["predicted_class"] in ["Normal", "Pneumonia", "COVID-19"]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0

    # Test Grad-CAM
    gradcam = GradCAM(manager.model, manager.model.get_target_layer())
    heatmap_2d, spatial_metrics = gradcam.generate_heatmap(
        result["tensor"],
        result["class_index"]
    )
    assert heatmap_2d is not None
    assert heatmap_2d.shape == (7, 7) or len(heatmap_2d.shape) == 2
    assert "activation_coverage_pct" in spatial_metrics
    assert "severity_score" in spatial_metrics
