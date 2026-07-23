import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from typing import Tuple, Dict, List
import numpy as np

from backend.config import MODEL_SAVE_PATH, NUM_CLASSES, CLASS_NAMES

class MedicalChestXRayClassifier(nn.Module):
    """
    Deep Learning Model for Chest X-Ray Medical Image Classification.
    Uses DenseNet121 architecture optimized for thoracic pathology detection.
    Classifier head matches the Kaggle COVID-19 Radiography training setup.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super(MedicalChestXRayClassifier, self).__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.densenet = models.densenet121(weights=weights)

        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self.target_layer = self.densenet.features.denseblock4.denselayer16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet(x)

    def get_target_layer(self):
        return self.target_layer

# Data Transformations
def get_transforms():
    """
    Standard image transformation pipeline for Chest X-Ray evaluation.
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return preprocess

class MedicalModelManager:
    """
    Manager class handling model loading, inference, and classification metrics.
    """
    def __init__(self, model_path: str = str(MODEL_SAVE_PATH)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MedicalChestXRayClassifier(num_classes=NUM_CLASSES, pretrained=False)
        self.transforms = get_transforms()
        self.class_names = CLASS_NAMES
        self.model_path = model_path
        self.weights_loaded = False
        self.val_acc = None
        self.epoch = None
        
        # Load model weights if present
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                if any(key.startswith("features.") for key in state_dict):
                    self.model.densenet.load_state_dict(state_dict)
                else:
                    self.model.load_state_dict(state_dict)
                self.weights_loaded = True
                if isinstance(checkpoint, dict):
                    self.val_acc = checkpoint.get("val_acc")
                    self.epoch = checkpoint.get("epoch")
                val_acc = self.val_acc
                acc_msg = f" (val acc: {val_acc * 100:.1f}%)" if val_acc is not None else ""
                print(f"[ModelManager] Loaded model weights from {self.model_path}{acc_msg}")
            except Exception as e:
                print(f"[ModelManager] Warning loading state dict ({e}). Initializing default weights.")
        else:
            print(f"[ModelManager] No saved weights found at {self.model_path}. Model running with default initialization.")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path: str) -> Dict:
        """
        Runs inference on an input medical image file path.
        Returns predicted class, confidence, class probabilities, and tensor.
        """
        try:
            pil_img = Image.open(image_path).convert("RGB")
            tensor_img = self.transforms(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor_img)
                probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
                
            pred_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[pred_idx])
            
            prob_dict = {
                self.class_names[i]: float(probabilities[i])
                for i in range(len(self.class_names))
            }
            
            return {
                "predicted_class": self.class_names[pred_idx],
                "class_index": pred_idx,
                "confidence": confidence,
                "probabilities": prob_dict,
                "tensor": tensor_img,
                "pil_image": pil_img
            }
        except Exception as e:
            raise RuntimeError(f"Error during model inference: {str(e)}")

# Global Singleton Instance
_model_manager = None

def get_model_manager() -> MedicalModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = MedicalModelManager()
    return _model_manager
