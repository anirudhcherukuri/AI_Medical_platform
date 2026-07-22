import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import os
from typing import Tuple, Dict, Any
from pathlib import Path

from backend.config import HEATMAPS_DIR

class GradCAM:
    """
    Grad-CAM Engine for Medical Image Interpretability & Feature Localization.
    Calculates gradient-weighted class activation heatmaps on target conv layers.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generates 2D normalized Grad-CAM heatmap array and spatial statistics.
        """
        # Ensure input tensor tracks gradients before forward pass
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class_idx is None:
            target_class_idx = int(torch.argmax(output, dim=1).item())
            
        # Target score
        score = output[0, target_class_idx]
        
        # Zero grads and backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Calculate pooled gradients
        gradients = self.gradients[0] # [C, H, W]
        activations = self.activations[0] # [C, H, W]
        
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True) # [C, 1, 1]
        cam = torch.sum(weights * activations, dim=0) # [H, W]
        
        cam = F.relu(cam) # Apply ReLU to keep positive contributions
        cam_np = cam.cpu().numpy()
        
        # Normalize to [0, 1]
        if np.max(cam_np) > 0:
            cam_np = (cam_np - np.min(cam_np)) / (np.max(cam_np) - np.min(cam_np) + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np)
            
        # Spatial analysis metrics
        high_intensity_pixels = np.sum(cam_np > 0.5)
        total_pixels = cam_np.size
        activation_ratio = float(high_intensity_pixels / total_pixels)
        peak_y, peak_x = np.unravel_index(np.argmax(cam_np), cam_np.shape)
        
        spatial_metrics = {
            "target_class_idx": target_class_idx,
            "peak_intensity": float(np.max(cam_np)),
            "activation_coverage_pct": round(activation_ratio * 100, 2),
            "peak_location_normalized": (round(float(peak_x / cam_np.shape[1]), 2), round(float(peak_y / cam_np.shape[0]), 2)),
            "severity_score": round(float(np.mean(cam_np[cam_np > 0.3])) if np.any(cam_np > 0.3) else 0.0, 2)
        }
        
        return cam_np, spatial_metrics

    def create_overlay(
        self,
        pil_image: Image.Image,
        heatmap_2d: np.ndarray,
        alpha: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates colorized OpenCV heatmap image and blended overlay with original.
        """
        img_np = np.array(pil_image)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            
        height, width = img_np.shape[:2]
        
        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap_2d, (width, height))
        
        # Convert heatmap to uint8 [0, 255]
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply JET Colormap
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
        
        # Alpha blending
        overlay = cv2.addWeighted(img_np, 1.0 - alpha, color_heatmap_rgb, alpha, 0)
        
        return color_heatmap_rgb, overlay

    def save_visualization(
        self,
        pil_image: Image.Image,
        heatmap_2d: np.ndarray,
        filename_prefix: str
    ) -> Dict[str, str]:
        """
        Saves heatmap color image and fused overlay image to storage/heatmaps/.
        Returns relative and absolute file paths.
        """
        color_heatmap, overlay = self.create_overlay(pil_image, heatmap_2d)
        
        heatmap_filename = f"{filename_prefix}_heatmap.png"
        overlay_filename = f"{filename_prefix}_overlay.png"
        
        heatmap_path = HEATMAPS_DIR / heatmap_filename
        overlay_path = HEATMAPS_DIR / overlay_filename
        
        # Save via PIL/OpenCV
        Image.fromarray(color_heatmap).save(heatmap_path)
        Image.fromarray(overlay).save(overlay_path)
        
        return {
            "heatmap_path": str(heatmap_path),
            "overlay_path": str(overlay_path),
            "heatmap_filename": heatmap_filename,
            "overlay_filename": overlay_filename
        }
