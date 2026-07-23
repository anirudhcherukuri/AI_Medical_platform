import io
import os
import numpy as np
import cv2
from PIL import Image
from fastapi import UploadFile, HTTPException
from typing import Tuple

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB

def validate_file(file: UploadFile, max_size_mb: float = 15.0):
    """
    Validates file extension, MIME type, and maximum upload size.
    Raises HTTPException 400 if validation fails.
    """
    # 1. File Extension Validation
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only PNG, JPG, and JPEG files are allowed."
        )

    # 2. MIME Type Validation
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only image/png and image/jpeg MIME types are allowed."
        )

    # 3. File Size Validation
    if hasattr(file, "size") and file.size is not None:
        if file.size > max_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed limit of {max_size_mb}MB."
            )

def validate_image(file_bytes: bytes) -> Image.Image:
    """
    Validates image file integrity using Pillow.
    Ensures file is not corrupted, empty, or unreadable.
    Returns RGB PIL Image.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid image. Uploaded file is empty (0 bytes)."
        )

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds maximum allowed limit of 15MB."
        )

    try:
        # Verify integrity
        img_stream = io.BytesIO(file_bytes)
        pil_img = Image.open(img_stream)
        pil_img.verify()

        # Re-open after verify (verify invalidates stream pointer)
        img_stream.seek(0)
        pil_img_rgb = Image.open(img_stream).convert("RGB")
        return pil_img_rgb
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image. File is corrupted or unreadable."
        )

def validate_chest_xray(pil_image: Image.Image) -> Tuple[bool, str]:
    """
    Enhanced Chest X-Ray Validator
    Rejects:
    - Color photos
    - Documents
    - CT scans
    - Hand/Foot X-rays
    - Abdomen/Pelvis X-rays
    """

    arr = np.array(pil_image)

    if arr.ndim != 3 or arr.shape[2] != 3:
        return False, "Invalid image format."

    h, w, _ = arr.shape

    if h < 128 or w < 128:
        return False, "Image resolution is too small."

    # -----------------------------
    # Convert to grayscale
    # -----------------------------
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (256, 256))

    # -----------------------------
    # Reject color photographs
    # -----------------------------
    color_std = np.std(arr[:, :, 0] - arr[:, :, 1]) + np.std(arr[:, :, 1] - arr[:, :, 2])

    if color_std > 8:
        return False, "Uploaded image appears to be a normal color photograph."

    # -----------------------------
    # Reject white documents
    # -----------------------------
    white_ratio = np.mean(gray > 235)

    if white_ratio > 0.55:
        return False, "Uploaded image appears to be a document."

    # -----------------------------
    # Reject extremely dark images
    # -----------------------------
    black_ratio = np.mean(gray < 20)

    if black_ratio > 0.75:
        return False, "Uploaded image is not a diagnostic Chest X-Ray."

    # -----------------------------
    # Reject CT scans
    # -----------------------------
    corner = np.mean([
        gray[:35, :35],
        gray[:35, -35:],
        gray[-35:, :35],
        gray[-35:, -35:]
    ])

    center = np.mean(gray[80:176, 80:176])

    if corner < 15 and center > 70:
        return False, "CT scan detected."

    # -----------------------------
    # Lung regions
    # -----------------------------
    left = gray[45:170, 30:110]
    center_region = gray[45:170, 105:150]
    right = gray[45:170, 146:226]

    left_mean = np.mean(left)
    right_mean = np.mean(right)
    center_mean = np.mean(center_region)

    left_std = np.std(left)
    right_std = np.std(right)

    # lungs should exist
    if left_std < 18 or right_std < 18:
        return False, "Thoracic lung fields not detected."

    # mediastinum should be brighter
    if center_mean < ((left_mean + right_mean) / 2):
        return False, "Thoracic anatomy not detected."

    # -----------------------------
    # Abdomen detection
    # -----------------------------
    upper = gray[:120, :]
    lower = gray[150:, :]

    edges_upper = cv2.Canny(upper, 50, 120)
    edges_lower = cv2.Canny(lower, 50, 120)

    upper_density = np.mean(edges_upper > 0)
    lower_density = np.mean(edges_lower > 0)

    if lower_density > upper_density * 1.35:
        return False, "Image appears to be an abdominal or pelvic radiograph."

    # -----------------------------
    # Spine continuity
    # -----------------------------
    spine = gray[:, 115:140]

    if np.std(spine) < 10:
        return False, "Chest anatomy not detected."

    # -----------------------------
    # Final confidence score
    # -----------------------------
    score = 0

    if color_std < 5:
        score += 1

    if left_std > 20:
        score += 1

    if right_std > 20:
        score += 1

    if center_mean > left_mean:
        score += 1

    if upper_density > lower_density:
        score += 1

    if score < 4:
        return False, "Uploaded image is not a valid Chest X-Ray."

    return True, "Valid Chest X-Ray"
    
