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
    Validates whether the input image is a Thoracic Chest Radiograph (Chest X-Ray).
    Checks color variance, background ratio, structural density, and bilateral thoracic lung cavity structures.
    Returns (is_valid, reason).
    """
    arr = np.array(pil_image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return False, "Image channel format is invalid."

    height, width, _ = arr.shape
    if height < 64 or width < 64:
        return False, "Image dimensions are too small for medical diagnostic analysis."

    # 1. Color Variance Check (Reject non-grayscale standard photos)
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)

    diff_rg = np.mean(np.abs(r - g))
    diff_gb = np.mean(np.abs(g - b))
    color_variance = diff_rg + diff_gb

    if color_variance > 5.0:
        return False, "Uploaded image is a color photo, not a medical radiograph."

    # 2. Structural Geometry & Background Void Check
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    resized_gray = cv2.resize(gray, (256, 256))

    # Background black pixel ratio (< 25 intensity)
    black_pixels = np.sum(resized_gray < 25)
    total_pixels = resized_gray.size
    black_ratio = black_pixels / total_pixels

    # Hand/Foot/Extremity X-Rays usually have 70%+ black background void surrounding thin bones
    if black_ratio > 0.70:
        return False, "Uploaded image is an extremity or hand X-Ray, not a Chest X-Ray."

    # Extremely bright / white documents / line drawings
    white_pixels = np.sum(resized_gray > 230)
    if (white_pixels / total_pixels) > 0.60:
        return False, "Uploaded image is a document or diagram, not a Chest X-Ray."

    # 3. Axial CT Scan FOV Check
    # Axial head/brain CT scans have circular boundary with near-black outer corners
    top_left_corner = np.mean(resized_gray[:35, :35])
    top_right_corner = np.mean(resized_gray[:35, 221:])
    bottom_left_corner = np.mean(resized_gray[221:, :35])
    bottom_right_corner = np.mean(resized_gray[221:, 221:])
    avg_corner_intensity = (top_left_corner + top_right_corner + bottom_left_corner + bottom_right_corner) / 4.0

    center_region = np.mean(resized_gray[80:176, 80:176])
    
    # CT scans usually have black corners (<15) and uniform circular head center with skull ring
    if avg_corner_intensity < 12.0 and center_region > 65.0:
        ring_sample_1 = np.mean(resized_gray[40:60, 40:216])
        ring_sample_2 = np.mean(resized_gray[196:216, 40:216])
        if abs(ring_sample_1 - ring_sample_2) < 15.0 and black_ratio > 0.40:
            return False, "Uploaded image is an Axial CT Scan, not a Chest X-Ray."

    # 4. Bilateral Thoracic Pulmonary Lung Cavity Verification
    h_256, w_256 = 256, 256
    left_lung_zone = resized_gray[int(0.20 * h_256):int(0.65 * h_256), int(0.12 * w_256):int(0.42 * w_256)]
    right_lung_zone = resized_gray[int(0.20 * h_256):int(0.65 * h_256), int(0.58 * w_256):int(0.88 * w_256)]
    mediastinum_zone = resized_gray[int(0.20 * h_256):int(0.65 * h_256), int(0.40 * w_256):int(0.60 * w_256)]

    left_lung_mean = float(np.mean(left_lung_zone))
    right_lung_mean = float(np.mean(right_lung_zone))
    mediastinum_mean = float(np.mean(mediastinum_zone))

    avg_lung_intensity = (left_lung_mean + right_lung_mean) / 2.0

    # In Chest X-Rays:
    # a) Lung fields contain dark air cavities
    # b) Central mediastinum (heart/spine) is brighter than bilateral lung fields
    if avg_lung_intensity > 155.0:
        return False, "Uploaded image is an Abdominal X-Ray or non-chest radiograph."

    if mediastinum_mean < (avg_lung_intensity - 10.0):
        return False, "Image lacks central mediastinum / thoracic cavity structure."

    if left_lung_mean > 165.0 or right_lung_mean > 165.0:
        return False, "Image does not display bilateral thoracic lung fields."

    return True, "Valid Thoracic Chest Radiograph."
