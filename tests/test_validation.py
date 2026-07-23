import io
import pytest
import numpy as np
import cv2
from PIL import Image
from fastapi import UploadFile, HTTPException

from backend.utils.validation import validate_file, validate_image, validate_chest_xray

def create_test_image(mode="chest_xray"):
    """
    Helper function generating synthetic images for testing validation rules.
    """
    height, width = 256, 256
    
    if mode == "chest_xray":
        # Realistic Chest X-Ray simulation (Grayscale, bilateral lung dark cavities, central mediastinum)
        canvas = np.zeros((height, width), dtype=np.uint8) + 15
        left_lung = np.zeros((height, width), dtype=np.uint8)
        right_lung = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(left_lung, (85, 115), (42, 70), 0, 0, 360, 150, -1)
        cv2.ellipse(right_lung, (170, 115), (42, 70), 0, 0, 360, 150, -1)
        lungs = cv2.add(left_lung, right_lung)
        lungs = cv2.GaussianBlur(lungs, (21, 21), 0)
        canvas = cv2.add(canvas, lungs)
        # Spine mediastinum
        cv2.rectangle(canvas, (120, 40), (136, 220), 85, -1)
        canvas = cv2.GaussianBlur(canvas, (7, 7), 0)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    elif mode == "color_photo":
        # Standard RGB color photograph (cat, landscape, logo)
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[:, :, 0] = 220  # Red
        arr[:, :, 1] = 80   # Green
        arr[:, :, 2] = 40   # Blue
        return Image.fromarray(arr)

    elif mode == "hand_xray":
        # Hand X-Ray (huge black void > 75%, thin bone digits)
        canvas = np.zeros((height, width), dtype=np.uint8) + 5
        # Draw thin finger bones
        for x in [70, 100, 130, 160]:
            cv2.line(canvas, (x, 180), (x, 50), 160, 6)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    elif mode == "ct_scan":
        # Axial Head/Brain CT Scan (circular skull boundary with black corners)
        canvas = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(canvas, (128, 128), 95, 160, 8)  # Skull ring
        cv2.circle(canvas, (128, 128), 90, 80, -1)  # Brain tissue
        rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    elif mode == "abdominal_xray":
        # Abdominal X-Ray (dense stomach/liver soft tissue in upper half, no upper bilateral dark lung fields)
        canvas = np.zeros((height, width), dtype=np.uint8) + 160
        # Pelvis bone structure at bottom
        cv2.ellipse(canvas, (128, 210), (80, 30), 0, 0, 360, 220, -1)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

def test_file_extension_and_mime_validation():
    # Valid PNG
    valid_png = UploadFile(filename="chest_xray.png", content_type="image/png")
    validate_file(valid_png)

    # Valid JPG
    valid_jpg = UploadFile(filename="chest_xray.jpg", content_type="image/jpeg")
    validate_file(valid_jpg)

    # Invalid Extension (PDF)
    invalid_pdf = UploadFile(filename="document.pdf", content_type="application/pdf")
    with pytest.raises(HTTPException) as exc:
        validate_file(invalid_pdf)
    assert exc.value.status_code == 400
    assert "Unsupported file type" in exc.value.detail

    # Invalid Extension (EXE)
    invalid_exe = UploadFile(filename="virus.exe", content_type="application/octet-stream")
    with pytest.raises(HTTPException) as exc:
        validate_file(invalid_exe)
    assert exc.value.status_code == 400

def test_corrupted_image_validation():
    # 0-byte file
    with pytest.raises(HTTPException) as exc:
        validate_image(b"")
    assert exc.value.status_code == 400
    assert "empty" in exc.value.detail

    # Corrupted random bytes
    with pytest.raises(HTTPException) as exc:
        validate_image(b"NOT_AN_IMAGE_FILE_HEADER_DATA_123456789")
    assert exc.value.status_code == 400
    assert "corrupted" in exc.value.detail

def test_chest_xray_validation_cases():
    # 1. Valid Chest X-Ray -> Accepted
    cxr_img = create_test_image("chest_xray")
    is_valid, reason = validate_chest_xray(cxr_img)
    assert is_valid is True
    assert "Valid" in reason

    # 2. Random Color Photo -> Rejected
    color_photo = create_test_image("color_photo")
    is_valid, reason = validate_chest_xray(color_photo)
    assert is_valid is False
    assert "color photo" in reason

    # 3. Hand X-Ray -> Rejected
    hand_xray = create_test_image("hand_xray")
    is_valid, reason = validate_chest_xray(hand_xray)
    assert is_valid is False
    assert "hand X-Ray" in reason or "extremity" in reason

    # 4. CT Scan -> Rejected
    ct_scan = create_test_image("ct_scan")
    is_valid, reason = validate_chest_xray(ct_scan)
    assert is_valid is False
    assert "CT Scan" in reason

    # 5. Abdominal X-Ray -> Rejected
    abdominal_xray = create_test_image("abdominal_xray")
    is_valid, reason = validate_chest_xray(abdominal_xray)
    assert is_valid is False
    assert "Abdominal" in reason or "non-chest" in reason
