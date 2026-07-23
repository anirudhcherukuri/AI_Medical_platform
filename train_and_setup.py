import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import datetime

# Ensure root in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import SAMPLES_DIR, MODELS_DIR, MODEL_SAVE_PATH, REPORTS_DIR, UPLOADS_DIR, HEATMAPS_DIR
from backend.models.classifier import MedicalChestXRayClassifier, get_transforms
from backend.database.connection import init_db, SessionLocal
from backend.database import crud
from backend.xai.gradcam import GradCAM
from backend.llm.report_generator import get_report_generator
from backend.utils.pdf_generator import generate_project_technical_report, generate_scan_pdf_report

def create_synthetic_chest_xray(condition: str = "Normal") -> Image.Image:
    """
    Generates realistic synthetic Chest X-Ray image with thoracic rib structures
    and condition-specific lung opacities/consolidations.
    """
    width, height = 512, 512
    # Base background (Dark thoracic cavitiy)
    canvas = np.zeros((height, width), dtype=np.uint8) + 15
    
    # Draw Lung Field Contours (Lighter gray regions)
    left_lung = np.zeros((height, width), dtype=np.uint8)
    right_lung = np.zeros((height, width), dtype=np.uint8)
    
    cv2.ellipse(left_lung, (180, 260), (90, 160), 0, 0, 360, 180, -1)
    cv2.ellipse(right_lung, (332, 260), (90, 160), 0, 0, 360, 180, -1)
    
    lungs = cv2.add(left_lung, right_lung)
    lungs = cv2.GaussianBlur(lungs, (45, 45), 0)
    canvas = cv2.add(canvas, lungs)
    
    # Draw Rib Cage Shadow Patterns
    for y in range(120, 420, 35):
        cv2.ellipse(canvas, (256, y), (190, 20), 0, 0, 180, 40, 6)
        
    # Draw Spine Column
    cv2.rectangle(canvas, (244, 80), (268, 460), 70, -1)
    canvas = cv2.GaussianBlur(canvas, (15, 15), 0)
    
    # Add pathology-specific opacities
    if condition == "Pneumonia":
        # Focal consolidation opacity in right lower lobe
        consolidation = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(consolidation, (330, 310), 65, 160, -1)
        cv2.circle(consolidation, (360, 330), 45, 140, -1)
        consolidation = cv2.GaussianBlur(consolidation, (35, 35), 0)
        canvas = cv2.add(canvas, consolidation)
        
    elif condition == "COVID-19":
        # Bilateral peripheral ground-glass opacities
        covid_opacities = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(covid_opacities, (130, 280), (55, 90), 20, 0, 360, 150, -1)
        cv2.ellipse(covid_opacities, (380, 280), (55, 90), -20, 0, 360, 150, -1)
        covid_opacities = cv2.GaussianBlur(covid_opacities, (41, 41), 0)
        canvas = cv2.add(canvas, covid_opacities)
        
    # Add Gaussian noise for realistic X-Ray quantum mottle texture
    noise = np.random.normal(0, 8, (height, width)).astype(np.uint8)
    canvas = cv2.add(canvas, noise)
    
    # Convert to RGB PIL Image
    rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb_canvas)


def train_and_initialize_system():
    print("=========================================================")
    print(" ADVANCED AI MEDICAL INTELLIGENCE PLATFORM SETUP ")
    print("=========================================================")
    
    # 1. Generate Sample Chest X-Rays
    print("\n[Step 1/5] Generating Synthetic Medical Chest X-Ray Datasets...")
    samples = {
        "normal": create_synthetic_chest_xray("Normal"),
        "pneumonia": create_synthetic_chest_xray("Pneumonia"),
        "covid": create_synthetic_chest_xray("COVID-19")
    }
    
    for name, img in samples.items():
        sample_path = SAMPLES_DIR / f"sample_{name}.jpg"
        img.save(sample_path)
        print(f"  -> Saved {sample_path}")

    # 2. Train/Fine-Tune PyTorch Model Weights with High Accuracy (Mini-Batch DataLoader)
    print("\n[Step 2/5] Training PyTorch DenseNet121 Diagnostic Classifier (15 High-Accuracy Epochs)...")
    import tempfile, shutil
    from torch.utils.data import TensorDataset, DataLoader, random_split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  -> Training device: {device}")
    model = MedicalChestXRayClassifier(num_classes=3, pretrained=True)
    model.to(device)

    # Augmented transform for training variety
    augment_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_transform = get_transforms()

    # Save 150 synthetic images to a temp folder (50 per class)
    print("  -> Generating 150 augmented synthetic radiograph samples...")
    class_map = {"Normal": 0, "Pneumonia": 1, "COVID-19": 2}
    all_tensors, all_labels = [], []
    for c_name, c_idx in class_map.items():
        for i in range(50):
            img = create_synthetic_chest_xray(c_name)
            t = augment_transform(img) if i % 5 != 0 else eval_transform(img)
            all_tensors.append(t)
            all_labels.append(c_idx)

    dataset = TensorDataset(torch.stack(all_tensors), torch.tensor(all_labels, dtype=torch.long))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    print("  -> Executing 15 high-precision mini-batch epochs...\n")
    for epoch in range(15):
        # --- TRAIN ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            _, preds = out.max(1)
            train_correct += preds.eq(yb).sum().item()
            train_total += yb.size(0)
        scheduler.step()

        # --- VALIDATE ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item() * xb.size(0)
                _, preds = out.max(1)
                val_correct += preds.eq(yb).sum().item()
                val_total += yb.size(0)

        ep_train_loss = train_loss / train_total
        ep_train_acc  = (train_correct / train_total) * 100.0
        ep_val_loss   = val_loss / val_total
        ep_val_acc    = (val_correct / val_total) * 100.0

        marker = " *** BEST ***" if ep_val_acc >= best_val_acc else ""
        print(f"     Epoch [{epoch+1:02d}/15] | "
              f"Train Loss: {ep_train_loss:.4f} | Train Acc: {ep_train_acc:.1f}% | "
              f"Val Loss: {ep_val_loss:.4f} | Val Acc: {ep_val_acc:.1f}%{marker}")

        if ep_val_acc >= best_val_acc:
            best_val_acc = ep_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\n  -> Training Complete! Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"  -> Best model checkpoint saved to: {MODEL_SAVE_PATH}")

    # 3. Database Initialization & Seeding
    print("\n[Step 3/5] Initializing Database & Audit Log Records...")
    init_db()
    db = SessionLocal()
    
    # Generate seed predictions for sample images
    model.eval()
    gradcam = GradCAM(model, model.get_target_layer())
    llm_gen = get_report_generator()
    
    seed_cases = [
        {"file": SAMPLES_DIR / "sample_normal.jpg", "name": "Arthur Pendelton", "pid": "P-NOR-101", "age": 38, "gender": "Male"},
        {"file": SAMPLES_DIR / "sample_pneumonia.jpg", "name": "Sarah Connor", "pid": "P-PNE-304", "age": 61, "gender": "Female"},
        {"file": SAMPLES_DIR / "sample_covid.jpg", "name": "David Martinez", "pid": "P-COV-882", "age": 49, "gender": "Male"}
    ]
    
    for case in seed_cases:
        pil_img = Image.open(case["file"]).convert("RGB")
        tensor_img = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor_img)
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            
        c_names = ["Normal", "Pneumonia", "COVID-19"]
        pred_class = c_names[pred_idx]
        prob_dict = {c_names[i]: float(probs[i]) for i in range(3)}
        
        heatmap_2d, spatial_metrics = gradcam.generate_heatmap(tensor_img, pred_idx)
        prefix = f"seed_{case['pid'].lower()}"
        vis_paths = gradcam.save_visualization(pil_img, heatmap_2d, prefix)
        
        patient_info = {
            "patient_id": case["pid"],
            "patient_name": case["name"],
            "age": case["age"],
            "gender": case["gender"],
            "scan_type": "Chest X-Ray PA View"
        }
        
        report_data = llm_gen.generate_report(pred_class, conf, prob_dict, spatial_metrics, patient_info)
        
        # Save image copy to uploads
        dest_upload_path = UPLOADS_DIR / f"{prefix}_orig.jpg"
        pil_img.save(dest_upload_path)
        
        rec = crud.create_scan_record(
            db=db,
            patient_id=case["pid"],
            patient_name=case["name"],
            age=case["age"],
            gender=case["gender"],
            scan_type="Chest X-Ray PA View",
            original_image_path=f"/storage/uploads/{prefix}_orig.jpg",
            heatmap_image_path=f"/storage/heatmaps/{vis_paths['heatmap_filename']}",
            overlay_image_path=f"/storage/heatmaps/{vis_paths['overlay_filename']}",
            predicted_class=pred_class,
            confidence=conf,
            probabilities=prob_dict,
            spatial_metrics=spatial_metrics,
            report_text=report_data["full_report_text"],
            risk_level=report_data["risk_level"],
            report_source=report_data["source"]
        )
        
        # Generate sample clinical PDF
        scan_dict = rec.to_dict()
        scan_dict["original_image_path"] = str(dest_upload_path)
        scan_dict["overlay_image_path"] = vis_paths["overlay_path"]
        generate_scan_pdf_report(scan_dict)
        print(f"  -> Seeded patient audit record: {case['name']} ({pred_class})")

    db.close()

    # 4. Generate Project Technical Report PDF
    print("\n[Step 4/5] Generating Project Submission Technical Report PDF...")
    proj_report_path = REPORTS_DIR / "Advanced_AI_Medical_Intelligence_Platform_Report.pdf"
    generate_project_technical_report(str(proj_report_path))

    # 5. Verification Complete
    print("\n[Step 5/5] Setup & Initialization Complete!")
    print("=========================================================")
    print(" To launch the server, run: python run.py")
    print(" Web UI will be available at: http://localhost:8000/")
    print(" OpenAPI Docs available at: http://localhost:8000/docs")
    print("=========================================================")

if __name__ == "__main__":
    train_and_initialize_system()
