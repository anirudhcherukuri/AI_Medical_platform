import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
from pathlib import Path

from backend.models.classifier import MedicalChestXRayClassifier, get_transforms
from backend.config import MODEL_SAVE_PATH, NUM_CLASSES, CLASS_NAMES

class MedicalDatasetTrainer:
    """
    Standalone Training & Fine-Tuning Pipeline for Chest X-Ray Diagnostics.
    Supports real medical image datasets (ImageFolder layout) or custom dataset directories.
    """
    def __init__(
        self,
        data_dir: str = None,
        num_classes: int = NUM_CLASSES,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        epochs: int = 5
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_dir = data_dir
        
        self.model = MedicalChestXRayClassifier(num_classes=self.num_classes, pretrained=True)
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train_on_directory(self, train_dir: str, val_dir: str = None):
        """
        Trains the PyTorch DenseNet121 model on a dataset directory formatted as:
        train_dir/
           ├── Normal/
           ├── Pneumonia/
           └── COVID-19/
        """
        print(f"[Trainer] Loading dataset from: {train_dir}")
        
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        
        print(f"[Trainer] Found {len(dataset)} images across {len(dataset.classes)} classes: {dataset.classes}")
        print(f"[Trainer] Starting training on {self.device} for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_loss = running_loss / total
            epoch_acc = (correct / total) * 100.0
            print(f"Epoch [{epoch+1}/{self.epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

        # Save trained model weights
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
        print(f"[Trainer] Model checkpoint successfully saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Medical Image Diagnostic Classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training dataset directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    trainer = MedicalDatasetTrainer(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    trainer.train_on_directory(args.data_dir)
