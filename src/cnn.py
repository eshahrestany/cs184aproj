# cnn.py

import os
import random
from typing import Tuple

import kagglehub
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# Model definition
# ============================================================

class MelanomaCNN(nn.Module):
    """
    Simple CNN for benign vs malignant classification.

    Input: RGB image resized to 128x128
    Output: logits over 2 classes (0=benign, 1=malignant)
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            # (3, 128, 128) -> (32, 64, 64)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # (32, 64, 64) -> (64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # (64, 32, 32) -> (128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # (128, 16, 16) -> (256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================
# Training and evaluation helpers
# ============================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 1e-4,
) -> None:
    """
    Train the CNN on the training DataLoader.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch:02d}/{num_epochs} - "
              f"Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on a DataLoader.

    Returns:
        (accuracy, avg_loss)
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0

    return acc, avg_loss


# ============================================================
# Main script: download data, train, save, eval
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Download Kaggle dataset and set up directories
    # --------------------------------------------------------
    # Dataset: hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
    dataset_root = kagglehub.dataset_download(
        "hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images"
    )
    dataset_root = os.path.join(dataset_root, "melanoma_cancer_dataset")

    train_dir = os.path.join(dataset_root, "train")
    test_dir = os.path.join(dataset_root, "test")

    if not os.path.isdir(train_dir):
        raise RuntimeError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise RuntimeError(f"Test directory not found: {test_dir}")

    # --------------------------------------------------------
    # Transforms and Datasets
    # --------------------------------------------------------
    img_size = 128

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # normalize to [-1,1] roughly
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    print(f"Classes:       {train_dataset.classes}")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------
    batch_size = 32
    num_workers = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model, training, saving
    # --------------------------------------------------------
    model = MelanomaCNN(num_classes=2)
    print(model)

    num_epochs = 10
    train(model, train_loader, device, num_epochs=num_epochs, lr=1e-4)

    weights_path = "models/cnn_melanoma.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved model weights to: {weights_path}")

    # --------------------------------------------------------
    # Evaluation on test set
    # --------------------------------------------------------
    # (Optionally reload just to prove it works)
    loaded_model = MelanomaCNN(num_classes=2)
    loaded_model.load_state_dict(torch.load(weights_path, map_location=device))

    test_acc, test_loss = evaluate(loaded_model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}  |  Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
