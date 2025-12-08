import os
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


#
# 1. DOWNLOAD DATASET FROM KAGGLE
#

print("Downloading dataset...")
dataset_path = kagglehub.dataset_download(
    "hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images"
)

base_dir = os.path.join(dataset_path, "melanoma_cancer_dataset")

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

print("Dataset downloaded to:", dataset_path)
print("Train directory:", train_dir)
print("Test directory:", test_dir)

#
# 2. TRANSFORMS
#

IMAGE_SIZE = 128

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#
# 3. LOAD DATA
# 

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)

#
# 4. PURE CNN MODEL WITH ADAPTIVE POOLING (trying to mimic resnet in case we need to change for some reason)
#

class PureCNN(nn.Module):
    def __init__(self, num_classes):
        super(PureCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128→64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64→32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32→16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16→8
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)  # → (batch, 256, 1, 1)
        x = torch.flatten(x, 1)  # → (batch, 256)
        x = self.classifier(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PureCNN(num_classes=num_classes).to(device)
print(model)

#
# 5. LOSS AND OPTIMIZER
#

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#
# 6. TRAINING LOOP
#

EPOCHS = 10

train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # TRAIN ACCURACY
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    avg_train_acc = correct_train / total_train * 100

    train_losses.append(avg_train_loss)
    train_accs.append(avg_train_acc)

    # VALIDATION
    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    avg_val_loss = val_running_loss / len(test_loader)
    avg_val_acc = correct_val / total_val * 100

    val_losses.append(avg_val_loss)
    val_accs.append(avg_val_acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | "
        f"Train Acc: {avg_train_acc:.2f}%, Val Acc: {avg_val_acc:.2f}%"
    )

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))



#
# 7. PLOTS
#

# LOSS PLOT
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# ACCURACY PLOT
plt.figure(figsize=(8, 5))
plt.plot(train_accs, label="Training Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()