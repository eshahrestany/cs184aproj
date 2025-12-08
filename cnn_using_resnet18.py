import os
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt


#
# 1. DOWNLOAD DATASET USING KAGGLEHUB
# 

print("Downloading dataset...")
download_path = kagglehub.dataset_download(
    "hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images"
)

dataset_root = os.path.join(download_path, "melanoma_cancer_dataset")

train_dir = os.path.join(dataset_root, "train")
test_dir  = os.path.join(dataset_root, "test")

print("Dataset downloaded to:", download_path)
print("Train directory:", train_dir)
print("Test directory:", test_dir)

# 
# 2. TRANSFORMS
# 

IMG_SIZE = 224
BATCH_SIZE = 32

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 
# 3. LOAD DATA USING IMAGEFOLDER
# 

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset  = datasets.ImageFolder(test_dir, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", train_dataset.classes)

# 
# 4. BUILD CNN MODEL (TRANSFER LEARNING: RESNET18)
# 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

model = model.to(device)

# 
# 5. LOSS AND OPTIMIZER
# 

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 
# 6. TRAINING LOOP
# 

EPOCHS = 5
train_losses = []
val_losses = []
train_accs = []
val_accs = []

def train():
    for epoch in range(EPOCHS):

        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, labels in loop:
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images).view(-1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # TRAIN ACCURACY
            preds = (torch.sigmoid(outputs) >= 0.5).long()
            correct += (preds.cpu() == labels.cpu().long()).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=running_loss / (len(loop)))

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = correct / total

        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        # ---- VALIDATION METRICS ----
        val_loss, val_acc = validate_epoch()
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f} | "
            f"Train Acc: {avg_train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%\n"
        )


# 
# 7. EVALUATION
# 

def validate_epoch():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images).view(-1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = (torch.sigmoid(outputs) >= 0.5).long()
            correct += (preds.cpu() == labels.cpu().long()).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(test_loader)
    avg_acc = correct / total
    return avg_loss, avg_acc


def evaluate():
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = torch.sigmoid(model(images).view(-1))
            predicted = (outputs >= 0.5).long().cpu().numpy()

            preds.extend(predicted)
            trues.extend(labels.cpu().numpy())

    acc = accuracy_score(trues, preds)
    print("\nTest Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(trues, preds, target_names=train_dataset.classes))

# 
# 8. RUN TRAIN + TEST
# 

train()
evaluate()

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
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()
