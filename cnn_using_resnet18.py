import os
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

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

def train():
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
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
            loop.set_postfix(loss=running_loss / (len(loop)))

    print("Training complete.\n")

# 
# 7. EVALUATION
# 

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