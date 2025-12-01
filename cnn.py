#training a cnn model to determine melanoma type

import kagglehub
import os
import argparse
from pathlib import Path
import time
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

path = kagglehub.dataset_download("hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to dataset folder that contains `train/` and `test/` subfolders.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_path", type=str, default="best_model.pth")
    p.add_argument("--freeze_backbone", action="store_true",
                 help="If set, freeze backbone and only train classifier head initially.")
    return p.parse_args()



def create_dataloaders(root_dir, img_size=224, batch_size=32, num_workers=4):
    """
    Expects:
      root_dir/train/benign, root_dir/train/malignant
      root_dir/test/benign,  root_dir/test/malignant
    Uses ImageFolder so class->index map will be alphabetical: ['benign','malignant'] -> [0,1]
    """
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_dir = os.path.join(root_dir, "train")
    test_dir  = os.path.join(root_dir, "test")

    train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_ds  = datasets.ImageFolder(test_dir,  transform=test_transforms)

    # compute class counts for class weighting
    counts = np.bincount([y for _, y in train_ds.samples])
    print("Train class counts (index order):", counts)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, counts, train_ds.classes