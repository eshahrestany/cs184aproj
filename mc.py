import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import kagglehub
import saveFeatures
from lesion_mask  import calc_threshold_mask as calc_lm

# Download the dataset from Kaggle
path = kagglehub.dataset_download("hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images")
path = os.path.join(path, "melanoma_cancer_dataset")
print(path)
train_benign = os.path.join(path, f"train/benign")
train_malignant = os.path.join(path, f"train/malignant")
test_benign = os.path.join(path, f"test/benign")
test_malignant = os.path.join(path, f"test/malignant")
#colors are wrong help
COLOR_RANGES = {
    "light brown": [(166, 143, 119), (171, 113, 53)],
    "dark brown":  [(43, 34, 25), (143, 70, 0)],
    "red":         [(92, 67, 67), (255, 0, 0)],
    "black":       [(0, 0, 0), (67, 67, 67)],
    "blue-gray/white": [(74, 103, 133), (255, 255, 255)]
}

def compute_MC1(image):
    CL = []
    detected_colors = []

    for color_name, (low, high) in COLOR_RANGES.items():
        low = np.array(low, dtype=np.uint8)
        high = np.array(high, dtype=np.uint8)

        # Create mask for the color
        mask = cv2.inRange(image, low, high)

        # If at least 1 pixel is detected → color present
        present = mask.any()

        CL.append(1 if present else 0)
        if present:
            detected_colors.append(color_name)
    return CL

def compute_MC2(image,mask_np):
    b, g, r = cv2.split(image)

    Lred   = (r > 150).astype(np.uint8)
    Lblue  = (b > 150).astype(np.uint8)
        # Restrict color masks to the lesion region
        #uses actual mask, not the combined image

    A_red = np.sum(Lred * mask_np)
    A_blue = np.sum(Lblue * mask_np)

    # Avoid division by zero
    if (A_red + A_blue) == 0:
        return 0.0

    MC2 = (A_red - A_blue) / (A_red + A_blue)
    return float(MC2)

def compute_MC4(image):
    b, g, r = cv2.split(image)

    # Brightness threshold based on percentile
    brightness = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thr = np.percentile(brightness, 85)
    bright_pixels = brightness >= thr

    # Grayish pixels: difference between channels is small
    gray_pixels = (np.abs(r - g) <= 20) & \
                (np.abs(r - b) <= 20) & \
                (np.abs(g - b) <= 20)

    # Blue-gray: bright AND grayish OR strong blue
    Lbluegray = ((bright_pixels & gray_pixels) | (b > thr)).astype(np.uint8)

    mask = calc_lm(image)
    mask_np = mask.cpu().numpy().astype(bool) if hasattr(mask, "cpu") else mask.astype(bool)
    lesion_only = image.copy()
    lesion_only[~mask_np] = 0  # zero out background
    inside = Lbluegray & mask_np

    # If any pixel is present → MC4 = 1
    return int(np.any(inside)) 


def convert_img_to_multi_color_IBC(image):
    ''' 
    Return: list of features, [MC1, MC2, MC3, MC4, ...]
    '''
    CL = compute_MC1(image)
    temp = []
    for i in CL:
        temp.append(sum(i))
    return temp
    
if __name__ == "__main__":
    train_features = []
    train_features.append(convert_img_to_multi_color_IBC(train_benign))
    train_features.append(compute_MC2(train_benign))
    train_features.append(compute_MC4(train_benign))
    