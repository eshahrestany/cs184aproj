import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import kagglehub

# Download the dataset from Kaggle
path = kagglehub.dataset_download("hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images")
path = os.path.join(path, "melanoma_cancer_dataset")
print(path)
train_benign = os.path.join(path, f"train/benign")
train_malignant = os.path.join(path, f"train/malignant")
test_benign = os.path.join(path, f"test/benign")
test_malignant = os.path.join(path, f"test/malignant")
COLOR_RANGES = {
    "light brown": [(10, 20, 50), (22, 255, 255)],
    "dark brown":  [(0, 20, 20), (20, 255, 150)],
    "red":         [(0, 120, 70), (10, 255, 255)],
    "black":       [(0, 0, 0), (180, 255, 40)],
    "blue-gray/white": [(85, 10, 80), (110, 60, 255)]
}

def compute_MC1(path_to_data):
    total_CL = []
    for i in os.listdir(path_to_data):
        image = cv2.imread(os.path.join(path_to_data,i))
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
        total_CL.append(CL)
    return total_CL

def compute_MC2(path_to_data, lesion_mask):
    #uses masked data, may need to use actual mask
    total_MC2 = []
    for i in os.listdir(path_to_data):
        image = cv2.imread(os.path.join(path_to_data,i))
        b, g, r = cv2.split(image)

        Lred   = (r > 150).astype(np.uint8)
        Lgreen = (g > 150).astype(np.uint8)
        Lblue  = (b > 150).astype(np.uint8)
         # Restrict color masks to the lesion region
         #uses actual mask, not the combined image
        A_red = np.sum(Lred * lesion_mask)
        A_blue = np.sum(Lblue * lesion_mask)

        # Avoid division by zero
        if (A_red + A_blue) == 0:
            total_MC2.append(0.0)

        MC2 = (A_red - A_blue) / (A_red + A_blue)
        total_MC2.append(float(MC2))
    return total_MC2

def compute_MC4(path_to_data, lesion_mask):
    total_MC4 = []
    for i in os.listdir(path_to_data):
        image = cv2.imread(os.path.join(path_to_data,i))
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

        inside = Lbluegray & lesion_mask

        # If any pixel is present → MC4 = 1
        total_MC4.append(int(np.any(inside)))
    return total_MC4

def convert_img_to_multi_color_IBC():
    ''' 
    Return: list of list of features, [MC1, MC2, MC3, MC4, ...]
    '''
    ans= []
    CL = compute_MC1()
    temp = []
    for i in CL:
        temp.append(sum(i))
    ans.append(temp)
    

 