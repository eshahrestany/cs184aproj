from b import *
from mc import *
from r import *
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import kagglehub
import csv
from lesion_mask  import calc_threshold_mask as calc_lm
FILE_NAME = "features.csv"
def saveFeatures (features):
    """
        List of lists with the features of every function
    """
    with open(FILE_NAME, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(features)


if __name__ == "__main__":
    train_features = []
    image = cv2.imread(os.path.join(train_benign,os.listdir(train_benign)[0]))
    mask = calc_lm(image)
    mask_np = mask.cpu().numpy().astype(bool) if hasattr(mask, "cpu") else mask.astype(bool)
    lesion_only = image.copy()
    lesion_only[~mask_np] = 0  # zero out background
    train_features.append(convert_img_to_multi_color_IBC(image))
    train_features.append(compute_MC2(image,mask_np))
    train_features.append(compute_MC4(image))
    train_features.append(compute_B1(image,mask_np))
    train_features.append(compute_B2(image,mask_np))
    train_features.append(compute_B10(image,mask_np))
    train_features.append(compute_B11(image,mask_np))
    print(train_features)
    saveFeatures([train_features])