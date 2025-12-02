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
    zipped = zip(*features)
    with open(FILE_NAME, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(zipped)


if __name__ == "__main__":
    train_features = []
    train_features.append(convert_img_to_multi_color_IBC(train_benign))
    train_features.append(compute_MC2(train_benign))
    train_features.append(compute_MC4(train_benign))
    train_features.append(compute_B1(train_benign))
    train_features.append(compute_B2(train_benign))
    saveFeatures(train_features)