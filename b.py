import numpy as np
import cv2
import torch.nn.functional as F
import os
import kagglehub
import saveFeatures
from lesion_mask  import calc_threshold_mask as calc_lm

path = kagglehub.dataset_download("hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images")
path = os.path.join(path, "melanoma_cancer_dataset")
print(path)
train_benign = os.path.join(path, f"train/benign")
train_malignant = os.path.join(path, f"train/malignant")
test_benign = os.path.join(path, f"test/benign")
test_malignant = os.path.join(path, f"test/malignant")

def compute_B1(image,mask_np, dtheta_deg=2):
    """
    Compute B1: average absolute angular derivative of radial mean brightness.

    Parameters
    ----------
    img : np.ndarray (HxWx3)
        Dermoscopic image in BGR
    lesion_mask : np.ndarray (HxW), 0/1 mask
        Binary lesion segmentation
    dtheta_deg : float
        Angular step in degrees

    Returns
    -------
    B1 : float
    """

    blue_channel = image[:, :, 0]


    # Compute lesion center = centroid
    ys, xs = np.where(mask_np == 1)
    cy = np.mean(ys)
    cx = np.mean(xs)

    # Get maximum lesion radius
    # distance of farthest lesion pixel from center
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    Rmax = int(np.max(dist))

    # Angular domain
    dtheta = np.deg2rad(dtheta_deg)
    theta_values = np.arange(0, 2*np.pi, dtheta)

    Rm_values = []

    for theta in theta_values:
        # Create radial line coordinates
        xs_line = cx + np.cos(theta) * np.arange(0, Rmax)
        ys_line = cy + np.sin(theta) * np.arange(0, Rmax)

        xs_line = xs_line.astype(int)
        ys_line = ys_line.astype(int)

        # Filter only valid points
        valid = (xs_line >= 0) & (xs_line < image.shape[1]) & \
                (ys_line >= 0) & (ys_line < image.shape[0])

        xs_line = xs_line[valid]
        ys_line = ys_line[valid]

        # Only keep points inside lesion
        inside_mask = mask_np[ys_line, xs_line] == 1
        xs_line = xs_line[inside_mask]
        ys_line = ys_line[inside_mask]

        # If no pixels in this angle, skip
        if len(xs_line) == 0:
            Rm_values.append(0)
            continue

        # Brightness values along radial line
        p_values = blue_channel[ys_line, xs_line]

        # Mean brightness along r1(θ)
        Rm = np.mean(p_values)
        Rm_values.append(Rm)

    # Convert to numpy for derivative
    Rm_values = np.array(Rm_values)

    # Compute absolute derivative
    # dRm/dθ using finite difference
    dRm = np.abs(np.diff(Rm_values, prepend=Rm_values[-1]))

    # B1 = average absolute derivative
    B1 = np.mean(dRm) / dtheta   # normalize by angular step
    return B1

def compute_B2(image, mask_np):
    # Convert image to grayscale brightness
    blue_channel = image[:, :, 0]

    # Compute lesion center = centroid
    ys, xs = np.where(mask_np == 1)
    cy = np.mean(ys)
    cx = np.mean(xs)
    
    # Get maximum lesion radius
    # distance of farthest lesion pixel from center
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    Rmax = int(np.max(dist))

    # Angular domain
    dtheta = np.deg2rad(2)
    theta_values = np.arange(0, 2*np.pi, dtheta)

    Rstd_values = []

    
    for theta in theta_values:
        # Create radial line coordinates
        xs_line = cx + np.cos(theta) * np.arange(0, Rmax)
        ys_line = cy + np.sin(theta) * np.arange(0, Rmax)

        xs_line = xs_line.astype(int)
        ys_line = ys_line.astype(int)

        # Filter only valid points
        valid = (xs_line >= 0) & (xs_line < image.shape[1]) & \
                (ys_line >= 0) & (ys_line < image.shape[0])

        xs_line = xs_line[valid]
        ys_line = ys_line[valid]

        # Only keep points inside lesion
        inside_mask = mask_np[ys_line, xs_line] == 1
        xs_line = xs_line[inside_mask]
        ys_line = ys_line[inside_mask]

        # If no pixels in this angle, skip
        if len(xs_line) == 0:
            Rstd_values.append(0)
            continue

        # Brightness values along radial line
        p_values = blue_channel[ys_line, xs_line]
        Rstd = np.std(p_values, ddof=1)  # unbiased estimator
        Rstd_values.append(Rstd)
    Rstd_values = np.array(Rstd_values)
    B2 = np.var(Rstd_values, ddof=1)
    return B2

def compute_B10(image, mask_np):
    """
    Compute B1: average absolute angular derivative of radial mean brightness.

    Parameters
    ----------
    img : np.ndarray (HxWx3)
        Dermoscopic image in BGR
    lesion_mask : np.ndarray (HxW), 0/1 mask
        Binary lesion segmentation
    dtheta_deg : float
        Angular step in degrees

    Returns
    -------
    B1 : float
    """

    blue_channel = image[:, :, 0]

    # Compute lesion center = centroid
    ys, xs = np.where(mask_np == 1)
    cy = np.mean(ys)
    cx = np.mean(xs)

    # Get maximum lesion radius
    # distance of farthest lesion pixel from center
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    Rmax = int(np.max(dist))

    # Angular domain
    dtheta = np.deg2rad(2)
    theta_values = np.arange(0, 2*np.pi, dtheta)

    Rm_values = []

    for theta in theta_values:
        # Create radial line coordinates
        xs_line = cx + np.cos(theta) * np.arange(0, Rmax)
        ys_line = cy + np.sin(theta) * np.arange(0, Rmax)

        xs_line = xs_line.astype(int)
        ys_line = ys_line.astype(int)

        # Filter only valid points
        valid = (xs_line >= 0) & (xs_line < image.shape[1]) & \
                (ys_line >= 0) & (ys_line < image.shape[0])

        xs_line = xs_line[valid]
        ys_line = ys_line[valid]

        # Only keep points inside lesion
        inside_mask = mask_np[ys_line, xs_line] == 1
        xs_line = xs_line[inside_mask]
        ys_line = ys_line[inside_mask]

        # If no pixels in this angle, skip
        if len(xs_line) == 0:
            Rm_values.append(0)
            continue

        # Brightness values along radial line
        p_values = blue_channel[ys_line, xs_line]

        # Mean brightness along r1(θ)
        Rm = np.mean(p_values)
        Rm_values.append(Rm)

    # Convert to numpy for derivative
    Rm_values = np.array(Rm_values)
    a, b = np.polyfit(theta_values, Rm_values, 1)

    # Normalize by mean brightness (avoid divide-by-zero)
    mean_Rm = np.mean(Rm_values)
    if mean_Rm == 0:
        return 0.0

    B10 = abs(a) / mean_Rm
    return B10
def compute_B11(image, mask_np):
    """
    Compute B1: average absolute angular derivative of radial mean brightness.

    Parameters
    ----------
    img : np.ndarray (HxWx3)
        Dermoscopic image in BGR
    lesion_mask : np.ndarray (HxW), 0/1 mask
        Binary lesion segmentation
    dtheta_deg : float
        Angular step in degrees

    Returns
    -------
    B1 : float
    """

    blue_channel = image[:, :, 0]
    # Compute lesion center = centroid
    ys, xs = np.where(mask_np == 1)
    cy = np.mean(ys)
    cx = np.mean(xs)

    # Get maximum lesion radius
    # distance of farthest lesion pixel from center
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    Rmax = int(np.max(dist))

    # Angular domain
    dtheta = np.deg2rad(2)
    theta_values = np.arange(0, 2*np.pi, dtheta)

    Rm_values = []

    for theta in theta_values:
        # Create radial line coordinates
        xs_line = cx + np.cos(theta) * np.arange(0, Rmax)
        ys_line = cy + np.sin(theta) * np.arange(0, Rmax)

        xs_line = xs_line.astype(int)
        ys_line = ys_line.astype(int)

        # Filter only valid points
        valid = (xs_line >= 0) & (xs_line < image.shape[1]) & \
                (ys_line >= 0) & (ys_line < image.shape[0])

        xs_line = xs_line[valid]
        ys_line = ys_line[valid]

        # Only keep points inside lesion
        inside_mask = mask_np[ys_line, xs_line] == 1
        xs_line = xs_line[inside_mask]
        ys_line = ys_line[inside_mask]

        # If no pixels in this angle, skip
        if len(xs_line) == 0:
            Rm_values.append(0)
            continue

        # Brightness values along radial line
        p_values = blue_channel[ys_line, xs_line]

        # Mean brightness along r1(θ)
        Rm = np.mean(p_values)
        Rm_values.append(Rm)

    # Convert to numpy for derivative
    Rm_values = np.array(Rm_values)
    if len(Rm_values) == 0 or np.mean(Rm_values) == 0:
        return 0.0
    R_range = np.max(Rm_values) - np.min(Rm_values)
    return  R_range / np.mean(Rm_values)

if __name__ == "__main__":
    B_features = [] 
    B_features.append(compute_B10(train_benign))
    B_features.append(compute_B11(train_benign))
    print(B_features)