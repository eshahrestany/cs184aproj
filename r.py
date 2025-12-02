# r.py

import math
import os
import random
from typing import Optional

import cv2
import kagglehub
import torch
import numpy as np
from matplotlib import image as mpimg

from lesion_mask import calc_threshold_mask, as_tensor_3chw

# -------------------------------------------------------------------------
# Dataset paths
# -------------------------------------------------------------------------

path = kagglehub.dataset_download("hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images")
path = os.path.join(path, "melanoma_cancer_dataset")

train_benign = os.path.join(path, "train/benign")
train_malignant = os.path.join(path, "train/malignant")

all_imgs = []
for d in [train_benign, train_malignant]:
    for fname in os.listdir(d):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            all_imgs.append(os.path.join(d, fname))


# -------------------------------------------------------------------------
# R1 computation
# -------------------------------------------------------------------------

def compute_r1(
    img: torch.Tensor,
    lesion_mask,
    n_angles: int = 64,
    inner_margin_frac: float = 0.25,
    outer_margin_frac: float = 0.25,
) -> float:
    """
    Compute R1: std of edge 'slope' parameter in the red channel.

    Args:
        img:          (3, H, W) RGB image tensor. dtype float or uint8.
        lesion_mask:  mask from calc_threshold_mask; will be converted to (H, W) bool.
        n_angles:     number of angles over [0, 2π) to sample.
        inner_margin_frac: fraction of global lesion radius D used for inside window.
        outer_margin_frac: fraction of global lesion radius D used for outside window.

    Returns:
        R1 as a Python float. Returns float('nan') if too few valid slopes.
    """
    # ---- 1. Normalize / convert inputs ----
    if not isinstance(img, torch.Tensor):
        raise TypeError("img must be a torch.Tensor")

    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"img must have shape (3, H, W), got {tuple(img.shape)}")

    # Convert lesion_mask to torch tensor if needed
    if isinstance(lesion_mask, np.ndarray):
        lesion_mask = torch.from_numpy(lesion_mask)
    elif not isinstance(lesion_mask, torch.Tensor):
        raise TypeError("lesion_mask must be a torch.Tensor or numpy.ndarray")

    # Make mask 2D (H, W) and boolean, handling common 3D cases
    if lesion_mask.ndim == 3:
        # (H, W, C) -> reduce over channels
        if lesion_mask.shape[-1] in (1, 3):
            lesion_mask = lesion_mask.any(dim=-1)
        # (C, H, W) -> reduce over channels
        elif lesion_mask.shape[0] in (1, 3):
            lesion_mask = lesion_mask.any(dim=0)
        else:
            raise ValueError(
                f"lesion_mask has unexpected 3D shape {tuple(lesion_mask.shape)}; "
                "cannot infer (H, W) mask"
            )
    elif lesion_mask.ndim != 2:
        raise ValueError(
            f"lesion_mask must be 2D (H, W) or 3D with a channel dim, got ndim={lesion_mask.ndim}"
        )

    # Now enforce spatial match
    if lesion_mask.shape != img.shape[1:]:
        raise ValueError(
            f"lesion_mask must have shape (H, W) matching img spatial dims. "
            f"Got img spatial={tuple(img.shape[1:])}, mask={tuple(lesion_mask.shape)}"
        )

    lesion_mask = lesion_mask.bool()

    img = img.float()
    if img.max() > 1.5:
        img = img / 255.0  # scale to [0,1] if given as 0–255

    red = img[0]  # (H, W)
    H, W = red.shape

    # ---- 2. Compute lesion centroid and global radius scale D ----
    ys, xs = torch.where(lesion_mask)
    if ys.numel() == 0:
        return float("nan")

    cy = ys.float().mean()
    cx = xs.float().mean()

    distances = torch.sqrt((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2)
    D = distances.max()
    if D <= 0:
        return float("nan")

    max_radius = float(D.item() * 1.5)
    inner_margin = float(inner_margin_frac * D.item())
    outer_margin = float(outer_margin_frac * D.item())

    # ---- 3. Helpers ----

    def border_radius_along_angle(theta: float) -> Optional[float]:
        """
        Walk from the centroid along angle theta until we leave the lesion.
        Returns the radius where we step outside, or None if no clear border.
        """
        step = 0.5  # subpixel step
        r = 0.0
        last_inside = True

        while r < max_radius:
            y = int(round(float(cy) + r * math.sin(theta)))
            x = int(round(float(cx) + r * math.cos(theta)))

            if y < 0 or y >= H or x < 0 or x >= W:
                break

            inside = bool(lesion_mask[y, x])
            if last_inside and not inside:
                return r

            last_inside = inside
            r += step

        return None

    def sample_radial_profile(
        theta: float, r_start: float, r_end: float, num_samples: int = 64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample red-channel intensities along a ray between radii r_start and r_end.

        Returns:
            rs:   1D tensor of radii (N,)
            vals: 1D tensor of intensities (N,)
        """
        if r_end <= r_start:
            return torch.empty(0), torch.empty(0)

        rs = torch.linspace(r_start, r_end, num_samples)
        ys = cy + rs * math.sin(theta)
        xs = cx + rs * math.cos(theta)

        ys_round = ys.round().long()
        xs_round = xs.round().long()

        valid = (ys_round >= 0) & (ys_round < H) & (xs_round >= 0) & (xs_round < W)
        if not valid.any():
            return torch.empty(0), torch.empty(0)

        ys_valid = ys_round[valid]
        xs_valid = xs_round[valid]
        rs_valid = rs[valid]

        vals = red[ys_valid, xs_valid]
        return rs_valid, vals

    # ---- 4. Sweep angles, compute slope per angle ----
    # Older PyTorch doesn't support endpoint=, so do n_angles+1 then drop last
    angles = torch.linspace(0.0, 2.0 * math.pi, steps=n_angles + 1)[:-1]
    slopes = []

    for theta in angles.tolist():
        r_border = border_radius_along_angle(theta)
        if r_border is None:
            continue

        r_start = max(0.0, r_border - inner_margin)
        r_end = min(max_radius, r_border + outer_margin)

        rs, vals = sample_radial_profile(theta, r_start, r_end, num_samples=64)
        if rs.numel() < 10:
            continue

        inside_mask = rs < r_border
        outside_mask = rs >= r_border

        if inside_mask.sum() < 3 or outside_mask.sum() < 3:
            continue

        inside_vals = vals[inside_mask]
        outside_vals = vals[outside_mask]

        delta_I = outside_vals.mean() - inside_vals.mean()
        window_width = (rs.max() - rs.min()).clamp(min=1e-6)
        slope = float(delta_I / window_width)
        slopes.append(slope)

    if len(slopes) >= 3:
        slopes_t = torch.tensor(slopes, dtype=torch.float32)
        R1 = float(slopes_t.std(unbiased=False))
    else:
        # choose your fallback:
        R1 = 0.0  # or slopes_t.std() on however many you have, or some sentinel

    return R1


# -------------------------------------------------------------------------
# Main loop: sample 100 lesions, compute R1
# -------------------------------------------------------------------------

if __name__ == "__main__":
    sample_paths = random.sample(all_imgs, 100)

    for lesion_path in sample_paths:
        print(lesion_path)

        # Load raw image as numpy (H, W, 3)
        img_np = mpimg.imread(lesion_path)

        # Convert to (3, H, W) tensor
        lesion = as_tensor_3chw(img_np)

        # IMPORTANT: pass the original numpy image to calc_threshold_mask
        mask = calc_threshold_mask(img_np)

        r1 = compute_r1(lesion, mask)

        print("R1:", r1)