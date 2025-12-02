import math
import os
import random
from typing import Optional

import kagglehub
import numpy as np
import torch
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


def compute_R1(
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
        return 0.0

    cy = ys.float().mean()
    cx = xs.float().mean()

    distances = torch.sqrt((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2)
    D = distances.max()
    if D <= 0:
        return 0.0

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


def compute_R2(
    img: torch.Tensor,
    lesion_mask,
    n_angles: int = 64,
) -> float:
    """
    Compute R2: radial asymmetry of the lesion.
    Defined as std(radius(θ)) / mean(radius(θ)) across sampled angles.

    Args:
        img: (3, H, W) RGB image tensor
        lesion_mask: 2D mask (H, W) from calc_threshold_mask
        n_angles: number of radial samples

    Returns:
        R2 as a Python float. NaN if insufficient valid radii.
    """
    # ----------------------------------------------
    # Validate + normalize inputs (same as R1)
    # ----------------------------------------------
    if isinstance(lesion_mask, np.ndarray):
        lesion_mask = torch.from_numpy(lesion_mask)
    elif not isinstance(lesion_mask, torch.Tensor):
        raise TypeError("lesion_mask must be torch.Tensor or numpy.ndarray")

    if lesion_mask.ndim == 3:
        if lesion_mask.shape[-1] in (1, 3):
            lesion_mask = lesion_mask.any(dim=-1)
        elif lesion_mask.shape[0] in (1, 3):
            lesion_mask = lesion_mask.any(dim=0)
        else:
            raise ValueError(
                f"Unexpected mask shape {tuple(lesion_mask.shape)} for compute_R2"
            )
    elif lesion_mask.ndim != 2:
        raise ValueError("lesion_mask must be 2D")

    lesion_mask = lesion_mask.bool()

    H, W = lesion_mask.shape

    ys, xs = torch.where(lesion_mask)
    if ys.numel() == 0:
        return 0

    cy = ys.float().mean()
    cx = xs.float().mean()

    # Maximum meaningful radius = farthest lesion pixel distance
    distances = torch.sqrt((ys.float() - cy)**2 + (xs.float() - cx)**2)
    D = distances.max().item()
    if D <= 0:
        return 0.0

    max_radius = D * 1.5

    # ----------------------------------------------
    # Helper: find border radius along angle
    # (copied from compute_R1 so behavior is identical)
    # ----------------------------------------------
    def border_radius(theta: float) -> Optional[float]:
        step = 0.5
        last_inside = True
        r = 0.0

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

    # ----------------------------------------------
    # Sweep angles and collect radii
    # ----------------------------------------------
    angles = torch.linspace(0, 2*math.pi, steps=n_angles+1)[:-1]
    radii = []

    for theta in angles.tolist():
        r = border_radius(theta)
        if r is not None:
            radii.append(r)

    if len(radii) < 5:
        return 0.0

    radii_t = torch.tensor(radii, dtype=torch.float32)

    mean_r = radii_t.mean().item()
    std_r  = radii_t.std(unbiased=False).item()

    if mean_r <= 1e-6:
        return 0.0

    R2 = std_r / mean_r
    return float(R2)


def compute_R3(
    img: torch.Tensor,
    lesion_mask,
) -> float:
    """
    R3: color heterogeneity inside the lesion.
    Implemented as std/mean of red-channel intensities over the lesion mask.

    Returns 0.0 on any failure / degenerate case.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError("img must be a torch.Tensor")
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"img must have shape (3, H, W), got {tuple(img.shape)}")

    # normalize mask
    if isinstance(lesion_mask, np.ndarray):
        lesion_mask = torch.from_numpy(lesion_mask)
    elif not isinstance(lesion_mask, torch.Tensor):
        raise TypeError("lesion_mask must be torch.Tensor or numpy.ndarray")

    if lesion_mask.ndim == 3:
        if lesion_mask.shape[-1] in (1, 3):
            lesion_mask = lesion_mask.any(dim=-1)
        elif lesion_mask.shape[0] in (1, 3):
            lesion_mask = lesion_mask.any(dim=0)
        else:
            raise ValueError(
                f"Unexpected mask shape {tuple(lesion_mask.shape)} for compute_R3"
            )
    elif lesion_mask.ndim != 2:
        raise ValueError("lesion_mask must be 2D")

    if lesion_mask.shape != img.shape[1:]:
        raise ValueError(
            f"lesion_mask must have shape (H, W) matching img spatial dims. "
            f"Got img spatial={tuple(img.shape[1:])}, mask={tuple(lesion_mask.shape)}"
        )

    lesion_mask = lesion_mask.bool()

    img = img.float()
    if img.max() > 1.5:
        img = img / 255.0

    red = img[0]  # (H, W)
    inside_vals = red[lesion_mask]

    if inside_vals.numel() < 10:
        return 0.0

    mean_I = inside_vals.mean().item()
    std_I = inside_vals.std(unbiased=False).item()

    if mean_I <= 1e-6:
        return 0.0

    R3 = std_I / mean_I
    return float(R3)


def compute_R4(
    img: torch.Tensor,
    lesion_mask,
) -> float:
    """
    R4: border irregularity.
    Based on shape circularity:

        circularity = 4π * area / perimeter^2
        R4 = 1 - circularity

    where:
        area     = number of lesion pixels
        perimeter = number of boundary pixels (4-connected)

    Returns 0.0 on any failure / degenerate case.
    """
    # img is unused but kept for API symmetry with R1/R2/R3
    if isinstance(lesion_mask, np.ndarray):
        lesion_mask = torch.from_numpy(lesion_mask)
    elif not isinstance(lesion_mask, torch.Tensor):
        raise TypeError("lesion_mask must be torch.Tensor or numpy.ndarray")

    if lesion_mask.ndim == 3:
        if lesion_mask.shape[-1] in (1, 3):
            lesion_mask = lesion_mask.any(dim=-1)
        elif lesion_mask.shape[0] in (1, 3):
            lesion_mask = lesion_mask.any(dim=0)
        else:
            raise ValueError(
                f"Unexpected mask shape {tuple(lesion_mask.shape)} for compute_R4"
            )
    elif lesion_mask.ndim != 2:
        raise ValueError("lesion_mask must be 2D")

    lesion_mask = lesion_mask.bool()
    H, W = lesion_mask.shape

    ys, xs = torch.where(lesion_mask)
    area = ys.numel()
    if area == 0:
        return 0.0

    # perimeter: count pixels that have at least one 4-neighbor outside lesion
    perimeter = 0
    ys_np = ys.cpu().numpy()
    xs_np = xs.cpu().numpy()

    for y, x in zip(ys_np, xs_np):
        y = int(y)
        x = int(x)
        # check 4-neighborhood
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W or not lesion_mask[ny, nx]:
                perimeter += 1
                break

    if perimeter <= 0:
        return 0.0

    area_f = float(area)
    perim_f = float(perimeter)

    circularity = 4.0 * math.pi * area_f / (perim_f * perim_f)
    # clamp a bit for numeric stability
    circularity = max(0.0, min(1.0, circularity))

    R4 = 1.0 - circularity
    return float(R4)


def compute_R5(
    img: torch.Tensor,
    lesion_mask,
    n_angles: int = 64,
    inner_margin_frac: float = 0.25,
    outer_margin_frac: float = 0.25,
) -> float:
    """
    R5 – Mean edge slope (convergent fits)

    What it measures:
        Typical sharpness of the lesion border transitions in the red channel.

    Implementation:
        Uses the same edge-slope procedure as compute_R1, but returns the
        mean slope over all angles where a valid slope could be computed.

    Returns:
        R5 as Python float. Returns 0.0 if no valid slopes are found.
    """
    # ---- 1. Normalize / convert inputs (same logic as compute_R1) ----
    if not isinstance(img, torch.Tensor):
        raise TypeError("img must be a torch.Tensor")

    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"img must have shape (3, H, W), got {tuple(img.shape)}")

    if isinstance(lesion_mask, np.ndarray):
        lesion_mask = torch.from_numpy(lesion_mask)
    elif not isinstance(lesion_mask, torch.Tensor):
        raise TypeError("lesion_mask must be a torch.Tensor or numpy.ndarray")

    if lesion_mask.ndim == 3:
        if lesion_mask.shape[-1] in (1, 3):
            lesion_mask = lesion_mask.any(dim=-1)
        elif lesion_mask.shape[0] in (1, 3):
            lesion_mask = lesion_mask.any(dim=0)
        else:
            raise ValueError(
                f"lesion_mask has unexpected 3D shape {tuple(lesion_mask.shape)}; "
                "cannot infer (H, W) mask"
            )
    elif lesion_mask.ndim != 2:
        raise ValueError("lesion_mask must be 2D (H, W) or 3D with a channel dim")

    if lesion_mask.shape != img.shape[1:]:
        raise ValueError(
            f"lesion_mask must have shape (H, W) matching img spatial dims. "
            f"Got img spatial={tuple(img.shape[1:])}, mask={tuple(lesion_mask.shape)}"
        )

    lesion_mask = lesion_mask.bool()

    img = img.float()
    if img.max() > 1.5:
        img = img / 255.0

    red = img[0]  # (H, W)
    H, W = red.shape

    # ---- 2. Centroid + radius scale ----
    ys, xs = torch.where(lesion_mask)
    if ys.numel() == 0:
        return 0.0

    cy = ys.float().mean()
    cx = xs.float().mean()

    distances = torch.sqrt((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2)
    D = distances.max()
    if D <= 0:
        return 0.0

    max_radius = float(D.item() * 1.5)
    inner_margin = float(inner_margin_frac * D.item())
    outer_margin = float(outer_margin_frac * D.item())

    # ---- 3. Helpers (same structure as in compute_R1) ----
    def border_radius_along_angle(theta: float) -> Optional[float]:
        """
        Walk from centroid along angle theta until we leave lesion.
        Return radius where we step outside, or None if no clear border.
        """
        step = 0.5
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
        Sample red intensities along ray between r_start and r_end.
        Returns (rs, vals).
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

    if len(slopes) == 0:
        return 0.0

    slopes_t = torch.tensor(slopes, dtype=torch.float32)
    R5 = float(slopes_t.mean())
    return R5


def compute_R6(
    img: torch.Tensor,
    lesion_mask,
    n_angles: int = 64,
) -> float:
    """
    R6 – Coefficient of variation of radius

    What it measures:
        Shape asymmetry of the lesion silhouette: how much the radius from
        centroid to border varies across angles.

    Formula (per article):
        R6 = std(R_red(θ)) / mean(R_red(θ))

    Returns:
        R6 as Python float. Returns 0.0 if no valid radii or degenerate mean.
    """
    # ---- 1. Normalize / convert mask ----
    if isinstance(lesion_mask, np.ndarray):
        lesion_mask = torch.from_numpy(lesion_mask)
    elif not isinstance(lesion_mask, torch.Tensor):
        raise TypeError("lesion_mask must be torch.Tensor or numpy.ndarray")

    if lesion_mask.ndim == 3:
        if lesion_mask.shape[-1] in (1, 3):
            lesion_mask = lesion_mask.any(dim=-1)
        elif lesion_mask.shape[0] in (1, 3):
            lesion_mask = lesion_mask.any(dim=0)
        else:
            raise ValueError(
                f"Unexpected mask shape {tuple(lesion_mask.shape)} for compute_R6"
            )
    elif lesion_mask.ndim != 2:
        raise ValueError("lesion_mask must be 2D")

    lesion_mask = lesion_mask.bool()
    H, W = lesion_mask.shape

    ys, xs = torch.where(lesion_mask)
    if ys.numel() == 0:
        return 0.0

    cy = ys.float().mean()
    cx = xs.float().mean()

    distances = torch.sqrt((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2)
    D = distances.max().item()
    if D <= 0:
        return 0.0

    max_radius = D * 1.5

    # ---- 2. Border radius along angle (same idea as R1/R5) ----
    def border_radius(theta: float) -> Optional[float]:
        step = 0.5
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

    # ---- 3. Sweep angles and collect radii ----
    angles = torch.linspace(0.0, 2.0 * math.pi, steps=n_angles + 1)[:-1]
    radii = []

    for theta in angles.tolist():
        r = border_radius(theta)
        if r is not None:
            radii.append(r)

    if len(radii) < 3:
        return 0.0

    radii_t = torch.tensor(radii, dtype=torch.float32)
    mean_r = radii_t.mean().item()
    std_r = radii_t.std(unbiased=False).item()

    if mean_r <= 1e-6:
        return 0.0

    R6 = std_r / mean_r
    return float(R6)


def compute_R7(
    img: torch.Tensor,
    lesion_mask,
    n_angle_bins: int = 36,
) -> float:
    """
    R7 – Range of branch counts per angle

    What it measures (per article):
        How unevenly pigmented network branches are distributed angularly.

    Implementation approximation:
        1. Use red channel inside lesion.
        2. Threshold to extract a darker "network" subset.
        3. Skeletonize this subset.
        4. Find branch points (skeleton pixels with >= 3 neighbors).
        5. Bin branch points by polar angle around lesion centroid.
        6. R7 = max_theta N_branch(theta) - min_theta N_branch(theta).

    Returns:
        R7 as float, 0.0 if no usable branches or on failure.
    """
    # img checks
    if not isinstance(img, torch.Tensor):
        raise TypeError("img must be a torch.Tensor")
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"img must have shape (3, H, W), got {tuple(img.shape)}")

    # mask normalization
    if isinstance(lesion_mask, np.ndarray):
        lesion_mask = torch.from_numpy(lesion_mask)
    elif not isinstance(lesion_mask, torch.Tensor):
        raise TypeError("lesion_mask must be torch.Tensor or numpy.ndarray")

    if lesion_mask.ndim == 3:
        if lesion_mask.shape[-1] in (1, 3):
            lesion_mask = lesion_mask.any(dim=-1)
        elif lesion_mask.shape[0] in (1, 3):
            lesion_mask = lesion_mask.any(dim=0)
        else:
            raise ValueError(
                f"Unexpected mask shape {tuple(lesion_mask.shape)} for compute_R7"
            )
    elif lesion_mask.ndim != 2:
        raise ValueError("lesion_mask must be 2D")

    if lesion_mask.shape != img.shape[1:]:
        raise ValueError(
            f"lesion_mask must have shape (H, W) matching img spatial dims. "
            f"Got img spatial={tuple(img.shape[1:])}, mask={tuple(lesion_mask.shape)}"
        )

    lesion_mask = lesion_mask.bool()
    H, W = lesion_mask.shape

    ys, xs = torch.where(lesion_mask)
    if ys.numel() == 0:
        return 0.0

    cy = ys.float().mean().item()
    cx = xs.float().mean().item()

    # red channel normalized
    img = img.float()
    if img.max() > 1.5:
        img = img / 255.0
    red = img[0]

    # approximate pigmented network: darker than mean - 0.5*std inside lesion
    inside_vals = red[lesion_mask]
    if inside_vals.numel() < 20:
        return 0.0

    mean_I = inside_vals.mean().item()
    std_I = inside_vals.std(unbiased=False).item()
    thresh = mean_I - 0.5 * std_I

    network_mask = (red <= thresh) & lesion_mask

    # need skeletonization
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        # If skimage is missing, degrade gracefully
        # print("Warning: scikit-image not installed, compute_R7 returns 0.0")
        return 0.0

    network_np = network_mask.cpu().numpy().astype(bool)
    skel = skeletonize(network_np)

    # find branch points: skeleton pixels with >= 3 neighbors in 8-connectivity
    skel_y, skel_x = np.where(skel)
    if skel_y.size == 0:
        return 0.0

    H_np, W_np = skel.shape
    branch_angles = []

    for y, x in zip(skel_y, skel_x):
        # count 8-connected neighbors
        neighbors = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H_np and 0 <= nx < W_np and skel[ny, nx]:
                    neighbors += 1
        if neighbors >= 3:
            # branch point: record its angle
            dy = float(y) - cy
            dx = float(x) - cx
            theta = math.atan2(dy, dx)
            if theta < 0:
                theta += 2.0 * math.pi
            branch_angles.append(theta)

    if len(branch_angles) == 0:
        return 0.0

    branch_angles = np.array(branch_angles, dtype=np.float32)

    # histogram into angular bins
    bins = np.linspace(0.0, 2.0 * math.pi, num=n_angle_bins + 1)
    counts, _ = np.histogram(branch_angles, bins=bins)

    max_c = int(counts.max())
    min_c = int(counts.min())
    R7 = float(max_c - min_c)
    return R7


def compute_R8(
    img: torch.Tensor,
    lesion_mask,
    n_angles: int = 64,
    n_radial_samples: int = 64,
) -> float:
    """
    R8 – Range of radial brightness std

    What it measures (per article):
        Spread in radial brightness variability across angles.

    Implementation:
        For each angle θ:
            - Sample red-channel intensities along a ray from center outward.
            - Restrict to samples inside the lesion.
            - Compute std of those intensities: Rstd(θ).
        Then:
            R8 = max_θ Rstd(θ) - min_θ Rstd(θ).

    Returns:
        R8 as float, 0.0 if insufficient data.
    """
    # img checks
    if not isinstance(img, torch.Tensor):
        raise TypeError("img must be a torch.Tensor")
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"img must have shape (3, H, W), got {tuple(img.shape)}")

    # mask normalization
    if isinstance(lesion_mask, np.ndarray):
        lesion_mask = torch.from_numpy(lesion_mask)
    elif not isinstance(lesion_mask, torch.Tensor):
        raise TypeError("lesion_mask must be torch.Tensor or numpy.ndarray")

    if lesion_mask.ndim == 3:
        if lesion_mask.shape[-1] in (1, 3):
            lesion_mask = lesion_mask.any(dim=-1)
        elif lesion_mask.shape[0] in (1, 3):
            lesion_mask = lesion_mask.any(dim=0)
        else:
            raise ValueError(
                f"Unexpected mask shape {tuple(lesion_mask.shape)} for compute_R8"
            )
    elif lesion_mask.ndim != 2:
        raise ValueError("lesion_mask must be 2D")

    if lesion_mask.shape != img.shape[1:]:
        raise ValueError(
            f"lesion_mask must have shape (H, W) matching img spatial dims. "
            f"Got img spatial={tuple(img.shape[1:])}, mask={tuple(lesion_mask.shape)}"
        )

    lesion_mask = lesion_mask.bool()
    H, W = lesion_mask.shape

    ys, xs = torch.where(lesion_mask)
    if ys.numel() == 0:
        return 0.0

    cy = ys.float().mean()
    cx = xs.float().mean()

    distances = torch.sqrt((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2)
    D = distances.max().item()
    if D <= 0:
        return 0.0

    max_radius = D * 1.5

    img = img.float()
    if img.max() > 1.5:
        img = img / 255.0
    red = img[0]  # (H, W)

    # helper to sample ray for a given angle
    def radial_intensities(theta: float) -> torch.Tensor:
        rs = torch.linspace(0.0, max_radius, steps=n_radial_samples)
        ys_f = cy + rs * math.sin(theta)
        xs_f = cx + rs * math.cos(theta)

        ys_i = ys_f.round().long()
        xs_i = xs_f.round().long()

        valid = (ys_i >= 0) & (ys_i < H) & (xs_i >= 0) & (xs_i < W)
        if not valid.any():
            return torch.empty(0)

        ys_i = ys_i[valid]
        xs_i = xs_i[valid]

        # only keep samples inside lesion
        inside = lesion_mask[ys_i, xs_i]
        if not inside.any():
            return torch.empty(0)

        ys_i = ys_i[inside]
        xs_i = xs_i[inside]

        vals = red[ys_i, xs_i]
        return vals

    # sweep angles and collect Rstd(θ)
    angles = torch.linspace(0.0, 2.0 * math.pi, steps=n_angles + 1)[:-1]
    radial_stds = []

    for theta in angles.tolist():
        vals = radial_intensities(theta)
        if vals.numel() < 5:
            continue
        std_val = vals.std(unbiased=False).item()
        radial_stds.append(std_val)

    if len(radial_stds) == 0:
        return 0.0

    radial_stds_t = torch.tensor(radial_stds, dtype=torch.float32)
    R8 = float(radial_stds_t.max().item() - radial_stds_t.min().item())
    return R8


def compute_R9(
    img: torch.Tensor,
    lesion_mask,
) -> float:
    """
    R9 – Coefficient of variation of intralesional brightness

    What it measures (per article):
        Overall heterogeneity of pixel brightness inside the lesion.

    Implementation:
        Use red-channel intensities inside the lesion:
            P_lesion = { I_red(y,x) : (y,x) in lesion }
        R9 = std(P_lesion) / mean(P_lesion)

    Returns:
        R9 as float. Returns 0.0 if too few pixels or degenerate mean.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError("img must be a torch.Tensor")
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"img must have shape (3, H, W), got {tuple(img.shape)}")

    # normalize mask
    if isinstance(lesion_mask, np.ndarray):
        lesion_mask = torch.from_numpy(lesion_mask)
    elif not isinstance(lesion_mask, torch.Tensor):
        raise TypeError("lesion_mask must be torch.Tensor or numpy.ndarray")

    if lesion_mask.ndim == 3:
        if lesion_mask.shape[-1] in (1, 3):
            lesion_mask = lesion_mask.any(dim=-1)
        elif lesion_mask.shape[0] in (1, 3):
            lesion_mask = lesion_mask.any(dim=0)
        else:
            raise ValueError(
                f"Unexpected mask shape {tuple(lesion_mask.shape)} for compute_R9"
            )
    elif lesion_mask.ndim != 2:
        raise ValueError("lesion_mask must be 2D")

    if lesion_mask.shape != img.shape[1:]:
        raise ValueError(
            f"lesion_mask must have shape (H, W) matching img spatial dims. "
            f"Got img spatial={tuple(img.shape[1:])}, mask={tuple(lesion_mask.shape)}"
        )

    lesion_mask = lesion_mask.bool()

    img = img.float()
    if img.max() > 1.5:
        img = img / 255.0

    red = img[0]  # (H, W)
    vals = red[lesion_mask]

    if vals.numel() < 10:
        return 0.0

    mean_I = vals.mean().item()
    std_I = vals.std(unbiased=False).item()

    if mean_I <= 1e-6:
        return 0.0

    R9 = std_I / mean_I
    return float(R9)


def compute_R10(
    img: torch.Tensor,
    lesion_mask,
    n_angles: int = 64,
    inner_margin_frac: float = 0.25,
    outer_margin_frac: float = 0.25,
    n_samples: int = 64,
    n_bins: int = 10,
) -> float:
    """
    R10 – Mode of edge-fit error (convergent fits)

    What it measures (per article):
        Most frequent squared edge-fit error among convergent fits, i.e.,
        how typically "bad" the edge model fit is around the lesion.

    Approximate implementation:
        For each angle θ:
          1. Find border radius r_border (centroid -> outside lesion).
          2. Sample red-channel profile in [r_border - inner_margin, r_border + outer_margin].
          3. Fit a simple linear model I(r) ~ a*r + b on that window.
          4. Compute SSE(θ) = sum_r (I(r) - (a*r + b))^2.
        Then:
          R10 = mode of SSE(θ) over all angles with valid fits.

    Returns:
        R10 as float. Returns 0.0 if no valid fits.
    """
    # ---- 1. Basic checks and normalization ----
    if not isinstance(img, torch.Tensor):
        raise TypeError("img must be a torch.Tensor")
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"img must have shape (3, H, W), got {tuple(img.shape)}")

    if isinstance(lesion_mask, np.ndarray):
        lesion_mask = torch.from_numpy(lesion_mask)
    elif not isinstance(lesion_mask, torch.Tensor):
        raise TypeError("lesion_mask must be torch.Tensor or numpy.ndarray")

    if lesion_mask.ndim == 3:
        if lesion_mask.shape[-1] in (1, 3):
            lesion_mask = lesion_mask.any(dim=-1)
        elif lesion_mask.shape[0] in (1, 3):
            lesion_mask = lesion_mask.any(dim=0)
        else:
            raise ValueError(
                f"Unexpected mask shape {tuple(lesion_mask.shape)} for compute_R10"
            )
    elif lesion_mask.ndim != 2:
        raise ValueError("lesion_mask must be 2D")

    if lesion_mask.shape != img.shape[1:]:
        raise ValueError(
            f"lesion_mask must have shape (H, W) matching img spatial dims. "
            f"Got img spatial={tuple(img.shape[1:])}, mask={tuple(lesion_mask.shape)}"
        )

    lesion_mask = lesion_mask.bool()
    H, W = lesion_mask.shape

    ys, xs = torch.where(lesion_mask)
    if ys.numel() == 0:
        return 0.0

    cy = ys.float().mean()
    cx = xs.float().mean()

    distances = torch.sqrt((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2)
    D = distances.max()
    if D <= 0:
        return 0.0

    max_radius = float(D.item() * 1.5)
    inner_margin = float(inner_margin_frac * D.item())
    outer_margin = float(outer_margin_frac * D.item())

    img = img.float()
    if img.max() > 1.5:
        img = img / 255.0
    red = img[0]  # (H, W)

    # ---- 2. Helpers ----
    def border_radius_along_angle(theta: float) -> Optional[float]:
        """
        Walk from centroid along angle theta until we leave the lesion.
        Return radius where we step outside, or None if no clear border.
        """
        step = 0.5
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
        theta: float, r_start: float, r_end: float, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample red-channel intensities along ray between r_start and r_end.
        Returns (rs, vals).
        """
        if r_end <= r_start:
            return torch.empty(0), torch.empty(0)

        rs = torch.linspace(r_start, r_end, num_samples)
        ys_f = cy + rs * math.sin(theta)
        xs_f = cx + rs * math.cos(theta)

        ys_i = ys_f.round().long()
        xs_i = xs_f.round().long()

        valid = (ys_i >= 0) & (ys_i < H) & (xs_i >= 0) & (xs_i < W)
        if not valid.any():
            return torch.empty(0), torch.empty(0)

        ys_i = ys_i[valid]
        xs_i = xs_i[valid]
        rs_valid = rs[valid]

        vals = red[ys_i, xs_i]
        return rs_valid, vals

    # ---- 3. Sweep angles, compute SSE per convergent fit ----
    angles = torch.linspace(0.0, 2.0 * math.pi, steps=n_angles + 1)[:-1]
    errors = []

    for theta in angles.tolist():
        r_border = border_radius_along_angle(theta)
        if r_border is None:
            continue

        r_start = max(0.0, r_border - inner_margin)
        r_end = min(max_radius, r_border + outer_margin)

        rs, vals = sample_radial_profile(theta, r_start, r_end, num_samples=n_samples)
        if rs.numel() < 10:
            continue

        # require some points on both sides of border to call it convergent
        inside_mask = rs < r_border
        outside_mask = rs >= r_border
        if inside_mask.sum() < 3 or outside_mask.sum() < 3:
            continue

        # ---- Linear regression fit: I(r) ≈ a*r + b ----
        x = rs.float()
        y = vals.float()

        x_mean = x.mean()
        y_mean = y.mean()
        x_centered = x - x_mean

        var_x = (x_centered ** 2).mean()
        if var_x <= 1e-8:
            continue

        cov_xy = (x_centered * (y - y_mean)).mean()
        a = cov_xy / var_x
        b = y_mean - a * x_mean

        y_pred = a * x + b
        sse = float(((y - y_pred) ** 2).sum().item())
        errors.append(sse)

    if len(errors) == 0:
        return 0.0

    # ---- 4. Mode of error distribution via histogram ----
    err_arr = np.array(errors, dtype=np.float32)

    if np.allclose(err_arr, err_arr[0]):
        # all errors basically the same
        return float(err_arr[0])

    # number of bins: min(n_bins, unique_values) but at least 3
    nb = max(3, min(n_bins, len(err_arr)))
    counts, bin_edges = np.histogram(err_arr, bins=nb)
    idx = int(np.argmax(counts))
    # bin center as mode estimate
    mode_val = 0.5 * (bin_edges[idx] + bin_edges[idx + 1])

    return float(mode_val)


def compute_R11(
    img: torch.Tensor,
    lesion_mask,
    n_axes: int = 18,
) -> float:
    """
    R11 – Max asymmetry normalized by eccentricity

    What it measures (per article):
        Worst-case asymmetry of the lesion silhouette, normalized by eccentricity.

    Implementation:
        1) Compute eccentricity E via skimage.regionprops if available,
           otherwise fall back to E = 1.0.
        2) For a set of axis orientations φ in [0, π):
            - Split lesion into two halves by line through centroid at angle φ.
            - Reflect one half across the axis.
            - Compute A(φ) = 1 - IoU( reflected_half, other_half ).
        3) R11 = max_φ A(φ) / max(E, eps).

    Returns:
        R11 as float, 0.0 on failure.
    """
    # --- normalize mask only (img is unused but kept for API symmetry) ---
    if isinstance(lesion_mask, np.ndarray):
        lesion_mask = torch.from_numpy(lesion_mask)
    elif not isinstance(lesion_mask, torch.Tensor):
        raise TypeError("lesion_mask must be torch.Tensor or numpy.ndarray")

    if lesion_mask.ndim == 3:
        if lesion_mask.shape[-1] in (1, 3):
            lesion_mask = lesion_mask.any(dim=-1)
        elif lesion_mask.shape[0] in (1, 3):
            lesion_mask = lesion_mask.any(dim=0)
        else:
            raise ValueError(
                f"Unexpected mask shape {tuple(lesion_mask.shape)} for compute_R11"
            )
    elif lesion_mask.ndim != 2:
        raise ValueError("lesion_mask must be 2D")

    lesion_mask = lesion_mask.bool()
    H, W = lesion_mask.shape

    ys, xs = torch.where(lesion_mask)
    if ys.numel() == 0:
        return 0.0

    cy = ys.float().mean().item()
    cx = xs.float().mean().item()

    lesion_np = lesion_mask.cpu().numpy().astype(bool)

    # ---- eccentricity E from regionprops, or fallback ----
    try:
        from skimage.measure import label, regionprops
        labeled = label(lesion_np)
        props = regionprops(labeled)
        if not props:
            E = 1.0
        else:
            # pick largest region
            props.sort(key=lambda p: p.area, reverse=True)
            E = float(props[0].eccentricity)
            if E <= 0.0:
                E = 1.0
    except ImportError:
        E = 1.0

    eps = 1e-6
    denom = max(E, eps)

    # ---- helper: asymmetry for a given axis angle φ ----
    coords = np.column_stack(np.where(lesion_np))  # (N, 2) -> (y, x)
    if coords.shape[0] == 0:
        return 0.0

    def asymmetry_for_angle(angle: float) -> float:
        # axis direction u, normal v
        u = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        v = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float32)

        ys = coords[:, 0].astype(np.float32)
        xs = coords[:, 1].astype(np.float32)

        px = xs - cx
        py = ys - cy

        # projection onto normal v decides the half
        proj_v = px * v[0] + py * v[1]

        half1_mask = proj_v >= 0  # one side
        half2_mask = proj_v < 0   # other side

        if not half1_mask.any() or not half2_mask.any():
            return 0.0

        # coordinates in basis (u, v)
        proj_u = px * u[0] + py * u[1]
        proj_v = proj_v  # already computed

        # reflect half1 across axis: v -> -v
        proj_u_h1 = proj_u[half1_mask]
        proj_v_h1 = proj_v[half1_mask]

        px_ref = proj_u_h1 * u[0] + (-proj_v_h1) * v[0]
        py_ref = proj_u_h1 * u[1] + (-proj_v_h1) * v[1]

        xs_ref = np.round(cx + px_ref).astype(int)
        ys_ref = np.round(cy + py_ref).astype(int)

        # build masks for reflected half1 and original half2
        refl_mask = np.zeros_like(lesion_np, dtype=bool)
        for y, x in zip(ys_ref, xs_ref):
            if 0 <= y < H and 0 <= x < W:
                refl_mask[y, x] = True

        half2_img = np.zeros_like(lesion_np, dtype=bool)
        ys_h2 = ys[half2_mask]
        xs_h2 = xs[half2_mask]
        half2_img[ys_h2.astype(int), xs_h2.astype(int)] = True

        overlap = np.logical_and(refl_mask, half2_img).sum()
        union = np.logical_or(refl_mask, half2_img).sum()

        if union == 0:
            return 0.0

        iou = overlap / union
        A = 1.0 - float(iou)
        return A

    # ---- sweep orientations and collect A(φ) ----
    angles = np.linspace(0.0, math.pi, num=n_axes, endpoint=False)
    A_vals = []
    for ang in angles:
        A_vals.append(asymmetry_for_angle(float(ang)))

    if len(A_vals) == 0:
        return 0.0

    A_vals = np.array(A_vals, dtype=np.float32)
    A_over_E = A_vals / denom

    R11 = float(A_over_E.max())
    return R11


def compute_R12(
    img: torch.Tensor,
    lesion_mask,
    n_angles: int = 64,
) -> float:
    """
    R12 – Total angular radius change

    What it measures (per article):
        Total “jaggedness” in the radial silhouette as you go around the lesion.

    Implementation:
        1) For angles θ_k sampled over [0, 2π):
            - Compute R(θ_k) = radius from centroid to lesion border along θ_k.
        2) R12 = sum_k | R(θ_k) - R(θ_{k-1}) |  (with wrap-around).

    Returns:
        R12 as float, 0.0 on failure.
    """
    # normalize mask
    if isinstance(lesion_mask, np.ndarray):
        lesion_mask = torch.from_numpy(lesion_mask)
    elif not isinstance(lesion_mask, torch.Tensor):
        raise TypeError("lesion_mask must be torch.Tensor or numpy.ndarray")

    if lesion_mask.ndim == 3:
        if lesion_mask.shape[-1] in (1, 3):
            lesion_mask = lesion_mask.any(dim=-1)
        elif lesion_mask.shape[0] in (1, 3):
            lesion_mask = lesion_mask.any(dim=0)
        else:
            raise ValueError(
                f"Unexpected mask shape {tuple(lesion_mask.shape)} for compute_R12"
            )
    elif lesion_mask.ndim != 2:
        raise ValueError("lesion_mask must be 2D")

    lesion_mask = lesion_mask.bool()
    H, W = lesion_mask.shape

    ys, xs = torch.where(lesion_mask)
    if ys.numel() == 0:
        return 0.0

    cy = ys.float().mean()
    cx = xs.float().mean()

    distances = torch.sqrt((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2)
    D = distances.max().item()
    if D <= 0:
        return 0.0

    max_radius = D * 1.5

    # border radius helper
    def border_radius(theta: float) -> float | None:
        step = 0.5
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

    # collect radii
    angles = torch.linspace(0.0, 2.0 * math.pi, steps=n_angles + 1)[:-1]
    radii = []

    for theta in angles.tolist():
        r = border_radius(theta)
        if r is None:
            radii.append(0.0)
        else:
            radii.append(float(r))

    if len(radii) < 2:
        return 0.0

    radii = np.array(radii, dtype=np.float32)
    # circular difference
    diffs = np.abs(np.roll(radii, 1) - radii)
    R12 = float(diffs.sum())
    return R12


def compute_R13(
    img: torch.Tensor,
    lesion_mask,
    n_axes: int = 18,
) -> float:
    """
    R13 – Asymmetry of silhouette (standard metric)

    What it measures (per article):
        Standard dermoscopy-style asymmetry score based on rotational
        comparison of halves and orientation of symmetry axis.

    Implementation:
        1) For φ in [0, π):
            - Compute A(φ) = 1 - IoU( reflected half across axis at φ,
                                      opposite half ).
        2) Let φ_sym = argmin_φ A(φ), and A_sym = min_φ A(φ).
        3) R13 = A_sym * |φ_sym - π/2|   (as in article's definition).

    Returns:
        R13 as float, 0.0 on failure.
    """
    # normalize mask
    if isinstance(lesion_mask, np.ndarray):
        lesion_mask = torch.from_numpy(lesion_mask)
    elif not isinstance(lesion_mask, torch.Tensor):
        raise TypeError("lesion_mask must be torch.Tensor or numpy.ndarray")

    if lesion_mask.ndim == 3:
        if lesion_mask.shape[-1] in (1, 3):
            lesion_mask = lesion_mask.any(dim=-1)
        elif lesion_mask.shape[0] in (1, 3):
            lesion_mask = lesion_mask.any(dim=0)
        else:
            raise ValueError(
                f"Unexpected mask shape {tuple(lesion_mask.shape)} for compute_R13"
            )
    elif lesion_mask.ndim != 2:
        raise ValueError("lesion_mask must be 2D")

    lesion_mask = lesion_mask.bool()
    H, W = lesion_mask.shape

    ys, xs = torch.where(lesion_mask)
    if ys.numel() == 0:
        return 0.0

    cy = ys.float().mean().item()
    cx = xs.float().mean().item()

    lesion_np = lesion_mask.cpu().numpy().astype(bool)
    coords = np.column_stack(np.where(lesion_np))
    if coords.shape[0] == 0:
        return 0.0

    def asymmetry_for_angle(angle: float) -> float:
        u = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        v = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float32)

        ys_c = coords[:, 0].astype(np.float32)
        xs_c = coords[:, 1].astype(np.float32)

        px = xs_c - cx
        py = ys_c - cy

        proj_v = px * v[0] + py * v[1]
        half1_mask = proj_v >= 0
        half2_mask = proj_v < 0

        if not half1_mask.any() or not half2_mask.any():
            return 0.0

        proj_u = px * u[0] + py * u[1]
        proj_v = proj_v

        proj_u_h1 = proj_u[half1_mask]
        proj_v_h1 = proj_v[half1_mask]

        px_ref = proj_u_h1 * u[0] + (-proj_v_h1) * v[0]
        py_ref = proj_u_h1 * u[1] + (-proj_v_h1) * v[1]

        xs_ref = np.round(cx + px_ref).astype(int)
        ys_ref = np.round(cy + py_ref).astype(int)

        refl_mask = np.zeros_like(lesion_np, dtype=bool)
        for y, x in zip(ys_ref, xs_ref):
            if 0 <= y < H and 0 <= x < W:
                refl_mask[y, x] = True

        half2_img = np.zeros_like(lesion_np, dtype=bool)
        ys_h2 = ys_c[half2_mask]
        xs_h2 = xs_c[half2_mask]
        half2_img[ys_h2.astype(int), xs_h2.astype(int)] = True

        overlap = np.logical_and(refl_mask, half2_img).sum()
        union = np.logical_or(refl_mask, half2_img).sum()

        if union == 0:
            return 0.0

        iou = overlap / union
        A = 1.0 - float(iou)
        return A

    angles = np.linspace(0.0, math.pi, num=n_axes, endpoint=False)
    A_vals = []
    for ang in angles:
        A_vals.append(asymmetry_for_angle(float(ang)))

    if len(A_vals) == 0:
        return 0.0

    A_vals = np.array(A_vals, dtype=np.float32)
    idx = int(np.argmin(A_vals))
    A_sym = float(A_vals[idx])
    phi_sym = float(angles[idx])

    R13 = A_sym * abs(phi_sym - (math.pi / 2.0))
    return R13


if __name__ == "__main__":
    sample_paths = random.sample(all_imgs, 10)

    for lesion_path in sample_paths:
        print(lesion_path)

        # Load raw image as numpy (H, W, 3)
        img_np = mpimg.imread(lesion_path)

        # Convert to (3, H, W) tensor
        lesion = as_tensor_3chw(img_np)

        # IMPORTANT: pass the original numpy image to calc_threshold_mask
        mask = calc_threshold_mask(img_np)

        r1 = compute_R1(lesion, mask)
        print("R1:", r1)

        r2 = compute_R2(lesion, mask)
        print("R2:", r2)

        r3 = compute_R3(lesion, mask)
        print("R3:", r3)

        r4 = compute_R4(lesion, mask)
        print("R4:", r4)

        r5 = compute_R5(lesion, mask)
        print("R5:", r5)

        r6 = compute_R6(lesion, mask)
        print("R6:", r6)

        r7 = compute_R7(lesion, mask)
        print("R7:", r7)

        r8 = compute_R8(lesion, mask)
        print("R8:", r8)

        r9 = compute_R9(lesion, mask)
        print("R9:", r9)

        r10 = compute_R10(lesion, mask)
        print("R10:", r10)

        r11 = compute_R11(lesion, mask)
        print("R11:", r11)

        r12 = compute_R12(lesion, mask)
        print("R12:", r12)

        r13 = compute_R13(lesion, mask)
        print("R13:", r13)