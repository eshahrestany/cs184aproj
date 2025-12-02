import numpy as np
import torch


def _to_numpy_hwc(image) -> np.ndarray:
    """
    Accepts:
        - numpy (H, W, 3)
        - torch (3, H, W) or (H, W, 3)
    Returns:
        numpy (H, W, 3)
    """
    if isinstance(image, np.ndarray):
        img = image
    elif torch.is_tensor(image):
        t = image.detach().cpu()
        if t.ndim == 3 and t.shape[0] == 3:
            # (3, H, W) -> (H, W, 3)
            img = t.permute(1, 2, 0).numpy()
        elif t.ndim == 3 and t.shape[2] == 3:
            # already (H, W, 3)
            img = t.numpy()
        else:
            raise ValueError(f"Unsupported torch image shape {tuple(t.shape)}")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Grayscale -> replicate to 3 channels
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    # RGBA -> drop alpha
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected image (H, W, 3), got {img.shape}")

    return img


def _to_mask_np(mask, shape_hw) -> np.ndarray:
    """
    Accepts:
        - numpy mask (H, W) or (H, W, C) or (C, H, W)
        - torch mask with similar shapes
    Returns:
        numpy bool mask (H, W)
    """
    if torch.is_tensor(mask):
        m = mask.detach().cpu().numpy()
    else:
        m = np.asarray(mask)

    if m.ndim == 3:
        # (H, W, C) or (C, H, W)
        if m.shape[-1] in (1, 3):
            m = m.any(axis=-1)
        elif m.shape[0] in (1, 3):
            m = m.any(axis=0)
        else:
            raise ValueError(f"Unexpected mask shape {m.shape}")
    elif m.ndim != 2:
        raise ValueError(f"Mask must be 2D or 3D, got ndim={m.ndim}")

    if m.shape != shape_hw:
        raise ValueError(
            f"Mask shape {m.shape} does not match image spatial shape {shape_hw}"
        )

    return m.astype(bool)


def compute_B1(image, mask_np, dtheta_deg=2):
    """
    B1: average absolute angular derivative of radial mean brightness
        (blue channel), normalized by angular step.

    image: np.ndarray(H,W,3) or torch.Tensor(3,H,W)/(H,W,3)
    mask_np: np.ndarray or torch.Tensor, broadcastable to (H,W) bool
    """
    img = _to_numpy_hwc(image)
    H, W, _ = img.shape
    mask_np = _to_mask_np(mask_np, (H, W))

    blue_channel = img[:, :, 0]

    ys, xs = np.where(mask_np)
    if ys.size == 0:
        return 0.0

    cy = np.mean(ys)
    cx = np.mean(xs)

    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    Rmax = int(np.max(dist))
    if Rmax <= 1:
        return 0.0

    dtheta = np.deg2rad(dtheta_deg)
    theta_values = np.arange(0, 2 * np.pi, dtheta)

    Rm_values = []

    for theta in theta_values:
        rs = np.arange(0, Rmax)
        xs_line = cx + np.cos(theta) * rs
        ys_line = cy + np.sin(theta) * rs

        xs_line = xs_line.astype(int)
        ys_line = ys_line.astype(int)

        valid = (
            (xs_line >= 0)
            & (xs_line < W)
            & (ys_line >= 0)
            & (ys_line < H)
        )
        xs_line = xs_line[valid]
        ys_line = ys_line[valid]

        if xs_line.size == 0:
            Rm_values.append(0.0)
            continue

        inside_mask = mask_np[ys_line, xs_line]
        xs_line = xs_line[inside_mask]
        ys_line = ys_line[inside_mask]

        if xs_line.size == 0:
            Rm_values.append(0.0)
            continue

        p_values = blue_channel[ys_line, xs_line]
        Rm_values.append(float(np.mean(p_values)))

    Rm_values = np.asarray(Rm_values, dtype=np.float32)
    if Rm_values.size == 0:
        return 0.0

    dRm = np.abs(np.diff(Rm_values, prepend=Rm_values[-1]))
    B1 = float(np.mean(dRm) / dtheta)
    if not np.isfinite(B1):
        return 0.0
    return B1


def compute_B2(image, mask_np):
    """
    B2: variance of angular radial brightness std (blue channel).

    image: np.ndarray(H,W,3) or torch.Tensor(3,H,W)/(H,W,3)
    mask_np: np.ndarray or torch.Tensor, broadcastable to (H,W) bool
    """
    img = _to_numpy_hwc(image)
    H, W, _ = img.shape
    mask_np = _to_mask_np(mask_np, (H, W))

    blue_channel = img[:, :, 0]

    ys, xs = np.where(mask_np)
    if ys.size == 0:
        return 0.0

    cy = np.mean(ys)
    cx = np.mean(xs)

    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    Rmax = int(np.max(dist))
    if Rmax <= 1:
        return 0.0

    dtheta = np.deg2rad(2)
    theta_values = np.arange(0, 2 * np.pi, dtheta)

    Rstd_values = []

    for theta in theta_values:
        rs = np.arange(0, Rmax)
        xs_line = cx + np.cos(theta) * rs
        ys_line = cy + np.sin(theta) * rs

        xs_line = xs_line.astype(int)
        ys_line = ys_line.astype(int)

        valid = (
            (xs_line >= 0)
            & (xs_line < W)
            & (ys_line >= 0)
            & (ys_line < H)
        )
        xs_line = xs_line[valid]
        ys_line = ys_line[valid]

        if xs_line.size == 0:
            Rstd_values.append(0.0)
            continue

        inside_mask = mask_np[ys_line, xs_line]
        xs_line = xs_line[inside_mask]
        ys_line = ys_line[inside_mask]

        if xs_line.size == 0:
            Rstd_values.append(0.0)
            continue

        p_values = blue_channel[ys_line, xs_line]
        if p_values.size < 2:
            Rstd_values.append(0.0)
        else:
            Rstd_values.append(float(np.std(p_values, ddof=1)))

    Rstd_values = np.asarray(Rstd_values, dtype=np.float32)
    if Rstd_values.size < 2:
        return 0.0

    B2 = float(np.var(Rstd_values, ddof=1))
    if not np.isfinite(B2):
        return 0.0
    return B2

def compute_B10(image, mask_np):
    """
    B10: absolute slope of angular trend in radial mean brightness (blue channel),
         normalized by mean brightness.

    image: np.ndarray(H,W,3) or torch.Tensor(3,H,W)/(H,W,3)
    mask_np: np.ndarray or torch.Tensor, broadcastable to (H,W) bool
    """
    img = _to_numpy_hwc(image)
    H, W, _ = img.shape
    mask_np = _to_mask_np(mask_np, (H, W))

    blue_channel = img[:, :, 0]

    ys, xs = np.where(mask_np)
    if ys.size == 0:
        return 0.0

    cy = np.mean(ys)
    cx = np.mean(xs)

    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    Rmax = int(np.max(dist))
    if Rmax <= 1:
        return 0.0

    dtheta = np.deg2rad(2)
    theta_values = np.arange(0, 2 * np.pi, dtheta)

    Rm_values = []

    for theta in theta_values:
        rs = np.arange(0, Rmax)
        xs_line = cx + np.cos(theta) * rs
        ys_line = cy + np.sin(theta) * rs

        xs_line = xs_line.astype(int)
        ys_line = ys_line.astype(int)

        valid = (
            (xs_line >= 0)
            & (xs_line < W)
            & (ys_line >= 0)
            & (ys_line < H)
        )
        xs_line = xs_line[valid]
        ys_line = ys_line[valid]

        if xs_line.size == 0:
            Rm_values.append(0.0)
            continue

        inside_mask = mask_np[ys_line, xs_line]
        xs_line = xs_line[inside_mask]
        ys_line = ys_line[inside_mask]

        if xs_line.size == 0:
            Rm_values.append(0.0)
            continue

        p_values = blue_channel[ys_line, xs_line]
        Rm_values.append(float(np.mean(p_values)))

    Rm_values = np.asarray(Rm_values, dtype=np.float32)
    if Rm_values.size == 0:
        return 0.0

    mean_Rm = float(np.mean(Rm_values))
    if mean_Rm == 0.0:
        return 0.0

    # Fit linear trend Rm(theta) ≈ a * theta + b
    a, b = np.polyfit(theta_values, Rm_values, 1)
    B10 = float(abs(a) / mean_Rm)
    if not np.isfinite(B10):
        return 0.0
    return B10


def compute_B11(image, mask_np):
    """
    B11: range of radial mean brightness (blue channel) across angles,
         normalized by mean brightness.

    image: np.ndarray(H,W,3) or torch.Tensor(3,H,W)/(H,W,3)
    mask_np: np.ndarray or torch.Tensor, broadcastable to (H,W) bool
    """
    img = _to_numpy_hwc(image)
    H, W, _ = img.shape
    mask_np = _to_mask_np(mask_np, (H, W))

    blue_channel = img[:, :, 0]

    ys, xs = np.where(mask_np)
    if ys.size == 0:
        return 0.0

    cy = np.mean(ys)
    cx = np.mean(xs)

    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    Rmax = int(np.max(dist))
    if Rmax <= 1:
        return 0.0

    dtheta = np.deg2rad(2)
    theta_values = np.arange(0, 2 * np.pi, dtheta)

    Rm_values = []

    for theta in theta_values:
        rs = np.arange(0, Rmax)
        xs_line = cx + np.cos(theta) * rs
        ys_line = cy + np.sin(theta) * rs

        xs_line = xs_line.astype(int)
        ys_line = ys_line.astype(int)

        valid = (
            (xs_line >= 0)
            & (xs_line < W)
            & (ys_line >= 0)
            & (ys_line < H)
        )
        xs_line = xs_line[valid]
        ys_line = ys_line[valid]

        if xs_line.size == 0:
            Rm_values.append(0.0)
            continue

        inside_mask = mask_np[ys_line, xs_line]
        xs_line = xs_line[inside_mask]
        ys_line = ys_line[inside_mask]

        if xs_line.size == 0:
            Rm_values.append(0.0)
            continue

        p_values = blue_channel[ys_line, xs_line]
        Rm_values.append(float(np.mean(p_values)))

    Rm_values = np.asarray(Rm_values, dtype=np.float32)
    if Rm_values.size == 0:
        return 0.0

    mean_Rm = float(np.mean(Rm_values))
    if mean_Rm == 0.0:
        return 0.0

    R_range = float(np.max(Rm_values) - np.min(Rm_values))
    B11 = R_range / mean_Rm
    if not np.isfinite(B11):
        return 0.0
    return B11

