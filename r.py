# biomarkers/r.py

import math
from typing import Optional

import torch


def compute_r1(
    img: torch.Tensor,
    lesion_mask: torch.Tensor,
    n_angles: int = 64,
    inner_margin_frac: float = 0.25,
    outer_margin_frac: float = 0.25,
) -> float:
    """
    Compute R1: the standard deviation of the edge 'slope' parameter in the red channel.

    This is a torch-only, simplified approximation of the edge-slope idea:
      - For each angle θ, we find the lesion border radius in that direction.
      - We sample red-channel intensities along a radial segment centered on the border.
      - We define a "slope" as the intensity jump across the border normalized by window width.
      - R1 is the std of these slopes over all valid angles.

    Args:
        img:          (3, H, W) RGB image tensor. dtype float or uint8.
        lesion_mask:  (H, W) boolean tensor, True inside lesion.
        n_angles:     number of angles over [0, 2π) to sample.
        inner_margin_frac: fraction of global lesion radius D used for inside window.
        outer_margin_frac: fraction of global lesion radius D used for outside window.

    Returns:
        R1 as a Python float. Returns float('nan') if too few valid slopes.
    """
    # ---- 1. Normalize inputs / shapes ----
    if not isinstance(img, torch.Tensor):
        raise TypeError("img must be a torch.Tensor")
    if not isinstance(lesion_mask, torch.Tensor):
        raise TypeError("lesion_mask must be a torch.Tensor")

    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError("img must have shape (3, H, W)")
    if lesion_mask.shape != img.shape[1:]:
        raise ValueError("lesion_mask must have shape (H, W) matching img spatial dims")

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

    # Approximate global "radius" as max distance from centroid to lesion pixels
    distances = torch.sqrt((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2)
    D = distances.max()
    if D <= 0:
        return float("nan")

    # Precompute some constants
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
                # Approximate border as current radius
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
            rs: 1D tensor of radii (N,)
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
    angles = torch.linspace(0.0, 2.0 * math.pi, steps=n_angles, endpoint=False)
    slopes = []

    for theta in angles.tolist():
        # 1) Find border radius along this angle
        r_border = border_radius_along_angle(theta)
        if r_border is None:
            continue

        # Define a window around the border
        r_start = max(0.0, r_border - inner_margin)
        r_end = min(max_radius, r_border + outer_margin)

        rs, vals = sample_radial_profile(theta, r_start, r_end, num_samples=64)
        if rs.numel() < 10:
            continue

        # Split into inside/outside segments relative to border
        inside_mask = rs < r_border
        outside_mask = rs >= r_border

        if inside_mask.sum() < 3 or outside_mask.sum() < 3:
            continue

        inside_vals = vals[inside_mask]
        outside_vals = vals[outside_mask]

        # Approximate slope as intensity jump across border divided by window width
        delta_I = outside_vals.mean() - inside_vals.mean()
        window_width = (rs.max() - rs.min()).clamp(min=1e-6)
        slope = float(delta_I / window_width)
        slopes.append(slope)

    # ---- 5. Aggregate slopes -> R1 ----
    if len(slopes) < 3:
        return float("nan")

    slopes_t = torch.tensor(slopes, dtype=torch.float32)
    R1 = float(slopes_t.std(unbiased=False))  # population std
    return R1

