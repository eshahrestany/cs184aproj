import math
import numpy as np
import torch


def compute_G1(
    img: torch.Tensor,
    lesion_mask,
) -> float:
    """
    G1 – Normalized perimeter irregularity (green channel)

    What it measures (per article):
        How irregular the border is versus a circle of equal area.

    Conceptual formula:
        Let Perim_G be lesion perimeter length on L_green.
        Let A be lesion area (pixels).

        For a perfect circle of area A, perimeter P_circle satisfies:
            P_circle = 2 * sqrt(pi * A)

        Define:
            norm_perim = Perim_G / sqrt(A)
            circle_norm = 2 * sqrt(pi)

        Then:
            G1 = norm_perim - circle_norm
               = Perim_G / sqrt(A) - 2 * sqrt(pi)

        So G1 = 0 for a perfect circle and increases with border irregularity.

    Implementation details:
        - We use the provided lesion_mask to define the region (implicitly
          L_green). The shape itself does not depend on channel.
        - Perimeter is approximated as the number of 4-connected boundary pixels.

    Returns:
        G1 as Python float. Returns 0.0 on any degenerate case.
    """
    # img is unused, but kept for API symmetry with R* functions
    # and possible future green-channel-dependent variants.

    # ---- Normalize / convert lesion_mask ----
    if isinstance(lesion_mask, np.ndarray):
        lesion_mask = torch.from_numpy(lesion_mask)
    elif not isinstance(lesion_mask, torch.Tensor):
        raise TypeError("lesion_mask must be a torch.Tensor or numpy.ndarray")

    if lesion_mask.ndim == 3:
        # handle (H, W, C) or (C, H, W) by reducing over channels
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

    lesion_mask = lesion_mask.bool()
    H, W = lesion_mask.shape

    ys, xs = torch.where(lesion_mask)
    area = ys.numel()
    if area == 0:
        return 0.0

    # ---- Perimeter: count boundary pixels (4-connected) ----
    perimeter = 0
    ys_np = ys.cpu().numpy()
    xs_np = xs.cpu().numpy()

    for y, x in zip(ys_np, xs_np):
        y = int(y)
        x = int(x)
        # A boundary pixel if it has at least one 4-neighbor outside lesion
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W or not lesion_mask[ny, nx]:
                perimeter += 1
                break

    if perimeter <= 0:
        return 0.0

    area_f = float(area)
    perim_f = float(perimeter)

    # normalized perimeter vs circle baseline
    norm_perim = perim_f / math.sqrt(area_f)
    circle_norm = 2.0 * math.sqrt(math.pi)

    G1 = norm_perim - circle_norm
    return float(G1)
