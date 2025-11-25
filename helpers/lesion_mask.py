import collections

import numpy as np
import torch
import torch.nn.functional as F


def as_tensor_3chw(img):
    """
    Accepts numpy (H,W,3)
    Returns torch tensor (3,H,W) float32 in [0,1].
    """

    img_copy = np.array(img, copy=True)
    t = torch.tensor(img_copy, dtype=torch.float32)

    if t.max() > 1.5:
        t = t / 255.0

    return t.permute(2, 0, 1).contiguous()


def rgb_to_gray_torch(img):
    """
    img: (3, H, W) in [0,1] or [0,255]
    returns gray: (H, W) in [0,1]
    """
    if img.dtype != torch.float32:
        img = img.float()
    if img.max() > 1.5:
        img = img / 255.0
    r, g, b = img[0], img[1], img[2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def otsu_threshold_torch(gray, num_bins=256):
    """
    gray: (H, W) tensor in [0,1]
    returns scalar threshold in [0,1]
    """
    gray_flat = gray.view(-1)
    hist = torch.histc(gray_flat, bins=num_bins, min=0.0, max=1.0)
    hist = hist.float()
    p = hist / hist.sum()  # probability of each bin

    bins = torch.arange(num_bins, device=gray.device).float()
    omega = torch.cumsum(p, dim=0)  # cumulative prob
    mu = torch.cumsum(p * bins, dim=0)  # cumulative mean
    mu_t = mu[-1]

    # between-class variance
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-8)

    # ignore invalid entries at extremes
    sigma_b2[omega == 0] = 0
    sigma_b2[omega == 1] = 0

    t = torch.argmax(sigma_b2)
    thresh = (t.float() / (num_bins - 1)).item()
    return thresh


def calc_threshold_mask(img, kernel_size=7):
    """
    img: (3, H, W) RGB tensor or (H, W, 3) numpy
    returns: mask (H, W) boolean tensor
    """
    img = as_tensor_3chw(img)
    gray = rgb_to_gray_torch(img)

    # otsu threshold
    t = otsu_threshold_torch(gray)

    # lesions are darker than surroundings
    rough_mask = gray < t
    m = rough_mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    pad = kernel_size // 2
    dilated = F.max_pool2d(m, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=pad)
    closed = (eroded[0, 0] > 0.5)  # (H, W) bool

    # flood-fill from borders over True pixels and turn them off
    H, W = closed.shape
    mask = closed.clone()
    visited = torch.zeros_like(mask, dtype=torch.bool)

    q = collections.deque()

    # enqueue all border pixels that are currently True
    for x in range(W):
        if mask[0, x]:
            q.append((0, x))
        if mask[H - 1, x]:
            q.append((H - 1, x))
    for y in range(H):
        if mask[y, 0]:
            q.append((y, 0))
        if mask[y, W - 1]:
            q.append((y, W - 1))

    # 4-connected flood fill
    while q:
        y, x = q.popleft()
        if visited[y, x]:
            continue
        if not mask[y, x]:
            continue
        visited[y, x] = True
        # turn off any region connected to the border
        mask[y, x] = False
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and mask[ny, nx]:
                q.append((ny, nx))

    return mask