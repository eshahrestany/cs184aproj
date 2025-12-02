import cv2
import numpy as np
import torch

COLOR_RANGES = {
    "light brown": [(166, 143, 119), (171, 113, 53)],
    "dark brown":  [(43, 34, 25),   (143, 70, 0)],
    "red":         [(92, 67, 67),   (255, 0, 0)],
    "black":       [(0, 0, 0),      (67, 67, 67)],
    "blue-gray/white": [(74, 103, 133), (255, 255, 255)],
}


def _to_numpy_image_bgr(image: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Accepts:
        - numpy (H, W, 3)
        - torch (3, H, W)
    Returns:
        uint8 BGR numpy image (H, W, 3)
    """
    if isinstance(image, torch.Tensor):
        # assume (3, H, W), float in [0,1] or [0,255], RGB order
        img = image.detach().cpu()
        if img.ndim != 3 or img.shape[0] != 3:
            raise ValueError(f"Expected torch (3, H, W), got {tuple(img.shape)}")
        img = img.permute(1, 2, 0).numpy()  # (H, W, 3), RGB
    else:
        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected numpy (H, W, 3), got {tuple(img.shape)}")

    # If float, normalize to [0,255]
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Assume RGB from most loaders; convert to BGR for OpenCV ranges
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


def _to_mask_bool(mask) -> np.ndarray:
    """
    Accepts:
        - numpy mask (H, W)
        - torch mask (H, W)
    Returns:
        bool numpy mask (H, W)
    """
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy()
    else:
        m = np.asarray(mask)

    # treat nonzero as True
    return m.astype(bool)


def compute_MC1(image, mask) -> list[int]:
    """
    MC1: presence of specific colors inside the lesion.

    Returns:
        CL: list of 0/1 flags for each color in COLOR_RANGES, in dict order.
    """
    img_bgr = _to_numpy_image_bgr(image)
    mask_np = _to_mask_bool(mask)

    if img_bgr.shape[:2] != mask_np.shape:
        raise ValueError(
            f"Image and mask shape mismatch for MC1: "
            f"image {img_bgr.shape[:2]}, mask {mask_np.shape}"
        )

    CL = []
    for color_name, (low, high) in COLOR_RANGES.items():
        low = np.array(low, dtype=np.uint8)
        high = np.array(high, dtype=np.uint8)

        # Color mask for the entire image
        color_mask = cv2.inRange(img_bgr, low, high).astype(bool)

        # Restrict to lesion region
        inside_lesion = color_mask & mask_np
        present = np.any(inside_lesion)

        CL.append(1 if present else 0)

    return CL


def compute_MC2(image, mask_np):
    """
    MC2: red vs blue area ratio inside the lesion.

    image:
        - np.ndarray (H, W, 3), RGB or BGR, float [0,1] or uint8
        - or torch.Tensor (3, H, W) / (H, W, 3)
    mask_np:
        - np.ndarray or torch.Tensor, 2D or 3D, will be reduced to (H, W) bool
    """

    # ---------- Normalize image to uint8 (H, W, 3) ----------
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
            img = np.transpose(img, (1, 2, 0))
    else:
        img = np.asarray(image)

    # Grayscale -> 3-channel
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    # RGBA -> drop alpha
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"compute_MC2: unexpected image shape {img.shape}")

    img = img.astype(np.float32)
    if img.max() <= 1.5:
        img = img * 255.0
    img_u8 = img.clip(0, 255).astype(np.uint8)

    H, W, _ = img_u8.shape

    # ---------- Normalize mask to (H, W) bool ----------
    if torch.is_tensor(mask_np):
        m = mask_np.detach().cpu().numpy()
    else:
        m = np.asarray(mask_np)

    if m.ndim == 3:
        # (H, W, C) or (C, H, W)
        if m.shape[-1] in (1, 3):
            m = m.any(axis=-1)
        elif m.shape[0] in (1, 3):
            m = m.any(axis=0)
        else:
            raise ValueError(f"compute_MC2: unexpected mask shape {m.shape}")
    elif m.ndim != 2:
        raise ValueError(f"compute_MC2: mask must be 2D or 3D, got ndim={m.ndim}")

    if m.shape != (H, W):
        raise ValueError(
            f"compute_MC2: mask shape {m.shape} does not match image spatial { (H, W) }"
        )

    mask_bool = m.astype(bool)

    # ---------- Compute red vs blue area inside lesion ----------
    b, g, r = cv2.split(img_u8)

    # Binary maps as float64 to avoid overflow
    Lred = (r > 150).astype(np.float64)
    Lblue = (b > 150).astype(np.float64)
    mask64 = mask_bool.astype(np.float64)

    A_red = np.sum(Lred * mask64)
    A_blue = np.sum(Lblue * mask64)

    denom = A_red + A_blue
    if denom == 0:
        return 0.0

    mc2 = (A_red - A_blue) / denom

    # keep it sane
    mc2 = float(np.clip(mc2, -1.0, 1.0))
    if not np.isfinite(mc2):
        return 0.0
    return mc2




def compute_MC4(image, mask) -> int:
    """
    MC4: presence of blue-gray structures inside the lesion.

    Returns:
        1 if any blue-gray pixel exists in the lesion, else 0.
    """
    img_bgr = _to_numpy_image_bgr(image)
    mask_np = _to_mask_bool(mask)

    if img_bgr.shape[:2] != mask_np.shape:
        raise ValueError(
            f"Image and mask shape mismatch for MC4: "
            f"image {img_bgr.shape[:2]}, mask {mask_np.shape}"
        )

    # Brightness threshold based on percentile
    brightness = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thr = np.percentile(brightness, 85)
    bright_pixels = brightness >= thr

    b, g, r = cv2.split(img_bgr)

    # Grayish pixels: difference between channels is small
    gray_pixels = (
        (np.abs(r - g) <= 20) &
        (np.abs(r - b) <= 20) &
        (np.abs(g - b) <= 20)
    )

    # Blue-gray: bright AND grayish OR strong blue
    Lbluegray = ((bright_pixels & gray_pixels) | (b > thr)).astype(bool)

    inside = Lbluegray & mask_np

    return int(np.any(inside))


def convert_img_to_multi_color_IBC(image, mask) -> float:
    """
    Convenience wrapper:
        Returns sum of MC1 flags = number of distinct colors present in lesion.
    """
    CL = compute_MC1(image, mask)
    return float(sum(CL))

