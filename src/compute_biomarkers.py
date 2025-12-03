# compute_biomarkers.py

import csv
import os
from typing import Dict, List, Tuple

import matplotlib.image as mpimg
import numpy as np
import multiprocessing as mp

from src.b import compute_B1, compute_B2, compute_B10, compute_B11
from src.g import compute_G1
from src.lesion_mask import as_tensor_3chw, calc_threshold_mask
from src.mc import compute_MC1, compute_MC2, compute_MC4
from src.r import (
    compute_R1, compute_R2, compute_R3, compute_R4, compute_R5, compute_R6,
    compute_R7, compute_R8, compute_R9, compute_R10, compute_R11, compute_R12, compute_R13
)

# ---------------------------------------------------------------------
# Feature ordering (must match the model input ordering)
# ---------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "B1", "B2", "B10", "B11",
    "MC1", "MC2", "MC4",
    "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12", "R13",
    "G1",
]


def _load_image_rgb(path: str) -> np.ndarray:
    """
    Load an image and force it to be (H, W, 3) float32 in [0, 1] RGB.
    Handles grayscale and RGBA images.
    """
    img = mpimg.imread(path)

    # Grayscale -> 3 channels
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    # RGBA -> drop alpha
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.ndim != 3 or img.shape[2] != 3:
        raise RuntimeError(f"Unexpected image shape for {path}: {img.shape}")

    img = img.astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0

    return img


def _safe_compute(name: str, img_path: str, fn):
    """
    Wrap a single biomarker computation so that if it fails, we know
    exactly which biomarker and which image caused the failure.
    """
    try:
        return fn()
    except Exception as e:
        print(f"[ERROR] Biomarker {name} failed for {img_path}: {e}")
        raise


def _extract_id_from_path(img_path: str) -> int:
    """
    Given something like:
        .../train/benign/melanoma_100.jpg
    returns:
        100 (as int)
    """
    fname = os.path.basename(img_path)          # melanoma_100.jpg
    stem, _ = os.path.splitext(fname)          # melanoma_100
    parts = stem.split("_")
    id_str = parts[-1]
    try:
        return int(id_str)
    except ValueError:
        print(f"[WARN] Could not parse numeric ID from filename '{fname}', using -1")
        return -1


def _compute_biomarkers_for_image(img_path: str, label: float) -> Dict[str, float]:
    """
    Compute all biomarkers for a single image.

    Returns a dict with:
        - ID (numeric, from filename)
        - label (0 or 1)
        - one key per FEATURE_NAMES entry

    If any biomarker computation fails, the exception is propagated.
    """
    img_np = _load_image_rgb(img_path)

    # Mask + tensor for torch-based biomarkers
    mask = calc_threshold_mask(img_np)
    lesion = as_tensor_3chw(img_np)  # (3, H, W) torch tensor

    # ---- B-series ----
    B1 = float(_safe_compute("B1", img_path, lambda: compute_B1(img_np, mask)))
    B2 = float(_safe_compute("B2", img_path, lambda: compute_B2(img_np, mask)))
    B10 = float(_safe_compute("B10", img_path, lambda: compute_B10(img_np, mask)))
    B11 = float(_safe_compute("B11", img_path, lambda: compute_B11(img_np, mask)))

    # ---- MC-series ----
    mc1_list = _safe_compute("MC1", img_path, lambda: compute_MC1(img_np, mask))
    MC1 = float(sum(mc1_list))
    MC2 = float(_safe_compute("MC2", img_path, lambda: compute_MC2(img_np, mask)))
    MC4 = float(_safe_compute("MC4", img_path, lambda: compute_MC4(img_np, mask)))

    # ---- R-series ----
    R1 = float(_safe_compute("R1", img_path, lambda: compute_R1(lesion, mask)))
    R2 = float(_safe_compute("R2", img_path, lambda: compute_R2(lesion, mask)))
    R3 = float(_safe_compute("R3", img_path, lambda: compute_R3(lesion, mask)))
    R4 = float(_safe_compute("R4", img_path, lambda: compute_R4(lesion, mask)))
    R5 = float(_safe_compute("R5", img_path, lambda: compute_R5(lesion, mask)))
    R6 = float(_safe_compute("R6", img_path, lambda: compute_R6(lesion, mask)))
    R7 = float(_safe_compute("R7", img_path, lambda: compute_R7(lesion, mask)))
    R8 = float(_safe_compute("R8", img_path, lambda: compute_R8(lesion, mask)))
    R9 = float(_safe_compute("R9", img_path, lambda: compute_R9(lesion, mask)))
    R10 = float(_safe_compute("R10", img_path, lambda: compute_R10(lesion, mask)))
    R11 = float(_safe_compute("R11", img_path, lambda: compute_R11(lesion, mask)))
    R12 = float(_safe_compute("R12", img_path, lambda: compute_R12(lesion, mask)))
    R13 = float(_safe_compute("R13", img_path, lambda: compute_R13(lesion, mask)))

    # ---- G-series ----
    G1 = float(_safe_compute("G1", img_path, lambda: compute_G1(lesion, mask)))

    # NaN/inf guard
    feat_array = np.array(
        [
            B1, B2, B10, B11,
            MC1, MC2, MC4,
            R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13,
            G1,
        ],
        dtype=np.float64,
    )
    feat_array = np.nan_to_num(feat_array, nan=0.0, posinf=0.0, neginf=0.0)

    row: Dict[str, float] = {
        "ID": _extract_id_from_path(img_path),
        "label": int(label),
    }
    for name, val in zip(FEATURE_NAMES, feat_array.tolist()):
        row[name] = float(val)

    return row


def save_features(out_csv_path: str, row: Dict[str, float], writer, file_handle):
    """
    Write a single feature row to the CSV and flush immediately.
    This ensures progress is saved even if the script is interrupted.
    """
    try:
        writer.writerow(row)
        file_handle.flush()
    except Exception as e:
        print(f"[ERROR] Could not write features to {out_csv_path}: {e}")
        raise


# =====================================================================
# Multiprocessing worker
# =====================================================================

def _worker_compute(args: Tuple[str, float]):
    """
    Worker entrypoint for multiprocessing.

    Args:
        args: (img_path, label)

    Returns:
        ("ok", row, label) on success
        ("fail", img_path, label) on failure
    """
    img_path, label = args
    try:
        row = _compute_biomarkers_for_image(img_path, label)
        return ("ok", row, label)
    except Exception as e:
        # Per-biomarker errors already logged via _safe_compute;
        # this is just a top-level "this image failed" log in the worker.
        print(f"[WARN] Worker failed for {img_path}: {e}")
        return ("fail", img_path, label)


# =====================================================================
# Main compute function
# =====================================================================

def compute_and_save_biomarkers(
    data_root: str,
    out_csv_path: str,
    num_workers: int = 1,
) -> None:
    """
    Walks:

        data_root/
            benign/
            malignant/

    where data_root is expected to be the Kaggle train directory:
        melanoma_cancer_dataset/train

    Computes all biomarkers for each image and writes to a single CSV:

        out_csv_path

    Behavior:
        - Always overwrites out_csv_path if it exists.
        - Writes one row per image as soon as it's computed.
        - Logs which biomarker fails if a computation crashes.
        - Can use multiprocessing with adjustable num_workers.
    """
    fieldnames = ["ID", "label"] + FEATURE_NAMES

    # Overwrite existing file
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    f = open(out_csv_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    classes = [("benign", 0.0), ("malignant", 1.0)]

    # -----------------------------------------------------------------
    # Collect all (img_path, label) samples from training data
    # -----------------------------------------------------------------
    samples: List[Tuple[str, float]] = []

    for cls_name, label in classes:
        folder = os.path.join(data_root, cls_name)
        if not os.path.isdir(folder):
            print(f"[WARN] Missing folder: {folder}")
            continue

        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(folder, fname)
            samples.append((img_path, label))

    num_total = len(samples)
    if num_total == 0:
        print(f"[WARN] No images found under {data_root}")
        f.close()
        return

    print(
        f"[INFO] Found {num_total} training images "
        f"({sum(1 for _, l in samples if l == 0.0)} benign, "
        f"{sum(1 for _, l in samples if l == 1.0)} malignant)"
    )
    print(f"[INFO] Using num_workers={num_workers}")

    num_failed = 0
    num_benign = 0
    num_malignant = 0

    try:
        if num_workers <= 1:
            # ------------------ Serial mode ------------------
            for img_path, label in samples:
                try:
                    row = _compute_biomarkers_for_image(img_path, label)
                    save_features(out_csv_path, row, writer, f)

                    if label == 0.0:
                        num_benign += 1
                    else:
                        num_malignant += 1
                except Exception as e:
                    num_failed += 1
                    print(f"[WARN] Failed {img_path}: {e}")
        else:
            # ----------------- Multiprocessing -----------------
            # Use spawn-safe top-level worker. On Windows, this is required.
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=num_workers) as pool:
                # tune chunksize a bit
                chunksize = max(1, num_total // (num_workers * 4))
                for status, payload, label in pool.imap_unordered(
                    _worker_compute, samples, chunksize=chunksize
                ):
                    if status == "ok":
                        row = payload  # dict
                        save_features(out_csv_path, row, writer, f)
                        if label == 0.0:
                            num_benign += 1
                        else:
                            num_malignant += 1
                    else:
                        img_path = payload  # str
                        num_failed += 1
                        print(f"[WARN] Failed {img_path} (worker reported failure)")
    finally:
        f.close()

    print(
        f"[INFO] Biomarker computation complete.\n"
        f"       Total images seen (train): {num_total}\n"
        f"       Successful benign:         {num_benign}\n"
        f"       Successful malignant:      {num_malignant}\n"
        f"       Failed this run:           {num_failed}\n"
        f"       CSV path:                  {out_csv_path}"
    )


# =====================================================================
# CLI entry point (uses Kaggle import)
# =====================================================================

if __name__ == "__main__":
    import argparse
    import kagglehub

    parser = argparse.ArgumentParser(
        description="Compute melanoma dermoscopic biomarkers into features.csv"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes to use (1 = no multiprocessing)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./features.csv",
        help="Output CSV file path (will be overwritten)",
    )
    args = parser.parse_args()

    print("[INFO] Downloading Kaggle melanoma dataset via kagglehub...")
    base = kagglehub.dataset_download(
        "hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images"
    )
    train_root = os.path.join(base, "melanoma_cancer_dataset", "train")
    out_csv = args.out

    print(f"[INFO] Using training data root: {train_root}")
    print(f"[INFO] Writing features to:      {out_csv}")
    print(f"[INFO] num_workers:              {args.num_workers}")

    compute_and_save_biomarkers(
        train_root,
        out_csv,
        num_workers=max(1, args.num_workers),
    )
