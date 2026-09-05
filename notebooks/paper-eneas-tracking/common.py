"""Shared utilities for the ENEAS reimplementation: real DAVIS-2017 video frames/masks, real
SAM2 point-prompted segmentation, and the identity-check helpers the semantic verification layer
is built from.
"""
import os
import numpy as np
import torch
from PIL import Image

DAVIS_ROOT = "data/DAVIS"
SEQUENCE = "gold-fish"   # 6 real, visually similar fish instances -- a real distractor-rich scene
TARGET_INSTANCE = 1
N_FRAMES = int(os.environ.get("N_FRAMES", 78))


def load_frame(idx: int) -> Image.Image:
    return Image.open(f"{DAVIS_ROOT}/JPEGImages/480p/{SEQUENCE}/{idx:05d}.jpg").convert("RGB")


def load_gt_mask(idx: int, instance: int = TARGET_INSTANCE) -> np.ndarray:
    arr = np.array(Image.open(f"{DAVIS_ROOT}/Annotations/480p/{SEQUENCE}/{idx:05d}.png"))
    return arr == instance


def mask_centroid(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (int(xs.mean()), int(ys.mean()))


def mask_bbox(mask: np.ndarray, pad: int = 10):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad)
    return (x0, y0, x1, y1)


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 1.0
