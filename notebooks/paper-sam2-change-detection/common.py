"""Shared utilities: simulated point supervision from real LEVIR-CD change masks, and the
bi-temporal selection rule that turns SAM2's point-prompted segmentations into a change
pseudo-label with a per-point confidence (used as the uncertainty weight downstream).
"""
import numpy as np

IMG_SIZE = 256
N_POINTS = 3  # simulated point clicks per image pair, sampled from the real change mask


def sample_points(mask: np.ndarray, n_points: int, rng: np.random.Generator):
    """Simulates the point annotations a human would click: n_points random locations
    inside the real ground-truth changed region. This is the ONLY use of ground truth during
    pseudo-labeling -- standing in for a human's clicks, not used as a training signal itself."""
    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        return []
    idx = rng.choice(len(ys), size=min(n_points, len(ys)), replace=False)
    return [(int(xs[i]), int(ys[i])) for i in idx]


def iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(inter / union) if union > 0 else 1.0


def bi_temporal_pseudo_label(mask_a: np.ndarray, mask_b: np.ndarray):
    """Given SAM2's segmentation of the SAME point on the before (A) and after (B) image:
    - candidate change region = mask_b (what's there now, at a point we know changed)
    - confidence = 1 - IoU(mask_a, mask_b): high when A and B disagree a lot at that point
      (strong evidence something really changed there), low when they roughly agree (the point
      likely wasn't a genuine change, or SAM2 segmented the same underlying object both times).
    """
    conf = 1.0 - iou(mask_a, mask_b)
    return mask_b, conf
