"""Shared constants and data utilities for the InstEditSeg reimplementation.

InstEditSeg (arXiv:2609.02004) reformulates medical segmentation as instruction-driven image
*editing*: instead of predicting a binary mask, a diffusion model renders a color-coded overlay
directly onto the original image, conditioned on a text instruction. This module builds that
"edited" target image from Kvasir-SEG's real (image, mask) pairs.
"""
import numpy as np
from PIL import Image

IMG_SIZE = 256
OVERLAY_COLOR = (255, 60, 60)   # red overlay, matches the paper's color-coded rendering
OVERLAY_ALPHA = 0.45
INSTRUCTION = "Highlight the polyp region in red."


def render_overlay(image: Image.Image, mask: Image.Image,
                    color=OVERLAY_COLOR, alpha=OVERLAY_ALPHA) -> Image.Image:
    """Composite a translucent color mask onto image wherever mask > 0 -- this composited
    image IS the diffusion model's generation target (not a separate mask channel)."""
    image = image.convert("RGB")
    mask_arr = np.array(mask.convert("L").resize(image.size)) > 127
    img_arr = np.array(image).astype(np.float32)
    overlay = np.array(color, dtype=np.float32)
    img_arr[mask_arr] = img_arr[mask_arr] * (1 - alpha) + overlay * alpha
    return Image.fromarray(img_arr.astype(np.uint8))


def extract_predicted_mask(generated: Image.Image, reference: Image.Image,
                            color=OVERLAY_COLOR, thresh=40) -> np.ndarray:
    """Rough inverse of render_overlay for evaluation: pixels that shifted toward `color`
    relative to the (approximate) un-overlaid reference are treated as the predicted region.
    This is an evaluation-time proxy, not something the model is directly trained to invert."""
    gen = np.array(generated.convert("RGB")).astype(np.float32)
    ref = np.array(reference.convert("RGB").resize(generated.size)).astype(np.float32)
    color = np.array(color, dtype=np.float32)
    diff_to_color = np.linalg.norm(gen - color, axis=-1)
    diff_from_ref = np.linalg.norm(gen - ref, axis=-1)
    return (diff_from_ref > thresh) & (diff_to_color < diff_from_ref)


def dice_score(pred_mask: np.ndarray, gt_mask: Image.Image) -> float:
    gt = np.array(gt_mask.convert("L").resize((pred_mask.shape[1], pred_mask.shape[0]))) > 127
    inter = np.logical_and(pred_mask, gt).sum()
    denom = pred_mask.sum() + gt.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0
