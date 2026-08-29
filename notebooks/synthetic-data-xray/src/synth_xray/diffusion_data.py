"""Real (image, mask) defect-inpainting training pairs from GDXray ground truth.

Reuses Part 1's exact data/ground-truth utilities (`data.py`, `groundtruth.py`) --
same real Welds series, same box-to-mask logic -- so Part 2's LoRA fine-tune trains
on the identical real defects Part 1's physics pipeline was calibrated against.
"""
from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

from synth_xray.data import find_groundtruth_file, find_images_in_series
from synth_xray.groundtruth import bbox_to_mask, parse_gdxray_bboxes, refine_mask_otsu

# SD-inpainting's native resolution -- cropping directly at this size avoids any
# resize step between the real pixel data and what the model actually trains on.
# These weld panoramas (e.g. 835x4919) are comfortably larger than 512 in both
# dimensions, so this never needs to upsample.
CROP_SIZE = 512
# Margin (px) a defect box must fit inside of, so the crop always has real
# non-defect context surrounding the mask, not just the defect itself.
PAD = 40


def image_id_from_path(path: pathlib.Path) -> int:
    """Map a GDXray series image filename to its ground-truth `image_id` column.

    Same convention verified in `generate_examples.py`: `<SERIES>_<NNNN>.png` -> N.
    """
    return int(path.stem.split("_")[-1])


def load_gray(path: pathlib.Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float64)


def crop_around_box(
    image: np.ndarray, box: tuple[int, int, int, int], crop_size: int = CROP_SIZE
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop a fixed-size square window centered on a defect box, clamped to image bounds."""
    row_min, col_min, row_max, col_max = box
    cy, cx = (row_min + row_max) // 2, (col_min + col_max) // 2
    half = crop_size // 2
    r0 = max(0, min(cy - half, image.shape[0] - crop_size))
    c0 = max(0, min(cx - half, image.shape[1] - crop_size))
    return image[r0 : r0 + crop_size, c0 : c0 + crop_size], (r0, c0)


def build_defect_dataset(series_dir: pathlib.Path) -> list[dict]:
    """Real (image crop, defect mask) pairs for every ground-truth box in a GDXray series.

    Each returned dict: {"image": float64 (CROP_SIZE, CROP_SIZE) grayscale array,
    "mask": bool (CROP_SIZE, CROP_SIZE) array, "image_id": int, "box": original bbox}.
    """
    gt_file = find_groundtruth_file(series_dir)
    if gt_file is None:
        return []
    boxes_by_image = parse_gdxray_bboxes(gt_file)

    examples: list[dict] = []
    for img_path in find_images_in_series(series_dir):
        image_id = image_id_from_path(img_path)
        boxes = boxes_by_image.get(image_id, [])
        if not boxes:
            continue
        image = load_gray(img_path)
        if image.shape[0] < CROP_SIZE or image.shape[1] < CROP_SIZE:
            continue  # panorama too small for a full-resolution crop

        for box in boxes:
            row_min, col_min, row_max, col_max = box
            if row_min < 0 or col_min < 0:
                # A handful of real GDXray boxes come out as -1 on one edge (the
                # raw annotation sits at the image boundary, and the 1-indexed ->
                # 0-indexed `- 1` in parse_gdxray_bboxes takes an already-0 raw
                # value to -1). A negative slice start wraps around in Python and
                # silently produces an empty (or wrong) mask rather than raising,
                # so this has to be caught explicitly rather than left to fail
                # downstream.
                continue
            if (row_max - row_min) < 2 or (col_max - col_min) < 2:
                continue  # degenerate/annotation-noise box
            if (row_max - row_min) > CROP_SIZE - 2 * PAD or (col_max - col_min) > CROP_SIZE - 2 * PAD:
                continue  # defect too large to sit inside a padded crop

            crop, (r0, c0) = crop_around_box(image, box, CROP_SIZE)
            local_box = (row_min - r0, col_min - c0, row_max - r0, col_max - c0)
            roi_mask = bbox_to_mask(crop.shape, local_box)
            if roi_mask.sum() == 0:
                continue  # box fell outside the crop window
            defect_mask = refine_mask_otsu(crop, roi_mask)
            if defect_mask.sum() < 4:
                continue  # Otsu split degenerated to almost nothing

            examples.append({"image": crop, "mask": defect_mask, "image_id": image_id, "box": box})

    return examples


def to_rgb_pil(gray: np.ndarray) -> Image.Image:
    """Grayscale float array -> 3-channel RGB PIL image (SD expects 3 input channels)."""
    normalized = np.clip(gray, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([normalized] * 3, axis=-1))


def to_mask_pil(mask: np.ndarray) -> Image.Image:
    """Boolean mask -> single-channel 'L' PIL image (0 = keep, 255 = inpaint)."""
    return Image.fromarray((mask.astype(np.uint8) * 255))


def to_scribble_pil(mask: np.ndarray) -> Image.Image:
    """Real defect mask -> a thin-boundary scribble image for ControlNet shape conditioning.

    `control_v11p_sd15_scribble` was trained on thin sketch/edge-like inputs (HED soft edges,
    binarized), not filled regions -- so this traces the mask's *boundary* (mask minus its
    erosion), white line on black, rather than passing the filled blob through directly. This is
    the real shape of the real defect, not a synthetic crack or a bare rectangle: the whole point
    of Part 3 is testing whether that's what plain mask-conditioning (Part 2) was missing.
    """
    boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3), bool))
    rgb = np.stack([boundary.astype(np.uint8) * 255] * 3, axis=-1)
    return Image.fromarray(rgb)
