"""Parsing and refining GDXray defect ground truth."""
import pathlib

import numpy as np
from skimage.filters import threshold_otsu


def parse_gdxray_bboxes(txt_path: pathlib.Path) -> list[tuple[int, int, int, int]]:
    """Parse a GDXray ground-truth file into 0-indexed (row_min, col_min, row_max, col_max) boxes."""
    boxes = []
    for line in pathlib.Path(txt_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        _, row_min, col_min, row_max, col_max = (int(float(v)) for v in line.split())
        boxes.append((row_min - 1, col_min - 1, row_max - 1, col_max - 1))
    return boxes


def bbox_to_mask(image_shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Convert a (row_min, col_min, row_max, col_max) box into a boolean mask, inclusive bounds."""
    row_min, col_min, row_max, col_max = bbox
    mask = np.zeros(image_shape, dtype=bool)
    mask[row_min:row_max + 1, col_min:col_max + 1] = True
    return mask


def refine_mask_otsu(image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Split the ROI into two Otsu clusters and return the minority-area one as the defect mask.

    GDXray's shipped ground truth is a bounding box, not a pixel mask; defects are
    the minority-area intensity cluster within that box in every case observed in
    the Welds/Castings groups (a defect is a small region, not most of the box).
    """
    roi_values = image[roi_mask]
    threshold = threshold_otsu(roi_values)
    high_cluster = roi_mask & (image >= threshold)
    low_cluster = roi_mask & (image < threshold)
    return high_cluster if high_cluster.sum() <= low_cluster.sum() else low_cluster
