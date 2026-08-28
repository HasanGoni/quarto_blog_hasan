"""Parsing and refining GDXray defect ground truth."""
import pathlib

import numpy as np
from skimage.filters import threshold_otsu


def parse_gdxray_bboxes(txt_path: pathlib.Path) -> dict[int, list[tuple[int, int, int, int]]]:
    """Parse a GDXray ground-truth file into per-image 0-indexed (row_min, col_min, row_max, col_max) boxes.

    Returns a dict keyed by each line's leading `image_id` column, mapping to that
    image's list of boxes. Keeping the image_id is the point: a flat list of boxes
    only lines up with a series' images by the accident of the ground-truth file
    being sorted by image_id, and a silent image/box mismatch is very hard to spot
    downstream. Callers select the boxes for the image they actually loaded --
    for a GDXray series, file `W0001_000N.png` is image_id N.

    Real GDXray ground-truth lines are `image_id col_min col_max row_min row_max`
    (verified against the real Welds group, series W0001: grouping ground_truth.txt's
    641 lines by their leading image_id column into the 10 images of that series,
    then checking each image's coordinate pairs against that image's actual
    (width, height) -- read from the corresponding real PNG -- shows the first
    pair's max value tracks image width and the second pair's max value tracks
    image height, for every one of those 10 images). The other two candidate
    orderings (treating the columns as row-first, or as unpaired) both put
    coordinate values far outside the image bounds.
    """
    boxes: dict[int, list[tuple[int, int, int, int]]] = {}
    for line in pathlib.Path(txt_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        image_id, col_min, col_max, row_min, row_max = (int(float(v)) for v in line.split())
        boxes.setdefault(image_id, []).append((row_min - 1, col_min - 1, row_max - 1, col_max - 1))
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
