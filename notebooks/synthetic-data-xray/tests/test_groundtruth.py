import numpy as np
import pytest
from synth_xray.groundtruth import parse_gdxray_bboxes, bbox_to_mask, refine_mask_otsu


def test_parse_gdxray_bboxes(tmp_path):
    # Real GDXray lines are `image_id col_min col_max row_min row_max` -- these
    # rows use distinct row/col spans (100x40 vs 8x40) so a column-order mixup
    # would produce a mismatched shape, not just a coincidentally-equal box.
    gt_file = tmp_path / "ground_truth_C0001.txt"
    gt_file.write_text("1 10 110 20 60\n1 5 13 30 70\n2 1 5 1 5\n")
    boxes = parse_gdxray_bboxes(gt_file)
    assert boxes == [(19, 9, 59, 109), (29, 4, 69, 12), (0, 0, 4, 4)]


def test_bbox_to_mask_shape_and_bounds():
    mask = bbox_to_mask((50, 50), (9, 9, 19, 19))
    assert mask.shape == (50, 50)
    assert mask.dtype == bool
    assert mask[9:20, 9:20].all()
    assert mask.sum() == 11 * 11
    assert not mask[8, 9]
    assert not mask[9, 8]


def test_refine_mask_otsu_recovers_bright_blob():
    image = np.full((40, 40), 100.0)
    roi = bbox_to_mask((40, 40), (10, 10, 29, 29))
    image[15:25, 15:25] = 220.0  # small bright defect inside the ROI
    refined = refine_mask_otsu(image, roi)
    assert refined.shape == (40, 40)
    assert refined.dtype == bool
    assert refined[18, 18]
    assert not refined[11, 11]
    assert not refined.any() or refined[roi].sum() < roi.sum()
