"""Wrap the Part 2 classical detector as a structured tool for LLM tool-calling."""
import json
import re
from pathlib import Path

import cv2

from baseline_detector import DATA_DIR, detect_candidates, load_ground_truth, score_image

IOU_HIT_THRESHOLD = 0.05
GT = load_ground_truth(DATA_DIR)


def _image_index(image_path: str) -> int | None:
    match = re.search(r"_(\d{4})\.png$", Path(image_path).name)
    return int(match.group(1)) if match else None


def resolve_image_path(image_path: str) -> Path:
    """Accept a full path or a bare C0001_XXXX.png filename under DATA_DIR."""
    path = Path(image_path)
    if path.exists():
        return path
    candidate = DATA_DIR / path.name
    if candidate.exists():
        return candidate
    return path


def inspect_casting_xray(image_path: str) -> str:
    """Run the classical top-hat defect detector from Part 2 on one GDXray casting frame.

    Args:
        image_path: Path to a GDXray Castings .png frame (e.g. C0001_0001.png).

    Returns:
        JSON string with candidate boxes, counts, and ground-truth comparison when available.
    """
    path = resolve_image_path(image_path)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return json.dumps({"error": f"Could not read image: {image_path}"})

    candidates = detect_candidates(gray)
    idx = _image_index(image_path)
    gt_boxes = GT.get(idx, []) if idx is not None else []
    tp, fp, fn = score_image(gt_boxes, candidates) if gt_boxes else (0, len(candidates), 0)

    payload = {
        "image": path.name,
        "image_index": idx,
        "image_size": {"width": int(gray.shape[1]), "height": int(gray.shape[0])},
        "ground_truth_boxes": len(gt_boxes),
        "candidate_boxes": len(candidates),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "candidates": [
            {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "width": int(x2 - x1),
                "height": int(y2 - y1),
            }
            for x1, y1, x2, y2 in candidates
        ],
        "iou_hit_threshold": IOU_HIT_THRESHOLD,
        "detector": "classical top-hat baseline (Part 2)",
    }
    return json.dumps(payload, indent=2)
