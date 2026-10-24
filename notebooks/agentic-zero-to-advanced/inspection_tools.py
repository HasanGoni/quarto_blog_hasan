"""Tools for the Arc 2 agent posts. Same Part 2 detector, but split into small tools the
agent can call one at a time, and with NO ground truth in any tool result (an honest agent must
not see the answer key). Ground truth is only used by the scoring code, after the agent is done.
"""
import cv2
import numpy as np

import void_data as vd
from baseline_detector import K_DEV, detect_candidates, match
from detector_tool import parse_image_id

# Evaluation size: the first N images of each split, in order (not picked by outcome).
N_PER_SPLIT = 200


def norm(image: str) -> str:
    parsed = parse_image_id(image)
    if parsed is None:
        raise FileNotFoundError(f"unknown image id {image!r}, expected e.g. test_00009")
    return f"{parsed[0]}_{parsed[1]:05d}"


def load_gray(image: str) -> np.ndarray:
    split, i = parse_image_id(norm(image))
    return vd.load_image(split, i)


def contrast_delta(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad: int = 8) -> float:
    """Mean intensity inside the box minus mean of a padded ring around it."""
    h, w = gray.shape
    inner = gray[y1:y2, x1:x2].astype(np.float32)
    y0, y1p, x0, x1p = max(0, y1 - pad), min(h, y2 + pad), max(0, x1 - pad), min(w, x2 + pad)
    ring = gray[y0:y1p, x0:x1p].astype(np.float32).copy()
    ring[y1 - y0 : y2 - y0, x1 - x0 : x2 - x0] = np.nan
    return float(inner.mean() - np.nanmean(ring))


# Tool 1: find candidate boxes. Cached per image so measure_candidate can refer to them by id.
_CANDIDATES: dict[str, list] = {}


def find_candidates(image: str) -> dict:
    """Run the classical detector on one X-ray image. Returns candidate boxes with an id each.
    The detector is noisy: most boxes are not real voids."""
    image = norm(image)
    gray = load_gray(image)
    boxes = detect_candidates(gray, K_DEV)
    _CANDIDATES[image] = boxes
    return {
        "image": image,
        "image_size": [int(gray.shape[1]), int(gray.shape[0])],
        "candidates": [
            {"id": i, "box": [int(a) for a in b], "w": int(b[2] - b[0]), "h": int(b[3] - b[1])}
            for i, b in enumerate(boxes)
        ],
    }


# Tool 2: look closer at one candidate.
def measure_candidate(image: str, candidate_id: int) -> dict:
    """Measure one candidate box: contrast against its surrounding ring (positive = brighter than
    surroundings, negative = darker), area, aspect ratio, and distance to the nearest image edge."""
    image = norm(image)
    gray = load_gray(image)
    boxes = _CANDIDATES.get(image)
    if boxes is None:
        find_candidates(image)
        boxes = _CANDIDATES[image]
    if not 0 <= candidate_id < len(boxes):
        return {"error": f"candidate_id {candidate_id} out of range (0..{len(boxes) - 1})"}
    x1, y1, x2, y2 = boxes[candidate_id]
    h, w = gray.shape
    return {
        "candidate_id": candidate_id,
        "contrast": round(contrast_delta(gray, x1, y1, x2, y2), 2),
        "area_px": int((x2 - x1) * (y2 - y1)),
        "aspect_ratio": round((x2 - x1) / max(1, (y2 - y1)), 2),
        "edge_distance_px": int(min(x1, y1, w - x2, h - y2)),
    }


TOOL_FUNCS = {"find_candidates": find_candidates, "measure_candidate": measure_candidate}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "find_candidates",
        "description": find_candidates.__doc__,
        "parameters": {"type": "object", "properties": {"image": {"type": "string", "description": "image id, e.g. test_00009"}}, "required": ["image"]}}},
    {"type": "function", "function": {
        "name": "measure_candidate",
        "description": measure_candidate.__doc__,
        "parameters": {"type": "object", "properties": {
            "image": {"type": "string"}, "candidate_id": {"type": "integer"}}, "required": ["image", "candidate_id"]}}},
]


def exact_voids(image: str) -> list[tuple]:
    split, i = parse_image_id(norm(image))
    return [tuple(v["bbox"]) for v in vd.load_voids(split, i)]


def score_kept(image: str, kept_ids: list[int]) -> tuple[int, int, int]:
    """(tp, fp, fn) for the boxes the agent decided to keep, against the exact void boxes."""
    image = norm(image)
    boxes = _CANDIDATES.get(image)
    if boxes is None:
        find_candidates(image)
        boxes = _CANDIDATES[image]
    kept = [boxes[i] for i in kept_ids if 0 <= i < len(boxes)]
    gt = exact_voids(image)
    mg, mc = match(gt, kept)
    return len(mg), len(kept) - len(mc), len(gt) - len(mg)
