"""Tools for the Arc 2 agent posts. Same Part 2 detector, but split into small tools the
agent can call one at a time, and with NO ground truth in any tool result (Part 3's
`inspect_casting_xray` also returned the TP/FP counts, which an honest agent must not see).
Ground truth is only used by the scoring code, after the agent is done.
"""
import cv2
import numpy as np

from baseline_detector import DATA_DIR, contrast_delta, detect_candidates, load_ground_truth, score_image
from detector_tool import resolve_image_path

GT = load_ground_truth(DATA_DIR)


def image_index(name: str) -> int:
    return int(name.split("_")[-1].split(".")[0])


def load_gray(image: str) -> np.ndarray:
    gray = cv2.imread(str(resolve_image_path(image)), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image)
    return gray


# Tool 1: find candidate boxes. Cached per image so measure_candidate can refer to them by id.
_CANDIDATES: dict[str, list] = {}


def find_candidates(image: str) -> dict:
    """Run the classical detector on one X-ray frame. Returns candidate boxes with an id each.
    The detector is noisy: most boxes are not real defects."""
    gray = load_gray(image)
    boxes = detect_candidates(gray)
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
    gray = load_gray(image)
    boxes = _CANDIDATES.get(image) or find_candidates(image) and _CANDIDATES[image]
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
        "parameters": {"type": "object", "properties": {"image": {"type": "string", "description": "frame name, e.g. C0001_0001.png"}}, "required": ["image"]}}},
    {"type": "function", "function": {
        "name": "measure_candidate",
        "description": measure_candidate.__doc__,
        "parameters": {"type": "object", "properties": {
            "image": {"type": "string"}, "candidate_id": {"type": "integer"}}, "required": ["image", "candidate_id"]}}},
]


def score_kept(image: str, kept_ids: list[int]) -> tuple[int, int, int]:
    """(tp, fp, fn) for the boxes the agent decided to keep, against real ground truth."""
    boxes = _CANDIDATES.get(image) or []
    kept = [boxes[i] for i in kept_ids if 0 <= i < len(boxes)]
    return score_image(GT.get(image_index(image), []), kept)
