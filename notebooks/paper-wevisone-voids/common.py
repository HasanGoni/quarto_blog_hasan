"""Shared pieces for the WeVisOne-VL void experiment: data access, box parsing, and the series' scoring rule.

The scoring is the one every part of "Agentic Developer: Zero to Advanced" uses (baseline_detector.match): a predicted box
hits a real void when their IoU is at least 0.2, one-to-one, greedy by IoU. The voids are tiny (a typical box is 8 to 17
pixels on a 224 pixel image), so 0.2 is the series' rule; `HIT_STRICT` = 0.5 is reported next to it as a harder check.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "agentic-factory-segmentation" / "data" / "void_dataset_v1"
OUT = HERE / "out"
HIT = 0.2
HIT_STRICT = 0.5
LABEL = "void"

_arrays: dict = {}
_voids: dict | None = None


def split_array(split: str) -> np.ndarray:
    if split not in _arrays:
        _arrays[split] = np.load(DATA_DIR / f"{split}.npz")["images"]
    return _arrays[split]


def load_image(split: str, i: int, scale: int = 1) -> Image.Image:
    """The 224x224 grayscale X-ray as RGB. `scale` enlarges it (bicubic) so the vision encoder gets more tokens per void."""
    img = Image.fromarray(split_array(split)[i]).convert("RGB")
    return img.resize((img.width * scale, img.height * scale), Image.BICUBIC) if scale != 1 else img


def gt_boxes(split: str, i: int) -> list[list[int]]:
    global _voids
    if _voids is None:
        _voids = {}
        for line in (DATA_DIR / "voids.jsonl").read_text().splitlines():
            v = json.loads(line)
            _voids.setdefault(v["id"], []).append(v["bbox"])
    return _voids.get(f"{split}_{i:05d}", [])


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt, pred, thr: float = HIT):
    """Greedy one-to-one match, copied from the series' baseline_detector.match so the numbers are comparable."""
    pairs = sorted(((iou(g, c), gi, ci) for gi, g in enumerate(gt) for ci, c in enumerate(pred)), reverse=True)
    mg, mc = set(), set()
    for s, gi, ci in pairs:
        if s < thr:
            break
        if gi in mg or ci in mc:
            continue
        mg.add(gi)
        mc.add(ci)
    return mg, mc


def score(gt, pred, thr: float = HIT) -> tuple[int, int, int]:
    mg, mc = match(gt, pred, thr)
    return len(mg), len(pred) - len(mc), len(gt) - len(mg)


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": None if p is None else round(p, 3),
            "recall": None if r is None else round(r, 3), "f1": round(f, 3)}


def parse_boxes(response: str, width: int, height: int, scale: int = 1) -> list[list[float]]:
    """WeVisOne's own parser (<ref>label</ref><box><x1><y1><x2><y2>,...</box>), boxes returned in ORIGINAL 224-pixel coordinates."""
    from wevisone_vl.visualization import parse_predictions
    boxes = []
    for items in parse_predictions(response, width, height).values():
        boxes += [[c / scale for c in it["coords"]] for it in items if it["type"] == "box"]
    return boxes


def bootstrap_ci(per_image: list[tuple[int, int, int]], key: str, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval over resampled IMAGES (the unit that was drawn), for precision, recall or f1."""
    rng, vals = random.Random(seed), []
    for _ in range(n):
        s = [per_image[rng.randrange(len(per_image))] for _ in per_image]
        r = prf(sum(x[0] for x in s), sum(x[1] for x in s), sum(x[2] for x in s))[key]
        if r is not None:
            vals.append(r)
    vals.sort()
    return round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals))], 3)
