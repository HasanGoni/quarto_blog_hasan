"""Wrap the Part 2 classical detector as a structured tool for LLM tool-calling.

The tool returns what a deployed detector would know: boxes and how strong each response is.
It does NOT return ground truth (a production tool has no answer key); scoring happens outside.
"""
import json
import re

import numpy as np

import void_data as vd
from baseline_detector import K_DEV, detect_candidates

_ID = re.compile(r"^(val|test|train)_(\d{5})$")


def parse_image_id(image_id: str):
    m = _ID.match(image_id.strip().removesuffix(".png"))
    return (m.group(1), int(m.group(2))) if m else None


def inspect_xray(image_id: str) -> str:
    """Run the classical top-hat void detector from Part 2 on one X-ray image.

    Args:
        image_id: Image id such as test_00009 (split name, underscore, five digits).

    Returns:
        JSON string with the candidate boxes (pixels), each with its size and strength, where
        strength is the peak response divided by the image's noise level.
    """
    parsed = parse_image_id(image_id)
    if parsed is None:
        return json.dumps({"error": f"Unknown image id: {image_id}. Expected e.g. test_00009."})
    split, i = parsed
    gray = vd.load_image(split, i)
    boxes, st = detect_candidates(gray, K_DEV, return_stages=True)
    sigma = max(st["sigma"], 1e-3)
    cands = []
    for x1, y1, x2, y2 in boxes:
        peak = float(st["response"][y1:y2, x1:x2].max()) / sigma
        cands.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": x2 - x1, "height": y2 - y1, "strength": round(peak, 1),
        })
    return json.dumps({
        "image": f"{split}_{i:05d}",
        "image_size": {"width": int(gray.shape[1]), "height": int(gray.shape[0])},
        "part_area_px": int((st["mask"] > 0).sum()),
        "candidate_boxes": len(cands),
        "candidates": cands,
        "threshold": f"{K_DEV} x noise level",
        "detector": "classical top-hat baseline (Part 2)",
    }, indent=2)
