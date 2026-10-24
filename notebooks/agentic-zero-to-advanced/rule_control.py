"""Part 5 control: the same dev findings as a plain rule, no model. Which candidates would it keep?

Rule (thresholds read off the DEV bands in agent_fixes.py, not tuned on test): keep a box if it does
not touch the image border, has area >= 20 px and contrast >= 20. Scored on the same first-200 images
of each split the agents were scored on.
"""
import json
from pathlib import Path

from inspection_tools import N_PER_SPLIT, find_candidates, measure_candidate, score_kept

OUT = Path(__file__).parent / "out"


def keep(m: dict, min_contrast: float = 20.0) -> bool:
    return m["edge_distance_px"] > 0 and m["area_px"] >= 20 and m["contrast"] >= min_contrast


def run(split: str, min_contrast: float = 20.0) -> dict:
    tp = fp = fn = kept = 0
    for i in range(N_PER_SPLIT):
        image = f"{split}_{i:05d}"
        ids = [c["id"] for c in find_candidates(image)["candidates"]
               if keep(measure_candidate(image, c["id"]), min_contrast)]
        t, f, n = score_kept(image, ids)
        tp, fp, fn, kept = tp + t, fp + f, fn + n, kept + len(ids)
    return {"split": split, "kept": kept, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp / (tp + fp), 3), "recall": round(tp / (tp + fn), 3)}


if __name__ == "__main__":
    out = {"dev": run("val"), "test": run("test"),
           "test_sweep": {str(c): run("test", c) for c in (5.0, 8.0, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0)}}
    (OUT / "rule_control.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
