"""Part 5 sanity check: can ANY of the measurements the agent sees tell a real defect from a
false alarm? AUC 0.5 = coin flip. Uses real GDXray ground truth, all 72 frames."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from baseline_detector import IOU_HIT_THRESHOLD, iou
from inspection_tools import DATA_DIR, GT, find_candidates, image_index, measure_candidate

rows, hits = [], []
for p in sorted(DATA_DIR.glob("C0001_*.png")):
    gt = GT.get(image_index(p.name), [])
    for c in find_candidates(p.name)["candidates"]:
        m = measure_candidate(p.name, c["id"])
        rows.append([m["contrast"], abs(m["contrast"]), m["area_px"], m["aspect_ratio"], m["edge_distance_px"]])
        hits.append(any(iou(g, tuple(c["box"])) >= IOU_HIT_THRESHOLD for g in gt))
X, y = np.array(rows, dtype=float), np.array(hits)
out = {"n_candidates": int(len(y)), "n_real": int(y.sum())}
for i, name in enumerate(["contrast", "abs_contrast", "area_px", "aspect_ratio", "edge_distance_px"]):
    auc = mannwhitneyu(X[y, i], X[~y, i]).statistic / (y.sum() * (~y).sum())
    out[name] = {"auc": round(float(auc), 3), "median_real": float(np.median(X[y, i])),
                 "median_false_alarm": float(np.median(X[~y, i]))}
Path(__file__).parent.joinpath("out/feature_check.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
