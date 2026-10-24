"""Part 5 sanity check: can ANY of the measurements the agent sees tell a real void from a
false alarm? AUC 0.5 = coin flip. Uses the exact voids of the DEV split (val, all 600 images);
the held-out split is not touched, because this check is what the agent's guidance is written from."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import void_data as vd
from baseline_detector import match
from inspection_tools import exact_voids, find_candidates, measure_candidate

rows, hits = [], []
for i in range(len(vd.split_ids("val"))):
    image = f"val_{i:05d}"
    cands = find_candidates(image)["candidates"]
    _, mc = match(exact_voids(image), [tuple(c["box"]) for c in cands])
    for c in cands:
        m = measure_candidate(image, c["id"])
        rows.append([m["contrast"], abs(m["contrast"]), m["area_px"], m["aspect_ratio"], m["edge_distance_px"]])
        hits.append(c["id"] in mc)
X, y = np.array(rows, dtype=float), np.array(hits)
out = {"split": "val (dev)", "n_candidates": int(len(y)), "n_real": int(y.sum())}
for i, name in enumerate(["contrast", "abs_contrast", "area_px", "aspect_ratio", "edge_distance_px"]):
    auc = mannwhitneyu(X[y, i], X[~y, i]).statistic / (y.sum() * (~y).sum())
    out[name] = {"auc": round(float(auc), 3), "median_real": float(np.median(X[y, i])),
                 "median_false_alarm": float(np.median(X[~y, i]))}
Path(__file__).parent.joinpath("out/feature_check.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
