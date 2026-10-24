"""Part 7: choose the contrast threshold of the `band` classifier on DEV (all 600 val images).

The pipeline's rules R1 (border) and R2 (area < 20) already drop two kinds of box. This sweep asks what
contrast floor the `band` classifier should use on top of them, scored by F1 on dev, before any held-out
image is looked at.
"""
import json
from pathlib import Path

import void_data as vd
from inspection_tools import find_candidates, measure_candidate, score_kept

OUT = Path(__file__).parent / "out"
GRID = (0.0, 5.0, 8.0, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0)

cache = []  # per image: (image, [(id, contrast, edge, area)])
for i in range(len(vd.split_ids("val"))):
    image = f"val_{i:05d}"
    ms = [measure_candidate(image, c["id"]) for c in find_candidates(image)["candidates"]]
    cache.append((image, [(m["candidate_id"], m["contrast"], m["edge_distance_px"], m["area_px"]) for m in ms]))

res = {}
for c in GRID:
    tp = fp = fn = 0
    for image, ms in cache:
        ids = [i for i, con, edge, area in ms if edge > 0 and area >= 20 and con >= c]
        t, f, n = score_kept(image, ids)
        tp, fp, fn = tp + t, fp + f, fn + n
    p, r = tp / (tp + fp), tp / (tp + fn)
    res[str(c)] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3)}
best = max(res, key=lambda k: res[k]["f1"])
(OUT / "band_dev_sweep.json").write_text(json.dumps({"grid": res, "best_contrast_floor": float(best)}, indent=2))
for k, v in res.items():
    print(f"contrast >= {k:>5}: P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f} (tp {v['tp']} fp {v['fp']})")
print("best on dev:", best)
