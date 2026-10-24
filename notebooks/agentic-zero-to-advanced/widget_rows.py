"""Part 7 widget data: every detector box on the first 200 dev (val) and first 200 held-out (test) images,
as [split (0 dev, 1 held-out), contrast, area_px, edge_distance_px, real (0/1)]. Real = the box hits an exact void."""
import json
from pathlib import Path

from baseline_detector import match
from inspection_tools import N_PER_SPLIT, exact_voids, find_candidates, measure_candidate

rows, gt = [], [0, 0]
for s, split in enumerate(("val", "test")):
    for i in range(N_PER_SPLIT):
        image = f"{split}_{i:05d}"
        cands = find_candidates(image)["candidates"]
        voids = exact_voids(image)
        gt[s] += len(voids)
        _, mc = match(voids, [tuple(c["box"]) for c in cands])
        for c in cands:
            m = measure_candidate(image, c["id"])
            rows.append([s, m["contrast"], m["area_px"], m["edge_distance_px"], int(c["id"] in mc)])
Path(__file__).parent.joinpath("out/widget_rows.json").write_text(json.dumps({"rows": rows, "gt": gt}))
print(len(rows), "boxes; real voids per split (dev, held-out):", gt, "; real boxes:", [sum(r[4] for r in rows if r[0] == s) for s in (0, 1)])
