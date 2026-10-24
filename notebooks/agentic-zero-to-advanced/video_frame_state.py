"""Part 7 video: run the guided pipeline once on one held-out frame and save everything the video shows
(boxes, measurements, the model's labels, the verdict, the note, the answer key) plus display images.
Image choice (stated in the post): the first held-out image from index 200 on (outside the 200 scored in the table) with
5-7 candidates, at least one hit, at least one miss and at least two boxes the rules would drop.
The video only ever draws what this script saved, so every number on screen is a real output."""
import json

import cv2
import numpy as np

import pipeline_graph as pl
from baseline_detector import match
from inspection_tools import exact_voids, load_gray

IMAGE = "test_00204"
ASSETS = pl.OUT_DIR.parent / "video" / "assets" / "frame"
ASSETS.mkdir(parents=True, exist_ok=True)

graph = pl.build_pipeline("llm")
final = graph.invoke(pl.initial_state(IMAGE))
gray = load_gray(IMAGE)
gt = [list(map(float, g)) for g in exact_voids(IMAGE)]
boxes = [tuple(c["box"]) for c in final["candidates"]]
mg, mc = match([tuple(g) for g in exact_voids(IMAGE)], boxes)
state = {"image": IMAGE, "size": [int(gray.shape[1]), int(gray.shape[0])],
         "candidates": final["candidates"], "measurements": final["measurements"], "labels": final["labels"],
         "kept": final["kept"], "rejected": final["rejected"], "verdict": final["verdict"], "reason": final["reason"],
         "summary": final["report"]["summary"], "summary_source": final["report"]["summary_source"],
         "llm_calls": final["llm_calls"], "tokens": final["tokens"], "trace": final["trace"],
         "answer_key": gt,
         "real_ids": sorted(mc),
         "gt_hit": [gi in mg for gi in range(len(gt))]}
(ASSETS / "state.json").write_text(json.dumps(state, indent=1, default=int))

# display copy: percentile stretch so the faint defects are visible on screen (the model never sees pixels)
lo, hi = np.percentile(gray, [2, 98])
disp = np.clip((gray.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
disp = cv2.resize(disp, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)  # 224 px is small on screen
cv2.imwrite(str(ASSETS / "frame.png"), disp)
# zoom crops: see video_frame_crops.py (reads state.json, so it never re-runs the model)
print(json.dumps({k: state[k] for k in ("verdict", "reason", "llm_calls", "tokens", "real_ids", "gt_hit", "size")}))
for m, l in zip(state["measurements"], state["labels"]):
    print(m["candidate_id"], m["contrast"], m["area_px"], m["edge_distance_px"], "|", l["label"], l["confidence"])
print("kept", [(k["id"], k["severity"], k["uncertain"]) for k in state["kept"]], "rejected", state["rejected"])
print(state["summary"])
