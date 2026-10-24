"""Part 7 video: zoomed crops of every candidate box, from the saved state.json (never re-runs the model).
Writes crop_<i>.png and crops_meta.json: where the box sits inside each crop, as fractions, so the video can mark it."""
import json
from pathlib import Path

import cv2

ASSETS = Path(__file__).parent / "video" / "assets" / "frame"
PAD = 16  # original pixels

state = json.loads((ASSETS / "state.json").read_text())
disp = cv2.imread(str(ASSETS / "frame.png"), cv2.IMREAD_GRAYSCALE)
H, W = disp.shape
S = W / state["size"][0]  # display scale (the saved image is upscaled)
meta = []
for i, c in enumerate(state["candidates"]):
    x1, y1, x2, y2 = (v * S for v in c["box"])
    cx0, cy0, cx1, cy1 = (int(v) for v in (max(0, x1 - PAD * S), max(0, y1 - PAD * S), min(W, x2 + PAD * S), min(H, y2 + PAD * S)))
    crop = disp[cy0:cy1, cx0:cx1]
    s = 260 / max(crop.shape[:2])
    cv2.imwrite(str(ASSETS / f"crop_{i}.png"), cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC))
    cw, ch = cx1 - cx0, cy1 - cy0
    meta.append({"fx1": (x1 - cx0) / cw, "fy1": (y1 - cy0) / ch, "fx2": (x2 - cx0) / cw, "fy2": (y2 - cy0) / ch})
(ASSETS / "crops_meta.json").write_text(json.dumps(meta, indent=1))
print(f"{len(meta)} crops written")
