"""Is the sandbox transparent? Replay every cell the model wrote in the Part 4 runs, in a fresh sandbox per episode,
and compare with what the unsandboxed Jupyter kernel returned at the time. No model involved: the cells are the recorded ones.

Compared per cell: stdout (as recorded, first 2000 characters), whether it raised an error, the submission, and the number of pictures.
Run:  .venv/bin/python -I minimal_loop/replay_equivalence.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from docker_session import DockerSession

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT.parent / "agentic-factory-segmentation/data/void_dataset_v2/val.npz")
images = d["images"]
COND = {"A": (False, False), "B": (True, False), "C": (True, True)}
tot = same = 0
diffs = []
for tag in ("A", "Ar2", "B", "Br2", "C", "Cr2"):
    run = json.load(open(ROOT / f"out/run_dev4_{tag}.json"))
    show, tools = COND[tag[0]]
    n_cells = n_same = 0
    for t in run["traces"]:
        i = int(t["image"].split("_")[1])
        s = DockerSession(images[i], timeout_sec=60, with_show=show, with_tools=tools)
        try:
            for c in t["cells"]:
                if not c["code"]:
                    continue
                r = s.run(c["code"])
                sub_rec = re.search(r"\[submitted\] (\{.*\})", c["stdout"])
                ok_out = r.stdout[:2000] == c["stdout"]
                ok_err = bool(r.error) == bool(c["error"])
                ok_sub = (r.submission is not None) == bool(sub_rec) and (r.submission is None or json.dumps(r.submission, sort_keys=True) == json.dumps(json.loads(sub_rec.group(1)), sort_keys=True))
                ok_pic = len(r.images) == len(c.get("images", []))
                n_cells += 1
                if ok_out and ok_err and ok_sub and ok_pic:
                    n_same += 1
                else:
                    diffs.append((tag, t["image"], c["step"], ok_out, ok_err, ok_sub, ok_pic, c["stdout"][:160].replace("\n", " | "), r.stdout[:160].replace("\n", " | "), c["error"][-100:], r.error[-100:]))
        finally:
            s.close()
    tot += n_cells
    same += n_same
    print(f"{tag:4s} episodes={len(run['traces'])} cells={n_cells} identical={n_same}", flush=True)
print(f"\nTOTAL cells replayed: {tot}; identical in the sandbox: {same}; different: {tot - same}")
for x in diffs[:12]:
    print("DIFF", x)
