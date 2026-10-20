"""Context for the Part 3 numbers: what do trivial rules score on the SAME 12 DEV images?
Ground truth is read here (scoring only). The 'constant' is the median void ratio of ALL 600 DEV images, not of the 12.
Run:  .venv/bin/python -I minimal_loop/trivial_baselines.py
"""
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
d = np.load(root.parent / "agentic-factory-segmentation/data/void_dataset_v2/val.npz")
dm, pm = d["defect_masks"], d["part_masks"]
gt_all = np.array([100.0 * (dm[i] == 1).sum() / pm[i].sum() for i in range(len(dm))])
run = json.load(open(root / "out/run_dev.json")); L = run["limit_pct"]
gt = np.array([r["gt_pct"] for r in run["rows"]]); n = len(gt)
agent = np.array([r["agent_pct"] for r in run["rows"]])
const = float(np.median(gt_all))
gt_fail = gt > L
print(f"n={n}  limit={L}%  ground-truth FAIL count: {int(gt_fail.sum())}  PASS count: {int((~gt_fail).sum())}")
print(f"constant guess (median of all 600 DEV images = {const:.2f}%): mean abs error {np.abs(gt - const).mean():.2f} pp, median {np.median(np.abs(gt - const)):.2f}")
print(f"agent:                                              mean abs error {np.abs(gt - agent).mean():.2f} pp, median {np.median(np.abs(gt - agent)):.2f}")
print(f"'always PASS': agrees on {int((~gt_fail).sum())} of {n}, false PASS {int(gt_fail.sum())}, false FAIL 0")
ag_fail = agent > L
print(f"agent:         agrees on {int((ag_fail == gt_fail).sum())} of {n}, false PASS {int((gt_fail & ~ag_fail).sum())}, false FAIL {int((~gt_fail & ag_fail).sum())}")
print(f"rank correlation agent vs truth (Spearman): {np.corrcoef(np.argsort(np.argsort(agent)), np.argsort(np.argsort(gt)))[0, 1]:.2f}")
