"""Part 5 live run: condition C (show + tools) with the model's code in the container, next to the two unsandboxed C runs of Part 4.
ONE sandboxed run: it shows the sandbox works end to end, it does not measure accuracy (Part 4 showed one run is noise)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
runs = {"C (Part 4, run 1, unsandboxed)": "C", "C (Part 4, run 2, unsandboxed)": "Cr2", "C (Part 5, container)": "Csbx"}
print(f"{'set-up':34s} {'n':>2s} {'REVIEW':>6s} {'MAE pp':>7s} {'median':>7s} {'verdicts ok':>11s} {'false PASS':>10s} {'false FAIL':>10s} {'steps':>5s} {'tokens':>7s} {'sec':>5s}")
for name, tag in runs.items():
    rows = json.load(open(ROOT / f"out/run_dev4_{tag}.json"))["rows"]
    e = np.array([r["abs_err"] for r in rows if r["abs_err"] is not None])
    print(f"{name:34s} {len(rows):2d} {sum(r['agent_status'] == 'REVIEW' for r in rows):6d} {e.mean():7.2f} {np.median(e):7.2f} "
          f"{sum(r['agent_status'] == r['gt_status'] for r in rows):>8d}/{len(rows)} {sum(r['gt_status'] == 'FAIL' and r['agent_status'] == 'PASS' for r in rows):10d} "
          f"{sum(r['gt_status'] == 'PASS' and r['agent_status'] == 'FAIL' for r in rows):10d} {np.mean([r['steps'] for r in rows]):5.1f} {np.mean([r['tokens'] for r in rows]):7.0f} {np.mean([r['seconds'] for r in rows]):5.0f}")
rows = json.load(open(ROOT / "out/run_dev4_Csbx.json"))["rows"]
print("\nsandboxed run, per image:")
for r in rows:
    print(f"  {r['image']}  gt {r['gt_pct']:5.2f}  agent {('-' if r['agent_pct'] is None else format(r['agent_pct'], '.2f')):>6s}  {r['agent_status']:6s}  steps {r['steps']:2d}  {r['reason']}")
