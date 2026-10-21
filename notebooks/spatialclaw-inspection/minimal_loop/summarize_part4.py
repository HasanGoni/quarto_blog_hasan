"""Part 4 summary: the same 12 DEV images under four set-ups. Reads only saved run files. Part 3's row is its saved run (8 turns)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
runs = {"P3 (text, 8 turns)": json.load(open(ROOT / "out/run_dev.json"))}
for c, name in (("A", "A (text, 14 turns)"), ("B", "B (+ show)"), ("C", "C (+ show + tools)")):
    runs[name] = json.load(open(ROOT / f"out/run_dev4_{c}.json"))
    runs[name.replace(")", ", run 2)")] = json.load(open(ROOT / f"out/run_dev4_{c}r2.json"))
LIM = 1.5


def stats(data):
    rows = data["rows"]
    mx = data.get("max_steps", 8)
    done = [r for r in rows if r["abs_err"] is not None]
    e = np.array([r["abs_err"] for r in done])
    return dict(n=len(rows), review=sum(r["agent_status"] == "REVIEW" for r in rows), mae=e.mean(), med=np.median(e), mx=e.max(),
                agree=sum(r["agent_status"] == r["gt_status"] for r in rows),
                fpass=sum(r["gt_status"] == "FAIL" and r["agent_status"] == "PASS" for r in rows),
                ffail=sum(r["gt_status"] == "PASS" and r["agent_status"] == "FAIL" for r in rows),
                steps=np.mean([r["steps"] for r in rows]), last=sum(r["steps"] >= mx - 1 for r in rows), tokens=np.mean([r["tokens"] for r in rows]),
                sec=np.mean([r["seconds"] for r in rows]), pics=sum(r.get("pictures", 0) for r in rows), big=sum(r["abs_err"] is not None and r["abs_err"] > 3 for r in rows))


gt = [r["gt_pct"] for r in runs["P3 (text, 8 turns)"]["rows"]]
print("12 DEV images val_00000..11, local Qwen3.8-27B, thinking off, temperature 0.3, ONE run per image per set-up\n")
print(f"{'set-up':28s} {'MAE pp':>7s} {'median':>7s} {'max':>6s} {'verdicts ok':>11s} {'false PASS':>10s} {'false FAIL':>10s} {'REVIEW':>6s} {'steps':>5s} {'last 2 turns':>12s} {'tokens':>7s} {'sec':>5s} {'pics':>5s} {'err>3pp':>7s}")
for name, data in runs.items():
    s = stats(data)
    print(f"{name:28s} {s['mae']:7.2f} {s['med']:7.2f} {s['mx']:6.2f} {s['agree']:>8d}/12 {s['fpass']:10d} {s['ffail']:10d} {s['review']:6d} {s['steps']:5.1f} {s['last']:>9d}/12 {s['tokens']:7.0f} {s['sec']:5.0f} {s['pics']:5d} {s['big']:7d}")
print(f"{'always PASS':28s} {'':7s} {'':7s} {'':6s} {sum(g <= LIM for g in gt):>8d}/12 {sum(g > LIM for g in gt):10d} {0:10d}")
print(f"{'constant 0.86 %':28s} {np.mean([abs(g - 0.86) for g in gt]):7.2f} {np.median([abs(g - 0.86) for g in gt]):7.2f}")
print("\nper image: gt %, then agent % (verdict) for each set-up")
names = list(runs)
print(f"{'image':10s} {'gt':>6s} " + " ".join(f"{n[:18]:>20s}" for n in names))
for i in range(12):
    cells = []
    for n in names:
        r = runs[n]["rows"][i]
        cells.append(f"{'-' if r['agent_pct'] is None else format(r['agent_pct'], '.2f'):>8s} ({r['agent_status']:6s}){r['steps']:3d}")
    print(f"val_{i:05d} {gt[i]:6.2f} " + " ".join(f"{c:>20s}" for c in cells))

print("\nrun-to-run noise: the SAME set-up run twice on the SAME 12 images")
for c, name in (("A", "A"), ("B", "B"), ("C", "C")):
    r1 = json.load(open(ROOT / f"out/run_dev4_{c}.json"))["rows"]
    r2 = json.load(open(ROOT / f"out/run_dev4_{c}r2.json"))["rows"]
    both = [(a["agent_pct"], b["agent_pct"]) for a, b in zip(r1, r2) if a["agent_pct"] is not None and b["agent_pct"] is not None]
    d = np.array([abs(x - y) for x, y in both])
    flips = sum(a["agent_status"] != b["agent_status"] for a, b in zip(r1, r2))
    print(f"  {name}: verdict changed between the two runs on {flips} of 12 images | median |difference| of the number {np.median(d):.2f} pp, mean {d.mean():.2f}, max {d.max():.2f} (n={len(d)})")

print("\npooled over both runs (24 episodes per set-up; 'always PASS' = 12 images x 2 for comparison)")
print(f"{'set-up':10s} {'MAE':>5s} {'median':>7s} {'p90':>6s} {'verdicts ok':>11s} {'false PASS':>10s} {'false FAIL':>10s} {'REVIEW':>6s} {'err>3pp':>7s} {'mean steps':>10s} {'mean tokens':>11s}")
for c in "ABC":
    rows = json.load(open(ROOT / f"out/run_dev4_{c}.json"))["rows"] + json.load(open(ROOT / f"out/run_dev4_{c}r2.json"))["rows"]
    e = np.array([r["abs_err"] for r in rows if r["abs_err"] is not None])
    print(f"{c:10s} {e.mean():5.2f} {np.median(e):7.2f} {np.percentile(e, 90):6.2f} {sum(r['agent_status'] == r['gt_status'] for r in rows):>8d}/24 "
          f"{sum(r['gt_status'] == 'FAIL' and r['agent_status'] == 'PASS' for r in rows):10d} {sum(r['gt_status'] == 'PASS' and r['agent_status'] == 'FAIL' for r in rows):10d} "
          f"{sum(r['agent_status'] == 'REVIEW' for r in rows):6d} {(e > 3).sum():7d} {np.mean([r['steps'] for r in rows]):10.1f} {np.mean([r['tokens'] for r in rows]):11.0f}")
print(f"{'always PASS':10s} {'':5s} {'':7s} {'':6s} {2 * sum(g <= LIM for g in gt):>8d}/24 {2 * sum(g > LIM for g in gt):10d} {0:10d}")
