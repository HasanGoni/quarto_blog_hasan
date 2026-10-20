"""Real run: local Qwen3.8-27B (vLLM) drives the minimal loop on the first N images of the v2 DEV split (val).

Ground truth (exact void ratio from the mask) is read ONLY here, after the agent has finished. The held-out test split is not touched.
Run:  .venv/bin/python -I minimal_loop/run_dev.py [N]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from loop import LIMIT_PCT, openai_chat, run_episode

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
DATA = Path(__file__).resolve().parents[2] / "agentic-factory-segmentation/data/void_dataset_v2/val.npz"
d = np.load(DATA)
images, dmask, pmask = d["images"], d["defect_masks"], d["part_masks"]
chat = openai_chat()
rows, traces = [], []
print(f"{'image':10s} {'gt%':>6s} {'agent%':>7s} {'abs err':>8s} {'gt':>6s} {'agent':>6s} {'steps':>5s} {'tokens':>7s} {'sec':>6s}  reason")
for i in range(N):
    gt = 100.0 * float((dmask[i] == 1).sum()) / float(pmask[i].sum())
    ep = run_episode(images[i], "Measure the void ratio of the part in `img` and submit it.", chat)
    v = ep.verdict
    gt_status = "FAIL" if gt > LIMIT_PCT else "PASS"
    err = None if v.void_ratio_pct is None else abs(v.void_ratio_pct - gt)
    rows.append(dict(image=f"val_{i:05d}", gt_pct=gt, agent_pct=v.void_ratio_pct, abs_err=err, gt_status=gt_status, agent_status=v.status,
                     steps=ep.steps, tokens=ep.prompt_tokens + ep.completion_tokens, seconds=ep.seconds, reason=v.reason,
                     has_cracks=bool((dmask[i] == 2).any())))
    traces.append(dict(image=f"val_{i:05d}", cells=ep.cells))
    r = rows[-1]
    print(f"{r['image']:10s} {gt:6.2f} {('-' if v.void_ratio_pct is None else f'{v.void_ratio_pct:.2f}'):>7s} "
          f"{('-' if err is None else f'{err:.2f}'):>8s} {gt_status:>6s} {v.status:>6s} {ep.steps:5d} {r['tokens']:7d} {ep.seconds:6.1f}  {v.reason}", flush=True)

done = [r for r in rows if r["abs_err"] is not None]
errs = np.array([r["abs_err"] for r in done]) if done else np.array([0.0])
print("\nSUMMARY (n=%d, DEV split, limit %.1f%%)" % (len(rows), LIMIT_PCT))
print("  finished with a measurement:", len(done), "| REVIEW:", sum(r["agent_status"] == "REVIEW" for r in rows))
print("  mean abs error (pct points): %.2f | median: %.2f | max: %.2f" % (errs.mean(), np.median(errs), errs.max()))
print("  verdict agrees with ground truth:", sum(r["agent_status"] == r["gt_status"] for r in rows), "of", len(rows))
print("  FALSE PASS (gt FAIL, agent PASS):", sum(r["gt_status"] == "FAIL" and r["agent_status"] == "PASS" for r in rows),
      "| false FAIL:", sum(r["gt_status"] == "PASS" and r["agent_status"] == "FAIL" for r in rows))
print("  mean steps: %.1f | mean tokens: %d | mean seconds: %.1f" % (np.mean([r["steps"] for r in rows]), np.mean([r["tokens"] for r in rows]), np.mean([r["seconds"] for r in rows])))
Path(__file__).resolve().parents[1].joinpath("out").mkdir(exist_ok=True)
json.dump(dict(limit_pct=LIMIT_PCT, rows=rows, traces=traces), open(Path(__file__).resolve().parents[1] / "out/run_dev.json", "w"), indent=1)
