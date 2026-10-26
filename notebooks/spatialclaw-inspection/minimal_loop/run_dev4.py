"""Part 4 real run: the SAME 12 DEV images as Part 3, local Qwen3.8-27B (vLLM), one condition per invocation.

  A  no show(), no tools, 14 turns   (Part 3's setup with a bigger budget: isolates the budget)
  B  show(),    no tools, 14 turns   (adds eyes)
  C  show() + tools, 14 turns        (adds eyes and hands)

Ground truth is read only here, after the agent has finished. The held-out test split is not touched.
Part 5: SANDBOX=1 runs the model's cells in the container (docker_session.py) instead of the unsandboxed kernel.
Run:  [SANDBOX=1] .venv/bin/python -I minimal_loop/run_dev4.py A|B|C [N] [tag]     (tag = 'r2' for the repeat run, same settings)
"""
import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from loop4 import LIMIT_PCT, openai_chat, run_episode4
from session4 import Session4
from docker_session import DockerSession

COND = {"A": (False, False), "B": (True, False), "C": (True, True)}
cond = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
show, tools = COND[cond]
ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT.parent / "agentic-factory-segmentation/data/void_dataset_v2/val.npz")
images, dmask, pmask = d["images"], d["defect_masks"], d["part_masks"]
imgdir = ROOT / f"out/part4_{cond}{TAG}_images"
imgdir.mkdir(parents=True, exist_ok=True)
chat = openai_chat(max_tokens=1500)
rows, traces = [], []
print(f"condition {cond}: show={show} tools={tools} max_steps=14")
print(f"{'image':10s} {'gt%':>6s} {'agent%':>7s} {'abs err':>8s} {'gt':>6s} {'agent':>6s} {'steps':>5s} {'tokens':>7s} {'sec':>6s}  reason")
for i in range(N):
    gt = 100.0 * float((dmask[i] == 1).sum()) / float(pmask[i].sum())
    ep = run_episode4(images[i], "Measure the void ratio of the part in `img` and submit it.", chat, with_show=show, with_tools=tools, max_steps=14,
                      session_cls=DockerSession if os.environ.get("SANDBOX") else Session4)
    v = ep.verdict
    gt_status = "FAIL" if gt > LIMIT_PCT else "PASS"
    err = None if v.void_ratio_pct is None else abs(v.void_ratio_pct - gt)
    for c in ep.cells:                       # pictures to files, so the JSON stays small
        for k, im in enumerate(c.get("images", [])):
            fn = f"val_{i:05d}_s{c['step']:02d}_{k}.png"
            (imgdir / fn).write_bytes(base64.b64decode(im.pop("png").split(",", 1)[1]))
            im["file"] = fn
    rows.append(dict(image=f"val_{i:05d}", gt_pct=gt, agent_pct=v.void_ratio_pct, abs_err=err, gt_status=gt_status, agent_status=v.status,
                     steps=ep.steps, tokens=ep.prompt_tokens + ep.completion_tokens, seconds=ep.seconds, reason=v.reason,
                     pictures=sum(len(c.get("images", [])) for c in ep.cells)))
    traces.append(dict(image=f"val_{i:05d}", cells=ep.cells))
    r = rows[-1]
    print(f"{r['image']:10s} {gt:6.2f} {('-' if v.void_ratio_pct is None else f'{v.void_ratio_pct:.2f}'):>7s} "
          f"{('-' if err is None else f'{err:.2f}'):>8s} {gt_status:>6s} {v.status:>6s} {ep.steps:5d} {r['tokens']:7d} {ep.seconds:6.1f}  {v.reason}", flush=True)

done = [r for r in rows if r["abs_err"] is not None]
errs = np.array([r["abs_err"] for r in done]) if done else np.array([0.0])
print("\nSUMMARY condition %s (n=%d, DEV split, limit %.1f%%)" % (cond, len(rows), LIMIT_PCT))
print("  finished with a measurement:", len(done), "| REVIEW:", sum(r["agent_status"] == "REVIEW" for r in rows))
print("  mean abs error (pct points): %.2f | median: %.2f | max: %.2f" % (errs.mean(), np.median(errs), errs.max()))
print("  verdict agrees with ground truth:", sum(r["agent_status"] == r["gt_status"] for r in rows), "of", len(rows))
print("  FALSE PASS (gt FAIL, agent PASS):", sum(r["gt_status"] == "FAIL" and r["agent_status"] == "PASS" for r in rows),
      "| false FAIL:", sum(r["gt_status"] == "PASS" and r["agent_status"] == "FAIL" for r in rows))
print("  mean steps: %.1f | mean tokens: %d | mean seconds: %.1f | pictures shown: %d" % (
    np.mean([r["steps"] for r in rows]), np.mean([r["tokens"] for r in rows]), np.mean([r["seconds"] for r in rows]), sum(r["pictures"] for r in rows)))
json.dump(dict(cond=cond, limit_pct=LIMIT_PCT, max_steps=14, rows=rows, traces=traces), open(ROOT / f"out/run_dev4_{cond}{TAG}.json", "w"), indent=1)
