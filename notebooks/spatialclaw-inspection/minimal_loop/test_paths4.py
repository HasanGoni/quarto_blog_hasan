"""Tests for Part 4: show() and the tool API, against the REAL kernel.
STAND-IN: the 'model' is a scripted list of replies. REAL: kernel, tools, image capture, message building, rule layer.
Run:  .venv/bin/python -I minimal_loop/test_paths4.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from loop4 import run_episode4

rng = np.random.default_rng(0)
IMG = np.full((40, 40), 230, np.uint8)
IMG[8:32, 8:32] = 70                      # dark square "part"
IMG[18:22, 18:22] = 130                   # a 4x4 lighter "void" (16 px)
IMG = IMG + rng.integers(0, 3, IMG.shape).astype(np.uint8)
SEEN = []


def scripted(*cells):
    it = iter(cells)
    def chat(messages):
        SEEN.append(messages[-1])
        c = next(it, "print('nothing left to say')")
        return f"Purpose: scripted step.\n```python\n{c}\n```", {"prompt": 10, "completion": 5}
    return chat


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name:52s} {detail}")
    assert ok, name


# 1. show() produces a picture that is attached to the NEXT message as an image part
SEEN.clear()
ep = run_episode4(IMG, "q", scripted("show(img, 'raw')", "submit(1.0)"), with_show=True)
last = SEEN[-1]["content"]
check("show: image part attached to next message", isinstance(last, list) and last[1]["type"] == "image_url", f"parts={[p['type'] for p in last]}")
check("show: data URL is a PNG", last[1]["image_url"]["url"].startswith("data:image/png;base64,"))
check("show: title travels in the text", "raw" in last[0]["text"])
check("show: trace keeps the picture", len(ep.cells[0]["images"]) == 1 and ep.cells[0]["images"][0]["title"] == "raw")

# 2. at most 2 pictures per turn
SEEN.clear()
ep = run_episode4(IMG, "q", scripted("show(img,'a'); show(img,'b'); show(img,'c')", "submit(1.0)"), with_show=True)
check("show: at most 2 pictures per turn", len(SEEN[-1]["content"]) == 3, f"parts={len(SEEN[-1]['content'])}")

# 3. conditions: without with_show the function does not exist; the model gets a plain NameError, never a crash
ep = run_episode4(IMG, "q", scripted("show(img)", "submit(1.0)"), with_show=False)
check("no-show condition: show() is a NameError", "NameError" in ep.cells[0]["error"], ep.cells[0]["error"].splitlines()[-1])
ep = run_episode4(IMG, "q", scripted("part_mask(img)", "submit(1.0)"), with_show=True, with_tools=False)
check("no-tools condition: part_mask is a NameError", "NameError" in ep.cells[0]["error"])

# 4. tools do what the doc says
code = ("pm = part_mask(img)\nprint('part px', int(pm.sum()))\n"
        "lab, bl = label_blobs((img > 100) & pm)\nprint(bl)\nsubmit(100*bl[0]['area']/pm.sum())")
ep = run_episode4(IMG, "q", scripted(code), with_show=True, with_tools=True)
check("tools: part_mask finds the 24x24 square", "part px 576" in ep.cells[0]["stdout"], ep.cells[0]["stdout"].splitlines()[0])
check("tools: label_blobs finds the 16 px void", "'area': 16" in ep.cells[0]["stdout"])
check("tools: submit uses them", ep.verdict.status == "FAIL" and abs(ep.verdict.void_ratio_pct - 100 * 16 / 576) < 1e-6, ep.verdict.reason)

# 5. overlay + crop feed show()
ep = run_episode4(IMG, "q", scripted("pm = part_mask(img)\nshow(overlay(img, [pm]), 'outline')\nshow(crop(img, 20, 20, r=8, zoom=3), 'zoom')", "submit(1.0)"),
                  with_show=True, with_tools=True)
check("overlay + crop produce 2 pictures", len(ep.cells[0]["images"]) == 2 and not ep.cells[0]["error"], ep.cells[0]["error"][-80:])

# 6. bad input is an error the model can read, not a crash
ep = run_episode4(IMG, "q", scripted("show(np.zeros((2,2,2,2)))", "submit(1.0)"), with_show=True)
check("show: bad shape is a readable error", "ValueError" in ep.cells[0]["error"])

# 7. the rule layer is unchanged: budget exhausted with pictures in play -> REVIEW
ep = run_episode4(IMG, "q", scripted("show(img)", "show(img)", "show(img)"), with_show=True, max_steps=3)
check("out of budget still -> REVIEW", ep.verdict.status == "REVIEW" and "out_of_steps" in ep.verdict.reason)
print("ALL PART-4 PATHS OK")
