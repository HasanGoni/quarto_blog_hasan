"""Part 10: wall-clock time per image, one image at a time. The 200-image runs used 8 images in parallel, three processes at once,
so their `seconds` are inflated by queueing and only comparable roughly. This script runs the first 20 held-out images that have at
least one candidate, strictly one after the other, each arm on its own (other users of the GPU can still add noise).

    uv run python sub_latency.py
"""
import json
import time
from pathlib import Path

import sub_agents as sa
from inspection_tools import find_candidates

OUT = Path(__file__).parent / "out" / "sub_latency.json"
N = 20


def main():
    images = [im for im in (f"test_{i:05d}" for i in range(200)) if find_candidates(im)["candidates"]][:N]
    res = json.loads(OUT.read_text()) if OUT.exists() else {"images": images, "arms": {}}   # resumable: an arm already timed is kept
    for arm, fn in sa.ARMS.items():
        if arm in res["arms"]:
            continue
        rows = []
        for im in images:
            t = time.time()
            r = fn(im)
            rows.append({"image": im, "wall": round(time.time() - t, 2), "calls": r["calls"], "n_candidates": r["n_candidates"]})
        w = sorted(x["wall"] for x in rows)
        res["arms"][arm] = {"rows": rows, "mean": round(sum(w) / len(w), 1), "median": w[len(w) // 2], "max": w[-1]}
        print(arm, res["arms"][arm]["mean"], res["arms"][arm]["median"], res["arms"][arm]["max"], flush=True)
        OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
