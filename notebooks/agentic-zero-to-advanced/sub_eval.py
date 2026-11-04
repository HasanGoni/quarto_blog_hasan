"""Part 10: run the arms of sub_agents.py over the first 200 held-out images. Resumes from out/sub_runs.jsonl.

    uv run python sub_eval.py                    # every arm, the repeat counts in REPEATS
    uv run python sub_eval.py fanout batched     # only some arms
    uv run python sub_eval.py --n 20             # a quick look at the first 20 images

One line per (arm, run, image). The judge arms run several times because the model is not perfectly repeatable even at
temperature 0 (the server batches requests); the agent arms run once because a run costs 10 to 20 minutes.
"""
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sub_agents as sa
from inspection_tools import N_PER_SPLIT

OUT = Path(__file__).parent / "out" / os.environ.get("SUB_RUNS_FILE", "sub_runs.jsonl")   # parallel processes each get their own file
REPEATS = {"single": 1, "supervisor": 1, "batched": 3, "fanout": 3, "panel3": 2}
WORKERS = 8
_lock = threading.Lock()


def done_keys() -> set:
    if not OUT.exists():
        return set()
    return {(r["arm"], r["run"], r["image"]) for r in map(json.loads, OUT.read_text().splitlines()) if r.get("status") == "ok"}


def run_one(arm: str, run: int, image: str) -> None:
    try:
        r = {**sa.ARMS[arm](image), "run": run, "status": "ok"}
    except Exception as exc:     # keep the failure, do not lose the other 199 images
        r = {"arm": arm, "run": run, "image": image, "status": f"error: {type(exc).__name__}: {str(exc)[:200]}"}
    with _lock, OUT.open("a") as f:
        f.write(json.dumps(r) + "\n")


def main(argv: list[str]) -> None:
    n = N_PER_SPLIT
    if "--n" in argv:
        n = int(argv[argv.index("--n") + 1])
    arms = [a for a in argv if a in sa.ARMS] or list(sa.ARMS)
    images = [f"test_{i:05d}" for i in range(n)]
    OUT.parent.mkdir(exist_ok=True)
    for arm in arms:
        for run in range(REPEATS[arm]):
            have = done_keys()
            todo = [im for im in images if (arm, run, im) not in have]
            print(f"{arm} run {run}: {len(todo)} images to do", flush=True)
            with ThreadPoolExecutor(WORKERS) as pool:
                list(pool.map(lambda im: run_one(arm, run, im), todo))


if __name__ == "__main__":
    main(sys.argv[1:])
