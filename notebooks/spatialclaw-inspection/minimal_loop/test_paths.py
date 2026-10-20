"""Control-flow tests for the minimal loop.

STAND-IN: the 'model' is a scripted list of replies, so every path is deterministic and takes seconds.
REAL: the kernel process, the cell timeout, the variable persistence, the rule layer.
Run:  .venv/bin/python -I minimal_loop/test_paths.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from loop import run_episode

IMG = np.zeros((16, 16), np.uint8)


def scripted(*cells):
    """Reply with the given code strings in order, each wrapped the way the real model must answer."""
    it = iter(cells)
    def chat(messages):
        c = next(it, "print('nothing left to say')")
        return f"Purpose: scripted step.\n```python\n{c}\n```", {"prompt": 10, "completion": 5}
    return chat


def check(name, ep, status, reason_part, steps=None):
    ok = ep.verdict.status == status and reason_part in ep.verdict.reason and (steps is None or ep.steps == steps)
    print(f"{'ok  ' if ok else 'FAIL'} {name:46s} -> {ep.verdict.status:6s} {ep.verdict.reason}  (steps={ep.steps})")
    assert ok, (name, ep.verdict, ep.steps)


check("submit above limit",              run_episode(IMG, "q", scripted("submit(2.0, 'big')")), "FAIL", "2.00 > 1.50", 1)
check("submit below limit",              run_episode(IMG, "q", scripted("submit(0.5)")), "PASS", "0.50 <= 1.50", 1)
check("variables persist between cells", run_episode(IMG, "q", scripted("x = 3.0", "submit(x)")), "FAIL", "3.00 > 1.50", 2)
check("never submits -> REVIEW",         run_episode(IMG, "q", scripted("print(1)", "print(2)", "print(3)"), max_steps=3), "REVIEW", "out_of_steps", 3)
check("errors only -> REVIEW",           run_episode(IMG, "q", scripted("1/0", "undefined_name", "1/0", "1/0"), max_failures=2), "REVIEW", "too_many_failures", 2)
check("recovers after an error",         run_episode(IMG, "q", scripted("print(nope)", "nope = 0.2\nsubmit(nope)")), "PASS", "0.20 <= 1.50", 2)
check("invalid submit is rejected",      run_episode(IMG, "q", scripted("submit(150)", "submit(1.0)")), "PASS", "1.00 <= 1.50", 2)
check("timeout interrupts, kernel lives", run_episode(IMG, "q", scripted("while True: pass", "submit(0.1)"), cell_timeout=2), "PASS", "0.10 <= 1.50", 2)


def no_block(messages):
    return "I think the answer is 2.", {"prompt": 1, "completion": 1}
check("format error is counted",         run_episode(IMG, "q", no_block, max_failures=2), "REVIEW", "too_many_failures", 2)

ep = run_episode(IMG, "q", scripted("a = np.ones((3, 3))\nprint(a.sum())", "submit(a.sum())"))
print("stdout the model sees after cell 1:", ep.cells[0]["stdout"].strip(), "| cells in trace:", len(ep.cells))
print("ALL PATHS OK")
