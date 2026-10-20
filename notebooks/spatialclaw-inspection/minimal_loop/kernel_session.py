"""A persistent Python kernel for the agent: one cell in, text out, variables survive between cells.

Written for this series from the paper's description of a 'persistent kernel workspace' (no upstream code).
It is NOT a sandbox: the cell runs as my user, in a throw-away working directory, with a wall-clock timeout.
Part 5 replaces this with a real sandbox.
"""
import ast
import json
import queue
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from jupyter_client import KernelManager

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Runs inside the kernel once per episode. `_BASE` lets the variable summary skip our own plumbing.
SETUP = '''
import json as _json
import numpy as np
import scipy.ndimage as ndi
img = np.load({img_path!r})
_SUBMISSION = None
def submit(void_ratio_pct, note=""):
    """Hand in the measurement. The pass/fail decision is NOT yours: a rule outside the notebook makes it."""
    global _SUBMISSION
    v = float(void_ratio_pct)
    if not (v == v) or not (0.0 <= v <= 100.0):
        raise ValueError("void_ratio_pct must be a number between 0 and 100")
    _SUBMISSION = _json.dumps({{"void_ratio_pct": v, "note": str(note)[:300]}})
    print("[submitted]", _SUBMISSION)
def _sc_vars():
    out = {{}}
    for k, v in list(globals().items()):
        if k.startswith("_") or k in _BASE:
            continue
        d = type(v).__name__
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            d += f" shape={{tuple(v.shape)}} dtype={{v.dtype}}"
        elif isinstance(v, (int, float, str, bool)):
            d += f" = {{str(v)[:40]}}"
        out[k] = d
    return _json.dumps(out)
_BASE = set(globals()) | {{"_BASE"}}
'''


@dataclass
class CellResult:
    stdout: str = ""
    error: str = ""            # last lines of the traceback, '' on success
    seconds: float = 0.0
    timed_out: bool = False
    variables: dict = field(default_factory=dict)
    submission: dict | None = None


class KernelSession:
    def __init__(self, image, timeout_sec: int = 60):
        import numpy as np
        self.timeout_sec = timeout_sec
        self.workdir = tempfile.TemporaryDirectory(prefix="sc_kernel_")
        img_path = Path(self.workdir.name) / "img.npy"
        np.save(img_path, image)
        self.km = KernelManager(kernel_name="python3")
        self.km.start_kernel(cwd=self.workdir.name)
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=60)
        self._exec(SETUP.format(img_path=str(img_path)), 30)

    def _exec(self, code: str, timeout: float, user_expressions: dict | None = None):
        msg_id = self.kc.execute(code, store_history=True, user_expressions=user_expressions or {})
        out, err, timed_out = [], "", False
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                timed_out = True
                self.km.interrupt_kernel()
                deadline = time.monotonic() + 5            # give the interrupt a moment, then drain
            try:
                msg = self.kc.get_iopub_msg(timeout=max(0.1, min(left, 1.0)) if not timed_out else 1.0)
            except queue.Empty:
                if timed_out:
                    break
                continue
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            t, c = msg["msg_type"], msg["content"]
            if t == "stream":
                out.append(c["text"])
            elif t == "execute_result":
                out.append(c["data"].get("text/plain", "") + "\n")
            elif t == "error":
                err = _ANSI.sub("", "\n".join(c["traceback"]))
            elif t == "status" and c["execution_state"] == "idle":
                break
        reply = {}
        try:
            reply = self.kc.get_shell_msg(timeout=5)["content"]
        except queue.Empty:
            pass
        return "".join(out), err, timed_out, reply

    def run(self, code: str) -> CellResult:
        t0 = time.monotonic()
        out, err, timed_out, _ = self._exec(code, self.timeout_sec)
        _, _, _, reply = self._exec("", 10, {"v": "_sc_vars()", "s": "_SUBMISSION"})
        ue = reply.get("user_expressions", {})
        get = lambda k: ast.literal_eval(ue[k]["data"]["text/plain"]) if ue.get(k, {}).get("status") == "ok" else None
        v, s = get("v"), get("s")
        tb = "\n".join(err.strip().splitlines()[-4:]) if err else ""
        if timed_out:
            tb = f"TimeoutError: cell ran longer than {self.timeout_sec}s and was interrupted"
        return CellResult(stdout=out, error=tb, seconds=time.monotonic() - t0, timed_out=timed_out,
                          variables=json.loads(v) if v else {}, submission=json.loads(s) if s else None)

    def close(self):
        try:
            self.kc.stop_channels()
            self.km.shutdown_kernel(now=True)
        finally:
            self.workdir.cleanup()
