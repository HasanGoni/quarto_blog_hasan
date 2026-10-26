"""Cell runner that lives INSIDE the container. Written for this series.

Protocol (one JSON object per line):
  host -> runner  {"id": 7, "code": "..."}
  runner -> host  {"id": 7, "stdout": "...", "error": "...", "timed_out": false, "variables": {...}, "submission": {...}|null,
                   "images": [[title, data-url], ...]}
The namespace NS persists between cells (the 'persistent kernel'). The model's prints go to a buffer. Before any model code
runs, fd 1 and fd 2 are pointed at /dev/null and the protocol keeps a private copy of the real stdout.
NOTE: the model's code runs in THIS process, so it can still reach that private copy. The container, not this file, is the
security boundary. The host therefore treats every response as untrusted (ids, schema, size).
"""
import ast
import base64
import builtins
import contextlib
import io
import json
import os
import signal
import sys
import traceback
from pathlib import Path

_proto = os.fdopen(os.dup(1), "w", buffering=1)
_devnull = os.open(os.devnull, os.O_RDWR)
os.dup2(_devnull, 1)
os.dup2(_devnull, 2)
NS = {"__name__": "__cell__", "__builtins__": builtins}   # the module, as in a notebook (a dict would break `__builtins__.round`)
TIMEOUT = int(os.environ.get("CELL_TIMEOUT", "60"))
SHOWN = Path("/scratch/shown")
_seen = set()


class CellTimeout(BaseException):
    pass


def _alarm(signum, frame):
    raise CellTimeout()


def _images(max_images=2):
    out = []
    for p in sorted(SHOWN.glob("*.png")) if SHOWN.exists() else []:
        if p.name in _seen:
            continue
        _seen.add(p.name)
        if len(out) < max_images:
            t = p.with_suffix(".txt")
            out.append([t.read_text() if t.exists() else "", "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()])
    return out


def _run_cell(code):
    """exec like a notebook: if the LAST statement is an expression, print its repr (what Jupyter does), unless it is None."""
    tree = ast.parse(code, "<cell>")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, "<cell>", "exec"), NS)
        val = eval(compile(last, "<cell>", "eval"), NS)
        if val is not None:
            print(repr(val))
    else:
        exec(compile(tree, "<cell>", "exec"), NS)


def _eval_json(expr):
    try:
        return json.loads(eval(expr, NS)) if expr == "_sc_vars()" else eval(expr, NS)
    except BaseException:
        return None


signal.signal(signal.SIGALRM, _alarm)
for line in sys.stdin:
    msg = json.loads(line)
    buf, err, timed_out = io.StringIO(), "", False
    signal.alarm(TIMEOUT)
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            _run_cell(msg["code"])
    except CellTimeout:
        timed_out = True
        err = f"TimeoutError: cell ran longer than {TIMEOUT}s and was interrupted"
    except BaseException:
        err = "\n".join(traceback.format_exc().strip().splitlines()[-4:])
    finally:
        signal.alarm(0)
    sub = NS.get("_SUBMISSION")
    resp = {"id": msg["id"], "stdout": buf.getvalue()[:20000], "error": err, "timed_out": timed_out,
            "variables": _eval_json("_sc_vars()") or {}, "submission": json.loads(sub) if isinstance(sub, str) else None, "images": _images()}
    _proto.write(json.dumps(resp) + "\n")
