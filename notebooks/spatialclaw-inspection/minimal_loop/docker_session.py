"""Part 5: the same session interface as Part 3/4, but the model's code runs in a locked-down container.

  docker run -i --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges
             --pids-limit 64 --memory 1g --cpus 1 --tmpfs /scratch  -v <image dir>:/data:ro  sc-inspect-sandbox

The host never trusts the container's answers: reads are size-capped, ids must match, every field is validated
(`validate_reply`: submission range and NaN, PNG-only image data URLs, text lengths), and a cell that does not answer
before a hard wall-clock limit gets the whole container killed (the in-container alarm can be defeated by the model).
A dead sandbox ends the episode as REVIEW; there is no restart that could hide the event.
"""
import base64
import binascii
import json
import math
import queue
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from kernel_session import SETUP, CellResult
from session4 import _TOOLS_SRC

IMAGE = "sc-inspect-sandbox"
# Size caps for what the container may send back. The runner's own limits are below these, so an honest reply never hits them.
MAX_LINE = 6_000_000          # one reply line, bytes
MAX_STDOUT, MAX_ERROR, MAX_IMAGES, MAX_URL, MAX_TITLE = 20_000, 2_000, 2, 4_000_000, 300
MAX_VARS, MAX_VAR_TEXT, MAX_NOTE = 60, 400, 300
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
DATA_PREFIX = "data:image/png;base64,"


class ProtocolError(ValueError):
    pass


def _no_constant(name):
    raise ProtocolError(f"non-finite JSON constant {name}")


def _is_real(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_reply(r, expected_id: int) -> dict:
    """Check one reply from the container against the protocol; raise ProtocolError on anything unexpected.
    Returns a normalised copy, so nothing unchecked reaches the loop."""
    def need(cond, why):
        if not cond:
            raise ProtocolError(why)
    need(isinstance(r, dict) and r.get("id") == expected_id and not isinstance(r.get("id"), bool), "wrong id")
    need(isinstance(r.get("stdout"), str) and len(r["stdout"]) <= MAX_STDOUT, "stdout")
    need(isinstance(r.get("error"), str) and len(r["error"]) <= MAX_ERROR, "error")
    need(isinstance(r.get("timed_out", False), bool), "timed_out")
    v = r.get("variables") or {}
    need(isinstance(v, dict), "variables")
    need(all(isinstance(k, str) and isinstance(x, str) for k, x in v.items()), "variables entries")
    # informational only (the loop shows the first 25): an honest model may define hundreds, so clip instead of refusing
    v = {k[:100]: x[:MAX_VAR_TEXT] for k, x in list(v.items())[:MAX_VARS]}
    sub = r.get("submission")
    if sub is not None:
        need(isinstance(sub, dict) and set(sub) == {"void_ratio_pct", "note"}, "submission shape")
        need(_is_real(sub["void_ratio_pct"]) and 0.0 <= sub["void_ratio_pct"] <= 100.0, "submission void_ratio_pct")
        need(isinstance(sub["note"], str) and len(sub["note"]) <= MAX_NOTE, "submission note")
        sub = {"void_ratio_pct": float(sub["void_ratio_pct"]), "note": sub["note"]}
    imgs = r.get("images") or []
    need(isinstance(imgs, list) and len(imgs) <= MAX_IMAGES, "images")
    clean = []
    for it in imgs:
        need(isinstance(it, list) and len(it) == 2 and isinstance(it[0], str) and isinstance(it[1], str), "image item")
        title, url = it
        need(len(title) <= MAX_TITLE and url.startswith(DATA_PREFIX) and len(url) <= MAX_URL, "image url")
        try:
            raw = base64.b64decode(url[len(DATA_PREFIX):], validate=True)
        except (binascii.Error, ValueError):
            raise ProtocolError("image is not base64")
        need(raw.startswith(PNG_MAGIC) and raw[12:16] == b"IHDR", "image is not a PNG")
        clean.append((title, url))
    return {"stdout": r["stdout"], "error": r["error"], "timed_out": r.get("timed_out", False),
            "variables": v, "submission": sub, "images": clean}
FLAGS = ["--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
         "--pids-limit", "64", "--memory", "1g", "--memory-swap", "1g", "--cpus", "1",
         "--tmpfs", "/scratch:rw,size=64m,uid=10001,gid=10001,mode=0700"]


class DockerSession:
    def __init__(self, image, timeout_sec: int = 60, with_show: bool = False, with_tools: bool = False,
                 hard_extra: int = 10, extra_flags: list | None = None):
        import numpy as np
        self.timeout_sec, self.hard = timeout_sec, timeout_sec + hard_extra
        self.name = "sc-sbx-" + uuid.uuid4().hex[:10]
        self.data = tempfile.TemporaryDirectory(prefix="sc_data_")
        np.save(Path(self.data.name) / "img.npy", image)
        Path(self.data.name).chmod(0o755)                        # the container user (uid 10001) must be able to read it
        (Path(self.data.name) / "img.npy").chmod(0o644)
        cmd = ["docker", "run", "-i", "--rm", "--name", self.name, *(FLAGS if extra_flags is None else extra_flags),
               "-v", f"{self.data.name}:/data:ro", "-e", f"CELL_TIMEOUT={timeout_sec}", IMAGE]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.q: queue.Queue = queue.Queue()
        def pump():                                    # one line per answer; None = stdout closed, b"" = a line over MAX_LINE
            out = self.p.stdout
            while True:
                line = out.readline(MAX_LINE + 1)      # bounded read: an endless line cannot grow host memory
                if not line:
                    break
                if len(line) > MAX_LINE and not line.endswith(b"\n"):
                    self.q.put(b"")
                    for _ in iter(lambda: out.read(65536), b""):   # keep draining (and discarding) so the container is never
                        pass                                       # blocked on a full pipe while it is being killed
                    break
                self.q.put(line)
            self.q.put(None)
        threading.Thread(target=pump, daemon=True).start()
        self.n, self.dead = 0, None
        setup = SETUP.format(img_path="/data/img.npy")
        if with_show or with_tools:
            drop = ([] if with_show else ["show"]) + ([] if with_tools else ["part_mask", "label_blobs", "overlay", "crop"])
            setup += "\n" + _TOOLS_SRC + "\n" + "".join(f"del {n}\n" for n in drop) + "_BASE |= set(globals())\n"
        r = self._call(setup, 60)
        assert r is not None and not r.get("error"), (r, self.dead)

    def _call(self, code: str, hard: float):
        """Send one cell, return the validated response dict, or None (and self.dead set) if the sandbox is gone or lying."""
        if self.dead:
            return None
        self.n += 1
        try:
            self.p.stdin.write((json.dumps({"id": self.n, "code": code}) + "\n").encode())
            self.p.stdin.flush()
        except (BrokenPipeError, OSError):
            self.dead = "sandbox_died"
            return None
        try:
            line = self.q.get(timeout=hard)
        except queue.Empty:
            self.kill()
            self.dead = f"sandbox_killed: no answer within {hard:.0f}s"
            return None
        if line is None:
            self.dead = "sandbox_died: container exited"
            return None
        if line == b"":                                # the pump saw a line over MAX_LINE
            self.kill()
            self.dead = "sandbox_protocol_error: reply longer than the size cap"
            return None
        try:
            r = validate_reply(json.loads(line, parse_constant=_no_constant), self.n)
        except (ProtocolError, ValueError, RecursionError) as e:    # json errors are ValueError
            self.kill()
            self.dead = "sandbox_protocol_error: answer was not the expected reply to this cell" + f" ({e})"
            return None
        return r

    def run(self, code: str) -> CellResult:
        t0 = time.monotonic()
        r = self._call(code, self.hard)
        if r is None:
            res = CellResult(error=f"sandbox: {self.dead}", seconds=time.monotonic() - t0, timed_out="killed" in (self.dead or ""))
            res.images, res.fatal = [], self.dead
            return res
        res = CellResult(stdout=r["stdout"], error=r["error"], seconds=time.monotonic() - t0, timed_out=r["timed_out"],
                         variables=r["variables"], submission=r["submission"])
        res.images = r["images"]
        res.fatal = None
        return res

    def kill(self):
        try:
            subprocess.run(["docker", "kill", self.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
        except subprocess.TimeoutExpired:
            pass
        self.p.kill()                                  # also drop the docker client, whatever the daemon is doing

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.kill()
        self.data.cleanup()
