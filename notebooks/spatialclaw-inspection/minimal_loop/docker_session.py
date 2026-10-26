"""Part 5: the same session interface as Part 3/4, but the model's code runs in a locked-down container.

  docker run -i --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges
             --pids-limit 64 --memory 1g --cpus 1 --tmpfs /scratch  -v <image dir>:/data:ro  sc-inspect-sandbox

The host never trusts the container's answers: ids must match, the schema is checked, and a cell that does not answer
before a hard wall-clock limit gets the whole container killed (the in-container alarm can be defeated by the model).
A dead sandbox ends the episode as REVIEW; there is no restart that could hide the event.
"""
import json
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
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.q: queue.Queue = queue.Queue()
        def pump():                                    # one line per answer; None means the container's stdout closed
            for line in self.p.stdout:
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
            self.p.stdin.write(json.dumps({"id": self.n, "code": code}) + "\n")
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
        try:
            r = json.loads(line)
            assert isinstance(r, dict) and r.get("id") == self.n and isinstance(r.get("stdout"), str) and isinstance(r.get("error"), str)
        except Exception:
            self.kill()
            self.dead = "sandbox_protocol_error: answer was not the expected reply to this cell"
            return None
        return r

    def run(self, code: str) -> CellResult:
        t0 = time.monotonic()
        r = self._call(code, self.hard)
        if r is None:
            res = CellResult(error=f"sandbox: {self.dead}", seconds=time.monotonic() - t0, timed_out="killed" in (self.dead or ""))
            res.images, res.fatal = [], self.dead
            return res
        res = CellResult(stdout=r["stdout"], error=r["error"], seconds=time.monotonic() - t0, timed_out=bool(r.get("timed_out")),
                         variables=r.get("variables") or {}, submission=r.get("submission"))
        res.images = [(t, u) for t, u in r.get("images", [])][:2]
        res.fatal = None
        return res

    def kill(self):
        subprocess.run(["docker", "kill", self.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
