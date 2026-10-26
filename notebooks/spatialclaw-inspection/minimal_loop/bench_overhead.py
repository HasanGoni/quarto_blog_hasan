"""What does the container cost? Session start+close and per-cell latency, unsandboxed kernel vs container. 10 sessions each, 30 trivial cells per session."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from docker_session import DockerSession
from session4 import Session4

IMG = np.zeros((224, 224), np.uint8)
for name, cls in (("unsandboxed Jupyter kernel", Session4), ("container", DockerSession)):
    starts, cells = [], []
    for _ in range(10):
        t = time.monotonic(); s = cls(IMG, timeout_sec=10, with_show=True, with_tools=True); starts.append(time.monotonic() - t)
        for _ in range(30):
            t = time.monotonic(); s.run("x = np.arange(1000).sum()"); cells.append(time.monotonic() - t)
        s.close()
    print(f"{name:28s} start: median {np.median(starts)*1000:6.0f} ms | per trivial cell: median {np.median(cells)*1000:5.1f} ms, p95 {np.percentile(cells, 95)*1000:5.1f} ms")
