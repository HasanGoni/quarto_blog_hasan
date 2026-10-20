"""Smoke test: upstream JupyterKernelManager + ReturnAnswer sentinel, loaded verbatim.

STAND-INS (labelled, not from the paper):
  * the 'agent' is a hard-coded list of cells -- no VLM is called
  * the image is a synthetic 'solder pad with voids' array, not a real X-ray
  * 'mask' comes from a plain threshold, not SAM3
What IS real: the kernel process, variable persistence, traceback capture,
the variable summary, and the ReturnAnswer sentinel that ends the loop.
"""
import asyncio, importlib.util, pathlib

HERE = pathlib.Path(__file__).parent
UP = HERE / "upstream/repo/spatial_agent"

def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, UP / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

mgr = load("sc_manager", "kernel/manager.py")
ra_source = (UP / "kernel_types/return_answer.py").read_text()

CELLS = [
    ("1. build evidence (stand-in for SAM3)", '''
import numpy as np
rng = np.random.default_rng(0)
img = np.full((200, 200), 200.0)                 # bright solder
yy, xx = np.mgrid[:200, :200]
for cy, cx, r in [(60, 70, 12), (120, 140, 18), (150, 60, 8)]:
    img[(yy-cy)**2 + (xx-cx)**2 < r*r] = 60.0    # dark voids
img += rng.normal(0, 3, img.shape)
pad = (yy-100)**2 + (xx-100)**2 < 90**2
voids = (img < 130) & pad
print("pad px:", int(pad.sum()), " void px:", int(voids.sum()))
'''),
    ("2. reuse variables from step 1", '''
from scipy import ndimage
lab, n = ndimage.label(voids)
sizes = ndimage.sum(voids, lab, range(1, n+1))
print("void count:", n, " sizes:", [int(s) for s in sizes])
print("void ratio: %.2f%%" % (100 * voids.sum() / pad.sum()))
'''),
    ("3. agent makes a mistake", '''
print(largest_void_area)
'''),
    ("4. agent revises after seeing the traceback", '''
largest_void_area = int(max(sizes))
print("largest void px:", largest_void_area)
'''),
    ("5. commit", '''
ReturnAnswer("FAIL" if 100 * voids.sum() / pad.sum() > 5 else "PASS")
'''),
]

async def main():
    km = mgr.JupyterKernelManager(timeout_sec=60)
    await km.start()
    await km.execute(ra_source)               # inject ReturnAnswer, as init_node does
    for title, code in CELLS:
        r = await km.execute(code)
        print(f"--- {title}  ({r.execution_time_sec:.2f}s)")
        if r.stdout.strip(): print(r.stdout.rstrip())
        if r.error:          print("TRACEBACK (last line):", r.error.strip().splitlines()[-1])
    print("--- variables the feedback step would summarise:")
    for k, v in (await km.get_variables()).items():
        print("  ", k, v)
    print("--- sentinel:", await km.check_sentinel())
    await km.shutdown()

asyncio.run(main())
