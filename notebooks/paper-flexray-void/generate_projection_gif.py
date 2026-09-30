"""Real (if toy) physics-based multi-angle projection simulation.

This is the one FleXray idea that transfers most directly to industrial void detection:
project a labeled 3D volume from many random angles to generate an unlimited set of
labeled 2D X-rays for free. There is no public micro-CT-of-castings-with-voids dataset to
run this on for real, so this demonstrates the *mechanism* on a synthetic toy volume: a
solid aluminum-like block with one embedded lower-density spherical void, ray-summed
(parallel-beam, Beer-Lambert-style line integral) from 24 angles around one axis.

Real numpy math, real rotation, real accumulated attenuation -- just a toy density field
instead of a real casting CT, because that source data doesn't exist publicly.
"""
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from PIL import Image
import io

N = 96
volume = np.zeros((N, N, N), dtype=np.float32)

# Solid casting block (attenuation coefficient ~1.0)
volume[16:80, 16:80, 16:80] = 1.0

# A void: much lower attenuation, off-center so rotation actually reveals shape changes
cz, cy, cx = 48, 40, 60
r = 10
zz, yy, xx = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
void_mask = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
volume[void_mask] = 0.12

# A second, smaller void near an edge -- disappears from some angles, exactly the kind of
# angle-dependent occlusion FleXray's oblique-angle robustness section is about
cz2, cy2, cx2, r2 = 60, 68, 24, 6
void2 = (zz - cz2) ** 2 + (yy - cy2) ** 2 + (xx - cx2) ** 2 <= r2 ** 2
volume[void2] = 0.12

frames = []
angles = np.arange(0, 360, 15)
vmin, vmax = None, None
projections = []
for angle in angles:
    rotated = rotate(volume, angle=angle, axes=(0, 2), reshape=False, order=1, mode="constant", cval=0.0)
    projection = rotated.sum(axis=0)  # parallel-beam line integral along one axis
    projections.append(projection)

vmax = max(p.max() for p in projections)

for angle, projection in zip(angles, projections):
    xray = np.exp(-projection / (vmax * 0.35))  # Beer-Lambert: denser path -> darker pixel... inverted for X-ray look
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=100)
    ax.imshow(xray, cmap="gray", vmin=xray.min(), vmax=1.0)
    ax.set_title(f"{angle}°", fontsize=10)
    ax.axis("off")
    buf = io.BytesIO()
    plt.tight_layout(pad=0.3)
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).convert("RGB"))

out_path = "../../posts/series/papers/images/flexray-void-projection-sweep.gif"
frames[0].save(
    out_path,
    save_all=True,
    append_images=frames[1:],
    duration=180,
    loop=0,
    optimize=True,
)
print("Saved", out_path)

import os
print("Size KB:", os.path.getsize(out_path) / 1024)
