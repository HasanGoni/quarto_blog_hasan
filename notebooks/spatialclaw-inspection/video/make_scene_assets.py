"""Assets for the Part 1 video, scene 2: the SAME seeded synthetic pad as smoke_kernel.py (stand-in image, not a real X-ray)."""
import pathlib
import numpy as np
from PIL import Image

OUT = pathlib.Path(__file__).parent / "assets"
rng = np.random.default_rng(0)
img = np.full((200, 200), 200.0)
yy, xx = np.mgrid[:200, :200]
for cy, cx, r in [(60, 70, 12), (120, 140, 18), (150, 60, 8)]:
    img[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = 60.0
img += rng.normal(0, 3, img.shape)
pad = (yy - 100) ** 2 + (xx - 100) ** 2 < 90 ** 2
voids = (img < 130) & pad
print("pad px", int(pad.sum()), "void px", int(voids.sum()))          # must match outputs/smoke_kernel.txt: 25433 / 1635
shown = np.where(pad, img, 0.35 * img)                                  # dim everything outside the pad
Image.fromarray(np.clip(shown, 0, 255).astype(np.uint8)).resize((600, 600), Image.NEAREST).save(OUT / "pad.png")
rgba = np.zeros((200, 200, 4), np.uint8); rgba[voids] = (224, 49, 49, 200)
Image.fromarray(rgba).resize((600, 600), Image.NEAREST).save(OUT / "voids_overlay.png")
