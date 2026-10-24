"""Posters of SAM3 part masks on v2 test images.  Light green tint = SAM3 mask, thin white line = exact part mask.
poster_sam3_sample.png (48 evenly spaced), poster_sam3_worst.png (the 48 lowest IoU).  plain numpy/PIL/scipy."""
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

root = Path(__file__).parent
D = (root / "data" / "void_dataset_v2").resolve(); O = root / "out/sam3"
Q = D.parent / "void_dataset_v2_posters"; Q.mkdir(exist_ok=True)
split = "test"
a = np.load(D / f"{split}.npz"); X, P = a["images"], a["part_masks"] > 0
S = np.load(O / f"part_masks_{split}.npz")["sam3_part_masks"] > 0
meta = {r["id"]: r for r in csv.DictReader(open(D / "metadata.csv"))}
iou = np.array([(S[i] & P[i]).sum() / max((S[i] | P[i]).sum(), 1) for i in range(len(X))])
G = np.array([40, 230, 90.])


def tile(i):
    rgb = np.stack([X[i]] * 3, -1).astype(float)
    rgb[S[i]] = rgb[S[i]] * .78 + G * .22
    e = S[i] & ~ndi.binary_erosion(S[i]); rgb[e] = rgb[e] * .5 + G * .5
    g = P[i] & ~ndi.binary_erosion(P[i]); rgb[g] = 255
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def poster(ids, path, title, cols=8, cell=224):
    rn = -(-len(ids) // cols); im = Image.new("RGB", (cols * cell, 56 + rn * (cell + 16)), (24, 24, 28)); dr = ImageDraw.Draw(im)
    dr.text((10, 18), title, fill=(240, 240, 240))
    for n, i in enumerate(ids):
        x, y = (n % cols) * cell, 56 + (n // cols) * (cell + 16); im.paste(tile(i), (x, y))
        dr.text((x + 3, y + cell + 2), f'{split}_{i:05d} {meta[f"{split}_{i:05d}"]["part_name"]} IoU={iou[i]:.2f}', fill=(190, 190, 190))
    im.save(path, optimize=True)


poster(list(np.linspace(0, len(X) - 1, 48).astype(int)), Q / "poster_sam3_sample.png", "SAM3 part masks, 48 evenly spaced test images. green = SAM3, white line = exact part mask")
poster(list(np.argsort(iou)[:48]), Q / "poster_sam3_worst.png", "SAM3 part masks, the 48 test images with the lowest IoU (0.00 = nothing found)")
print("mean IoU", iou.mean().round(3), "found", (iou > 0).mean().round(3))
