"""Overlay images and posters for v2.  python make_posters_v2.py
Overlay = unchanged image + very light tint and thin outline: red = void, cyan = crack. Folders by content:
void_only, crack_only, void_and_crack, clean (no defect). Each folder: one PNG per image + numbered contact-sheet pages.
poster_<folder>.png = 48 images evenly spaced by id; poster_faintest.png = the 48 images whose WEAKEST defect has the lowest score."""
import csv, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

D = Path(__file__).parent / "data" / "void_dataset_v2"
Q = D.resolve().parent / "void_dataset_v2_posters"; Q.mkdir(exist_ok=True)
data = {s: np.load(D / f"{s}.npz") for s in ("train", "val", "test")}
data = {s: (d["images"], d["defect_masks"]) for s, d in data.items()}
meta = list(csv.DictReader(open(D / "metadata.csv")))
col = {1: np.array([255, 60, 60.]), 2: np.array([40, 220, 255.])}


def overlay(sp, i, a=0.22):
    rgb = np.stack([data[sp][0][i]] * 3, -1).astype(float)
    for c, t in col.items():
        m = data[sp][1][i] == c; e = m & ~ndi.binary_erosion(m)
        rgb[m] = rgb[m] * (1 - a) + t * a; rgb[e] = rgb[e] * .55 + t * .45
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def poster(items, path, cols, title, cell=224):
    rn = -(-len(items) // cols); P = Image.new("RGB", (cols * cell, 56 + rn * (cell + 16)), (24, 24, 28)); dr = ImageDraw.Draw(P)
    dr.text((10, 18), title, fill=(240, 240, 240))
    for n, r in enumerate(items):
        x, y = (n % cols) * cell, 56 + (n // cols) * (cell + 16)
        P.paste(overlay(r["split"], int(r["id"].split("_")[1])), (x, y))
        dr.text((x + 3, y + cell + 2), f'{r["id"]} v={r["n_voids"]} c={r["n_cracks"]} min={r["min_score"]}', fill=(190, 190, 190))
    P.save(path, optimize=True)


def kind(r):
    v, c = int(r["n_voids"]), int(r["n_cracks"])
    return "clean" if v + c == 0 else "void_only" if c == 0 else "crack_only" if v == 0 else "void_and_crack"


groups = {}
for r in meta:
    groups.setdefault(kind(r), []).append(r)
for g, items in groups.items():
    d = Q / g; d.mkdir(exist_ok=True)
    for r in items:
        overlay(r["split"], int(r["id"].split("_")[1])).save(d / f'{r["id"]}.png', optimize=True)
    pick = [items[int(k)] for k in np.linspace(0, len(items) - 1, min(48, len(items)))]
    poster(pick, Q / f"poster_{g}.png", 8, f"{g}: {len(items)} images, 48 evenly spaced. red = void, cyan = crack (light overlay)")
    for p in range(0, len(items), 120):
        poster(items[p:p + 120], d / f"page_{p // 120 + 1:02d}.jpg", 12, f"{g}: images {p + 1} to {min(p + 120, len(items))} of {len(items)}")
    print(g, len(items))
withd = sorted([r for r in meta if kind(r) != "clean"], key=lambda r: float(r["min_score"]))[:48]
poster(withd, Q / "poster_faintest.png", 8, "faintest: the 48 images whose weakest defect has the lowest visibility score (all >= 8)")
