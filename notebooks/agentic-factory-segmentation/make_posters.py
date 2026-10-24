"""Overlay images and posters, sorted by how visible the labelled voids really are.  python make_posters.py
Reads void_visibility.csv / image_quality.csv (audit_visibility.py). Per image the overlay is the unchanged image with a very light
red tint on the void masks plus a thin outline. Tiers (per void score, see audit_visibility.py):
  bad       at least one labelled void scores below T_BAD: the label is there, the evidence in the image is not
  confusing no void below T_BAD, but at least one below T_OK: faint, small, or sitting in strong texture
  good      every void scores at least T_OK
  clean     no void at all (kept apart: nothing to overlay)
"""
import csv, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

T_BAD, T_OK = 1.0, 4.0
root = Path(__file__).parent
D = root / "data" / "void_dataset_v1"
Q = D.resolve().parent / "void_dataset_v1_quality"
rows = list(csv.DictReader(open(Q / "image_quality.csv")))
for r in rows:
    r["min_score"] = float(r["min_score"]) if r["min_score"] != "" else None
    r["tier"] = "clean" if r["min_score"] is None else "bad" if r["min_score"] < T_BAD else "confusing" if r["min_score"] < T_OK else "good"
with open(Q / "image_tiers.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

data = {s: np.load(D / f"{s}.npz") for s in ("train", "val", "test")}
data = {s: (d["images"], d["void_masks"] > 0) for s, d in data.items()}
TINT = np.array([255, 60, 60], float)


def overlay(split, i, alpha=0.22):
    img, vm = data[split][0][i], data[split][1][i]
    rgb = np.stack([img] * 3, -1).astype(float)
    edge = vm & ~ndi.binary_erosion(vm)
    rgb[vm] = rgb[vm] * (1 - alpha) + TINT * alpha
    rgb[edge] = rgb[edge] * 0.55 + TINT * 0.45
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def poster(items, path, cols, title, cell=224):
    rowsn = -(-len(items) // cols); head = 56
    P = Image.new("RGB", (cols * cell, head + rowsn * (cell + 16)), (24, 24, 28)); dr = ImageDraw.Draw(P)
    dr.text((10, 18), title, fill=(240, 240, 240))
    for n, r in enumerate(items):
        x, y = (n % cols) * cell, head + (n // cols) * (cell + 16)
        sp, i = r["split"], int(r["id"].split("_")[1])
        P.paste(overlay(sp, i), (x, y))
        dr.text((x + 3, y + cell + 2), f'{r["id"]}  n={r["n_voids"]}  min={r["min_score"]}', fill=(190, 190, 190))
    P.save(path, optimize=True)


for tier in ("good", "bad", "confusing", "clean"):
    d = Q / tier; d.mkdir(exist_ok=True)
    items = sorted([r for r in rows if r["tier"] == tier], key=lambda r: (r["min_score"] if r["min_score"] is not None else 0, r["id"]))
    for r in items:
        overlay(r["split"], int(r["id"].split("_")[1])).save(d / f'{r["id"]}.png', optimize=True)
    # a readable poster: 8 x 6 = 48 evenly spaced across the tier, sorted by score; plus every image on numbered pages
    pick = [items[int(k)] for k in np.linspace(0, len(items) - 1, min(48, len(items)))]
    poster(pick, Q / f"poster_{tier}.png", 8, f"{tier}: {len(items)} images, 48 evenly spaced by score (low to high). light red = labelled void")
    for p in range(0, len(items), 120):
        poster(items[p:p + 120], d / f"page_{p // 120 + 1:02d}.jpg", 12, f"{tier}: images {p + 1} to {min(p + 120, len(items))} of {len(items)}", cell=224)
    print(tier, len(items))
