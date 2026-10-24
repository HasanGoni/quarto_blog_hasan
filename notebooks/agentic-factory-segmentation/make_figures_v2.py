"""Figures for the v2 section of the dataset post.  python make_figures_v2.py"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import void_dataset as v1
import void_dataset_v2 as v2
from lookalike_check import count

root = Path(__file__).parent
IMG = root.parent.parent / "posts/series/agentic-factory-segmentation/images"
X, y = v1.load_fmnist(root / "data" / "fashion_mnist")
yy, xx = np.mgrid[:224, :224]
RED, CY = np.array([255, 60, 60.]), np.array([40, 220, 255.])


def tint(img, masks, a=0.22):
    rgb = np.stack([img] * 3, -1).astype(float) / 255
    for m, c in masks:
        e = m & ~ndi.binary_erosion(m); rgb[m] = rgb[m] * (1 - a) + c / 255 * a; rgb[e] = rgb[e] * .5 + c / 255 * .5
    return rgb


# ---- problem figure: three real v1 validation images, plain | with labels | what is wrong
idx = v1.split_indices(len(X), {"train": 2800, "val": 600, "test": 600})["val"]
vis = {}
for r in csv.DictReader(open((root / "data" / "void_dataset_v1").resolve().parent / "void_dataset_v1_quality" / "void_visibility.csv")):
    if r["split"] == "val":
        vis.setdefault(int(r["id"].split("_")[1]), []).append(float(r["score"]))
picks = {"look-alike": [], "invisible": []}
for i in range(600):
    img, void, part, prm, voids, st = v1.make_sample(X[idx[i]], v1.image_rng(1, i), return_stages=True)
    g = np.cos(prm["grad_angle"]) * (xx - 112) / 112 + np.sin(prm["grad_angle"]) * (yy - 112) / 112
    cb = ndi.gaussian_filter(st["clean"] * prm["gain"] * (1 + prm["grad"] * g / 1.5), prm["blur"]) * 255
    k, cents = count(cb, part.astype(bool), prm["noise"], cb, return_cents=True)
    if k >= 1 and len(voids) >= 1 and len(picks["look-alike"]) < 2 and i > 40:
        picks["look-alike"].append((i, img, void > 0, cents))
    if vis.get(i) and min(vis[i]) < 0.5 and len(picks["invisible"]) < 1 and i > 10:
        lab, n = ndi.label(void > 0); sc = vis[i]; w = int(np.argmin(sc)) + 1
        picks["invisible"].append((i, img, void > 0, [tuple(np.argwhere(lab == w).mean(0))]))
    if len(picks["look-alike"]) == 2 and picks["invisible"]:
        break
items = picks["look-alike"] + picks["invisible"]
fig, ax = plt.subplots(2, 3, figsize=(11, 7.6))
for j, (i, img, void, cents) in enumerate(items):
    ax[0, j].imshow(img, cmap="gray", vmin=0, vmax=255); ax[0, j].set_title(f"v1 val_{i:05d}, as a model sees it", fontsize=9)
    ax[1, j].imshow(tint(img, [(void, RED)]))
    for cy, cx in cents:
        ax[1, j].add_patch(plt.Circle((cx, cy), 14, fill=False, ec="white", lw=1.4, ls="--"))
    ax[1, j].set_title("red = labelled void, white circle = " + ("unlabelled look-alike" if j < 2 else "a labelled void you cannot see"), fontsize=8)
    ax[0, j].axis("off"); ax[1, j].axis("off")
plt.tight_layout(); plt.savefig(IMG / "voidset2-problem.png", dpi=110); plt.close()

# ---- v2 samples: plain next to overlay
D2 = root / "data/void_dataset_v2"
a = np.load(D2 / "test.npz"); Xi, M = a["images"], a["defect_masks"]
ids = [i for i in range(len(Xi)) if (M[i] > 0).any()][::40][:6]
fig, ax = plt.subplots(2, 6, figsize=(15, 5.4))
for j, i in enumerate(ids):
    ax[0, j].imshow(Xi[i], cmap="gray", vmin=0, vmax=255); ax[0, j].set_title(f"test_{i:05d}", fontsize=9)
    ax[1, j].imshow(tint(Xi[i], [(M[i] == 1, RED), (M[i] == 2, CY)]))
    ax[0, j].axis("off"); ax[1, j].axis("off")
plt.suptitle("v2: top row is the image as a model sees it, bottom row adds the labels (red void, cyan crack)", fontsize=10)
plt.tight_layout(); plt.savefig(IMG / "voidset2-samples.png", dpi=100); plt.close()
print("ok", [(i) for i, *_ in items])
