"""Figures for the dataset post, all drawn from the built dataset. python make_figures.py"""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from void_dataset import N, image_rng, load_fmnist, make_sample

root = Path(__file__).parent
d = root / "data" / "void_dataset_v1"
OUT = root.parent.parent / "posts/series/agentic-factory-segmentation/images"
OUT.mkdir(parents=True, exist_ok=True)
X, y = load_fmnist(root / "data" / "fashion_mnist")
meta = list(csv.DictReader(open(d / "metadata.csv")))
voids = [json.loads(l) for l in (d / "voids.jsonl").read_text().splitlines()]
sp = {s: {k: v for k, v in np.load(d / f"{s}.npz").items()} for s in ("train", "val", "test")}
rows = {s: [m for m in meta if m["split"] == s] for s in sp}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9})

# A. the recipe, step by step, for one real item of the dataset
i = next(k for k, m in enumerate(rows["train"]) if m["part_name"] == "Ankle boot" and int(m["n_voids"]) == 3 and float(m["min_contrast"]) > 0.07)
sid = 0
img, void, part, prm, vs, st = make_sample(X[int(rows["train"][i]["src_index"])], image_rng(sid, i), return_stages=True)
assert np.array_equal(img, sp["train"]["images"][i])               # this IS the stored image, not a lookalike
panels = [("1. Fashion-MNIST item\n(28x28, shown larger)", X[int(rows["train"][i]["src_index"])], "gray", "nearest"),
          ("2. upsampled to 224x224,\nsmoothed", st["upsampled"], "gray", None), ("3. part mask\n(its silhouette)", st["part"], "gray", None),
          ("4. grey level read as\nthickness", st["thickness"], "gray", None),
          ("5. voids remove thickness\n(dark = more removed)", st["factor"], "gray", None), ("6. attenuation,\nexp(-mu * thickness)", st["transmitted"], "gray", None),
          ("7. + gain drift, blur, noise:\nthe image the model sees", st["final"], "gray", None), ("8. the exact void mask\n(red outline)", st["final"], "gray", None)]
fig, ax = plt.subplots(2, 4, figsize=(11, 7.0))
for a, (t, im, cm, interp) in zip(ax.ravel(), panels):
    a.imshow(im, cmap=cm, vmin=0, vmax=1 if im.dtype != np.uint8 else 255, interpolation=interp or "bilinear"); a.set_title(t); a.axis("off")
ax.ravel()[7].contour(void, levels=[0.5], colors="red", linewidths=1.2)
fig.suptitle(f"One training image, built step by step  (ankle boot, {prm['n_voids']} voids, mu={prm['mu']:.2f}, blur={prm['blur']:.2f}px, noise={prm['noise']:.3f})", y=0.995)
fig.tight_layout(h_pad=2.4); fig.savefig(OUT / "voidset-recipe.png", dpi=110); plt.close(fig)

# B. a sample of the test split with the exact masks
fig, ax = plt.subplots(4, 6, figsize=(11, 7.6)); rng = np.random.default_rng(5)
pick = rng.choice(len(rows["test"]), 24, replace=False)
for a, k in zip(ax.ravel(), pick):
    a.imshow(sp["test"]["images"][k], cmap="gray", vmin=0, vmax=255)
    if sp["test"]["void_masks"][k].any(): a.contour(sp["test"]["void_masks"][k], levels=[0.5], colors="red", linewidths=0.9)
    nvk = int(rows['test'][k]['n_voids']); a.set_title(f"{rows['test'][k]['part_name']}, {nvk} void{'s' if nvk != 1 else ''}", fontsize=8); a.axis("off")
fig.suptitle("24 random images from the test split, exact void masks in red", y=0.995); fig.tight_layout(); fig.savefig(OUT / "voidset-samples.png", dpi=105); plt.close(fig)

# C. distributions of everything that was randomised
ed = np.array([v["eq_diam_px"] for v in voids]); ct = np.array([v["contrast"] for v in voids])
fig, ax = plt.subplots(2, 3, figsize=(11, 5.8))
nv = np.array([int(m["n_voids"]) for m in meta]); ax[0, 0].bar(range(6), [int((nv == k).sum()) for k in range(6)], color="#4c78a8"); ax[0, 0].set_title("voids per image"); ax[0, 0].set_xlabel("number of voids")
ax[0, 1].hist(ed, bins=np.geomspace(3, 30, 30), color="#4c78a8"); ax[0, 1].set_xscale("log"); ax[0, 1].set_title("void size (equivalent diameter)"); ax[0, 1].set_xlabel("pixels, log scale")
ax[0, 2].hist(ct, bins=np.geomspace(0.003, 0.7, 30), color="#4c78a8"); ax[0, 2].set_xscale("log"); ax[0, 2].set_title("void contrast (brightening against the part)"); ax[0, 2].set_xlabel("transmission difference, log scale")
ax[1, 0].hist([float(m["noise"]) for m in meta], bins=25, color="#59a14f"); ax[1, 0].set_title("detector noise level")
ax[1, 1].hist([float(m["blur"]) for m in meta], bins=25, color="#59a14f"); ax[1, 1].set_title("detector blur (sigma, pixels)")
ax[1, 2].hist([float(m["mu"]) for m in meta], bins=25, color="#59a14f"); ax[1, 2].set_title("part opacity (mu)")
fig.suptitle("Everything that is randomised, over all 4,000 images (7,794 voids)", y=0.995); fig.tight_layout(); fig.savefig(OUT / "voidset-distributions.png", dpi=105); plt.close(fig)

# D. faint to obvious: void crops at fixed contrast percentiles, taken from the test split
tv = [v for v in voids if v["split"] == "test"]; tv.sort(key=lambda v: v["contrast"])
fig, ax = plt.subplots(1, 6, figsize=(11, 2.4))
for a, p in zip(ax, (3, 15, 35, 60, 85, 98)):
    v = tv[int(len(tv) * p / 100)]; k = int(v["id"].split("_")[1]); cy, cx = int(v["cy"]), int(v["cx"]); h = 24
    y0, x0 = min(max(cy - h, 0), N - 2 * h), min(max(cx - h, 0), N - 2 * h)
    a.imshow(sp["test"]["images"][k][y0:y0 + 2 * h, x0:x0 + 2 * h], cmap="gray", vmin=0, vmax=255)
    a.contour(sp["test"]["void_masks"][k][y0:y0 + 2 * h, x0:x0 + 2 * h], levels=[0.5], colors="red", linewidths=0.9)
    a.set_title(f"contrast {v['contrast']:.3f}\n(~{v['contrast'] * 255:.0f} of 255)", fontsize=8); a.axis("off")
fig.suptitle("The same dataset from faint to obvious voids (test split, percentiles 3 to 98 of contrast)", y=1.04, fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "voidset-contrast-range.png", dpi=110, bbox_inches="tight"); plt.close(fig)

# E. near-twin part shapes between test and train (Fashion-MNIST contains many look-alikes)
src = {s: np.array(sorted(int(m["src_index"]) for m in rows[s])) for s in sp}
tr, te = X[src["train"]].reshape(len(src["train"]), -1) / 255.0, X[src["test"]].reshape(len(src["test"]), -1) / 255.0
D = np.sqrt(np.maximum((te ** 2).sum(1)[:, None] + (tr ** 2).sum(1)[None] - 2 * te @ tr.T, 0)); nn = D.argmin(1); nd = D.min(1)
order = np.argsort(nd)[[0, 1, 2, 3, 4, 5]]
fig, ax = plt.subplots(2, 6, figsize=(11, 4.1))
for c, t in enumerate(order):
    ax[0, c].imshow(X[src["test"][t]], cmap="gray"); ax[0, c].set_title(f"test source\ndistance {nd[t]:.2f}", fontsize=8)
    ax[1, c].imshow(X[src["train"][nn[t]]], cmap="gray"); ax[1, c].set_title("its nearest train source", fontsize=8)
    ax[0, c].axis("off"); ax[1, c].axis("off")
fig.suptitle("The six closest test-to-train pairs of part shapes: no source item is shared, but look-alikes exist", y=1.0, fontsize=9)
fig.tight_layout(h_pad=2.0); fig.savefig(OUT / "voidset-near-twins.png", dpi=110, bbox_inches="tight"); plt.close(fig)
print("figures written to", OUT)
