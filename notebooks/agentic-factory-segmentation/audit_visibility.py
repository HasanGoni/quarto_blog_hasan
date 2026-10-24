"""Void visibility audit: how visible is each labelled void in the image a model actually sees?

The generator is deterministic per image, so every image is rebuilt with its stages. For each void:
  signal  = blur(transmitted) - blur(clean)   the exact grey-level change the void causes (no noise, no neighbours needed)
  sigma   = the detector noise at that spot    (noise * (0.6 + 0.8 * signal level), as in the generator)
  clutter = std of the void-free image in a ring around the void (the part's own texture, edges, gradient)
  score   = mean signal * sqrt(area) / sqrt(sigma^2 + clutter^2)   a matched-filter style detectability, in grey levels
Writes void_visibility.csv and image_quality.csv next to the dataset. python audit_visibility.py
"""
import csv, sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from void_dataset import N, image_rng, load_fmnist, make_sample, split_indices

root = Path(__file__).parent
D = root / "data" / "void_dataset_v1"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else D.resolve().parent / "void_dataset_v1_quality"
out.mkdir(parents=True, exist_ok=True)
X, y = load_fmnist(root / "data" / "fashion_mnist")
idx = split_indices(len(X), {"train": 2800, "val": 600, "test": 600})
yy, xx = np.mgrid[:N, :N]
vrows, irows = [], []
for sid, (split, src) in enumerate(idx.items()):
    stored = np.load(D / f"{split}.npz")["images"]
    for i, s in enumerate(src):
        img, void, part, prm, voids, st = make_sample(X[s], image_rng(sid, i), return_stages=True)
        assert (img == stored[i]).all(), (split, i)               # rebuilt image is exactly the stored one
        g = np.cos(prm["grad_angle"]) * (xx - N / 2) / (N / 2) + np.sin(prm["grad_angle"]) * (yy - N / 2) / (N / 2)
        f = lambda t: ndi.gaussian_filter(t * prm["gain"] * (1 + prm["grad"] * g / 1.5), prm["blur"]) * 255
        clean_b, trans_b = f(st["clean"]), f(st["transmitted"])
        signal, sigma_map = trans_b - clean_b, prm["noise"] * (0.6 + 0.8 * trans_b / 255) * 255
        lab, n = ndi.label(void)
        edge_dist = ndi.distance_transform_edt(part)
        scores = []
        for k in range(1, n + 1):
            m = lab == k
            ring = ndi.binary_dilation(m, iterations=8) & ~ndi.binary_dilation(m, iterations=2) & ~void & part.astype(bool)
            d, sg = signal[m].mean(), sigma_map[m].mean()
            cl = clean_b[ring].std() if ring.sum() > 8 else 0.0
            score = d * np.sqrt(m.sum()) / np.sqrt(sg ** 2 + cl ** 2)
            scores.append(score)
            vrows.append(dict(id=f"{split}_{i:05d}", split=split, k=k, area=int(m.sum()), signal=round(d, 2), sigma=round(sg, 2),
                              clutter=round(cl, 2), score=round(score, 2), edge_dist=round(float(edge_dist[m].min()), 1)))
        irows.append(dict(id=f"{split}_{i:05d}", split=split, n_voids=n, min_score=round(min(scores), 2) if scores else "",
                          max_score=round(max(scores), 2) if scores else "", noise=round(prm["noise"], 4), blur=round(prm["blur"], 3),
                          mu=round(prm["mu"], 3), cluster=prm["cluster"]))
    print(split, "rebuilt exactly", len(src))
for name, rows in (("void_visibility.csv", vrows), ("image_quality.csv", irows)):
    with open(out / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(vrows), "voids,", len(irows), "images ->", out)
