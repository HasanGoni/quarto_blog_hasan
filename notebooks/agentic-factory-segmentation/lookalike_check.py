"""How many unlabelled void-like patches does the DEFECT-FREE part contain?  python lookalike_check.py
Residual = defect-free image minus its local mean (normalised blur inside the part, sigma 10). A look-alike is a connected patch of
residual > 8 grey levels, >= 30 px, at least 8 px inside the part, with the same visibility score formula (noise only, no clutter) >= 8.
Run on the first 600 images of val for v1 and v2, using each version's own generator."""
import numpy as np
from pathlib import Path
from scipy import ndimage as ndi
import void_dataset as v1, void_dataset_v2 as v2

root = Path(__file__).parent
X, y = v1.load_fmnist(root / "data" / "fashion_mnist")


def count(clean_b, part, noise, blurred, return_cents=False):
    inner = ndi.binary_erosion(part, iterations=8)
    w = ndi.gaussian_filter(part.astype(float), 10)
    loc = ndi.gaussian_filter(np.where(part, clean_b, 0), 10) / np.maximum(w, 1e-3)
    r = np.where(inner, clean_b - loc, 0)
    lab, n = ndi.label(r > 8); k = 0; cents = []
    for c in range(1, n + 1):
        m = lab == c
        if m.sum() >= 30:
            sg = noise * (0.6 + 0.8 * blurred[m].mean() / 255) * 255
            if r[m].mean() * np.sqrt(m.sum()) / sg >= 8:
                k += 1; cents.append(tuple(np.argwhere(m).mean(0)))
    return (k, cents) if return_cents else k


def run_v1(n=600):
    idx = v1.split_indices(len(X), {"train": 2800, "val": 600, "test": 600})["val"]; out = []
    yy, xx = np.mgrid[:224, :224]
    for i in range(n):
        img, void, part, prm, voids, st = v1.make_sample(X[idx[i]], v1.image_rng(1, i), return_stages=True)
        g = np.cos(prm["grad_angle"]) * (xx - 112) / 112 + np.sin(prm["grad_angle"]) * (yy - 112) / 112
        cb = ndi.gaussian_filter(st["clean"] * prm["gain"] * (1 + prm["grad"] * g / 1.5), prm["blur"]) * 255
        out.append(count(cb, part.astype(bool), prm["noise"], cb))
    return np.array(out)


def run_v2(n=600):
    perm = np.random.default_rng(v2.SEED + 2).permutation(len(X)); pos = 2800 + 0; out = []
    # re-walk the exact source order used by make_dataset_v2 (train consumed 2800 usable + skipped ones)
    import csv
    meta = list(csv.DictReader(open(root / "data/void_dataset_v2/metadata.csv")))
    for r in [m for m in meta if m["split"] == "val"][:n]:
        s = int(r["src_index"]); i = int(r["id"].split("_")[1])
        part, thick, inner = v2.prepare_part(X[s], int(r["part_class"]))
        mu = float(r["mu"])
        rng = v2.image_rng(1, i)
        prm = dict(blur=rng.uniform(0.6, 1.4), noise=rng.uniform(0.01, 0.04), gain=rng.uniform(0.90, 1.0), grad=rng.uniform(0.0, 0.08), grad_angle=rng.uniform(0, 2 * np.pi))
        g = np.cos(prm["grad_angle"]) * (v2.xx - 112) / 112 + np.sin(prm["grad_angle"]) * (v2.yy - 112) / 112
        cb = ndi.gaussian_filter(np.where(part, np.exp(-mu * thick), 1.0) * prm["gain"] * (1 + prm["grad"] * g / 1.5), prm["blur"]) * 255
        out.append(count(cb, part, prm["noise"], cb))
    return np.array(out)


if __name__ == "__main__":
  for name, f in (("v1", run_v1), ("v2", run_v2)):
    a = f()
    print(f"{name}: images with at least one unlabelled look-alike {np.mean(a > 0) * 100:.1f}%  mean per image {a.mean():.2f}  (600 val images)")
