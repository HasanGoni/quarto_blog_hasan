"""A synthetic void dataset with exact masks, built from Fashion-MNIST silhouettes.

Recipe, one image at a time (every random choice is recorded in the metadata, because each one is a slice axis later):
  1. take a Fashion-MNIST item (28x28) and upsample it to 224x224: this is the PART (its silhouette is the part mask)
  2. read the grey level as THICKNESS: more grey, more material
  3. insert 0 to 5 VOIDS inside the part: each removes a fraction of the local thickness; the void mask is the shape inserted
  4. simulate attenuation (Beer-Lambert): transmitted intensity = exp(-mu * thickness); air transmits everything
  5. add detector effects: gain drift, blur, signal-dependent noise
The void mask is exact by construction. A void can never lie outside the part (the containment rule).
"""
import gzip
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

N = 224                      # image side in pixels
SEED = 20261017
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {  # md5 values as listed in the Fashion-MNIST README
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


def load_fmnist(data_dir: Path):
    """Download (once) and checksum-verify Fashion-MNIST, return all 70,000 images and labels. Read as plain bytes, no pickle."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, md5 in FILES.items():
        path = data_dir / name
        if not path.exists():
            urllib.request.urlretrieve(BASE + name, path)
        got = hashlib.md5(path.read_bytes()).hexdigest()
        if got != md5:
            raise RuntimeError(f"{name}: md5 {got} != expected {md5}")
    def idx(name, off, shape):
        return np.frombuffer(gzip.open(data_dir / name).read(), np.uint8, offset=off).reshape(shape)
    X = np.concatenate([idx("train-images-idx3-ubyte.gz", 16, (-1, 28, 28)), idx("t10k-images-idx3-ubyte.gz", 16, (-1, 28, 28))])
    y = np.concatenate([idx("train-labels-idx1-ubyte.gz", 8, (-1,)), idx("t10k-labels-idx1-ubyte.gz", 8, (-1,))])
    return X, y


def image_rng(split_id: int, i: int) -> np.random.Generator:
    """One independent generator per image, so the dataset does not depend on generation order."""
    return np.random.default_rng([SEED, split_id, i])


def _smooth_field(rng, size=7):
    """A smooth random field in [-1, 1], used to make void outlines irregular."""
    f = np.asarray(Image.fromarray(rng.normal(size=(size, size)).astype(np.float32), mode="F").resize((N, N), Image.BICUBIC))
    return f / (np.abs(f).max() + 1e-9)


def make_sample(src28: np.ndarray, rng: np.random.Generator, return_stages: bool = False):
    """Returns image (uint8), void mask (uint8 0/1), part mask (uint8 0/1), per-image params, per-void records."""
    prm = dict(mu=rng.uniform(1.6, 3.0), blur=rng.uniform(0.6, 1.8), noise=rng.uniform(0.01, 0.05),
               gain=rng.uniform(0.90, 1.0), grad=rng.uniform(0.0, 0.08), grad_angle=rng.uniform(0, 2 * np.pi))
    big = np.asarray(Image.fromarray(src28).resize((N, N), Image.BICUBIC), np.float32) / 255
    big = ndi.gaussian_filter(big, 2.0)
    part = big > 0.18
    inner = ndi.binary_erosion(part, iterations=3)               # voids stay clear of the part's edge
    thick = part * (0.35 + 0.65 * big)

    n_target = int(rng.choice([0, 1, 2, 3, 4, 5], p=[0.15, 0.25, 0.22, 0.18, 0.12, 0.08]))
    prm["cluster"] = bool(rng.random() < 0.2 and n_target >= 2)  # porosity-like groups of voids
    cand_y, cand_x = np.where(inner & (big > 0.35))
    yy, xx = np.mgrid[:N, :N]
    factor = np.ones((N, N), np.float32)                        # remaining thickness fraction at every pixel
    kinds, anchor = [], None
    for _ in range(n_target if len(cand_y) else 0):
        if prm["cluster"] and anchor is not None:
            near = np.where((cand_y - anchor[0]) ** 2 + (cand_x - anchor[1]) ** 2 < 30 ** 2)[0]
            k = int(rng.choice(near)) if len(near) else int(rng.integers(len(cand_y)))
        else:
            k = int(rng.integers(len(cand_y)))
        cy, cx = cand_y[k], cand_x[k]
        anchor = anchor or (cy, cx)
        r = float(np.exp(rng.uniform(np.log(2.0), np.log(12.0))))   # radius, log-uniform: many small, few large
        e = rng.uniform(1.0, 2.5); th = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th); v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        d = np.sqrt((u / (r * np.sqrt(e))) ** 2 + (v / (r / np.sqrt(e))) ** 2)
        kind = "blob" if rng.random() < 0.4 else "ellipse"
        shape = d <= (1 + 0.45 * _smooth_field(rng) if kind == "blob" else 1.0)
        removed = float(np.exp(rng.uniform(np.log(0.05), np.log(0.85))))   # fraction of the local thickness the void removes, log-uniform:
        depth = 1.0 - removed                                              # many faint voids (5% removed), few obvious ones (85% removed)
        factor = np.minimum(factor, np.where(shape & inner, depth, 1.0).astype(np.float32))
        kinds.append(kind)

    void_all = factor < 1
    labels, n_comp = ndi.label(void_all)
    keep = np.zeros_like(void_all)
    for c in range(1, n_comp + 1):
        if (labels == c).sum() >= 6:                              # drop specks too small to be a defect after clipping
            keep |= labels == c
    factor = np.where(keep, factor, 1.0).astype(np.float32)
    void = keep
    labels, n_comp = ndi.label(void)

    clean = np.where(part, np.exp(-prm["mu"] * thick), 1.0)
    trans = np.where(part, np.exp(-prm["mu"] * thick * factor), 1.0)
    g = np.cos(prm["grad_angle"]) * (xx - N / 2) / (N / 2) + np.sin(prm["grad_angle"]) * (yy - N / 2) / (N / 2)
    trans_g = trans * prm["gain"] * (1 + prm["grad"] * g / 1.5)
    blurred = ndi.gaussian_filter(trans_g, prm["blur"])
    noisy = np.clip(blurred + rng.normal(0, prm["noise"] * (0.6 + 0.8 * blurred)), 0, 1)
    img = np.round(noisy * 255).astype(np.uint8)

    voids = []
    for c in range(1, n_comp + 1):
        m = labels == c
        ys, xs = np.where(m)
        voids.append(dict(area_px=int(m.sum()), eq_diam_px=float(2 * np.sqrt(m.sum() / np.pi)), cy=float(ys.mean()), cx=float(xs.mean()),
                          contrast=float(np.mean(trans[m] - clean[m])), bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]))
    prm.update(n_voids=len(voids), void_area=int(void.sum()), part_area=int(part.sum()),
               max_eq_diam=max((v["eq_diam_px"] for v in voids), default=0.0), min_contrast=min((v["contrast"] for v in voids), default=0.0))
    out = (img, void.astype(np.uint8), part.astype(np.uint8), prm, voids)
    if return_stages:
        return out + (dict(upsampled=big, part=part, thickness=thick, factor=factor, clean=clean, transmitted=trans, final=img / 255.0),)
    return out


def split_indices(total_sources: int, sizes: dict[str, int]):
    """Assign each source item to at most one split, so no part shape is shared between splits."""
    perm = np.random.default_rng(SEED).permutation(total_sources)
    out, start = {}, 0
    for name, n in sizes.items():
        out[name] = perm[start:start + n]
        start += n
    return out
