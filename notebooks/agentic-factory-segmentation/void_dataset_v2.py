"""Synthetic defect dataset v2: voids AND cracks on Fashion-MNIST silhouettes, every defect clearly visible, no unlabelled look-alikes.

What changed from v1, and why (found by looking at v1's images, see docs in the post):
  * v1 left the Fashion-MNIST prints, stripes and gaps in the part, so grey patches that look like voids existed unlabelled.
    v2 flattens the part's inner texture (smooth thickness) and fills small holes of garments. Only inserted defects are anomalies.
  * v1 chose how much material a void removes without looking at the result; dark thick parts hid them. v2 picks the attenuation so
    every part transmits 15 to 45 percent, and REDRAWS any defect whose visibility score is below T_VIS (score: audit_visibility.py).
  * v2 adds cracks (thin, curved, 3 to 5 px wide). Mask values: 1 = void, 2 = crack.
"""
import gzip
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from PIL import Image

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


def _smooth_field(rng, size=7):
    """A smooth random field in [-1, 1], used to make void outlines irregular."""
    f = np.asarray(Image.fromarray(rng.normal(size=(size, size)).astype(np.float32), mode="F").resize((N, N), Image.BICUBIC))
    return f / (np.abs(f).max() + 1e-9)


T_VIS = 8.0          # minimum visibility score of every labelled defect
GARMENT = {0, 2, 3, 4, 6}  # T-shirt, pullover, dress, coat, shirt: close gaps made by prints and stripes
yy, xx = np.mgrid[:N, :N]


def image_rng(split_id, i):
    return np.random.default_rng([SEED + 2, split_id, i])


def prepare_part(src28, cls):
    """Silhouette + flat thickness. Returns (part mask, thickness) or None if the part is unusable."""
    big = ndi.gaussian_filter(np.asarray(Image.fromarray(src28).resize((N, N), Image.BICUBIC), np.float32) / 255, 2.0)
    part = big > 0.18
    if cls in GARMENT:
        part = ndi.binary_closing(part, iterations=6)
    filled = ndi.binary_fill_holes(part)
    lab, n = ndi.label(filled & ~part)
    for c in range(1, n + 1):
        if (lab == c).sum() < 900:
            part |= lab == c
    # keep the largest piece only
    lab, n = ndi.label(part)
    if n > 1:
        part = lab == (1 + int(np.argmax(ndi.sum(part, lab, range(1, n + 1)))))
    # smooth thickness: normalised blur of the grey level inside the part (kills prints and stripes, keeps broad shading)
    w = ndi.gaussian_filter(part.astype(np.float32), 16)
    sm = ndi.gaussian_filter(np.where(part, big, 0).astype(np.float32), 16) / np.maximum(w, 1e-3)
    thick = np.where(part, 0.45 + 0.55 * np.clip(sm, 0, 1), 0).astype(np.float32)
    inner = ndi.binary_erosion(part, iterations=8)
    if part.sum() < 5000 or (inner & (thick > 0.5)).sum() < 1500:
        return None
    return part, thick, inner


def _blob(rng, cy, cx):
    r = float(np.exp(rng.uniform(np.log(3.0), np.log(12.0))))
    e = rng.uniform(1.0, 2.2); th = rng.uniform(0, np.pi)
    u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th); v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
    d = np.sqrt((u / (r * np.sqrt(e))) ** 2 + (v / (r / np.sqrt(e))) ** 2)
    if rng.random() < 0.4:
        return d <= 1 + 0.4 * _smooth_field(rng)
    return d <= 1.0


def _crack(rng, cy, cx, inner):
    """A thin curved line: a random walk with slowly turning direction, width 3 to 5 px, kept inside `inner`."""
    L = int(rng.uniform(30, 90)); ang = rng.uniform(0, 2 * np.pi); pts = []
    y, x = float(cy), float(cx)
    for _ in range(L // 2):
        iy, ix = int(round(y)), int(round(x))
        if not (0 <= iy < N and 0 <= ix < N and inner[iy, ix]):
            break
        pts.append((iy, ix)); ang += rng.normal(0, 0.28)
        y += 2 * np.sin(ang); x += 2 * np.cos(ang)
    if len(pts) < 14:
        return None
    m = np.zeros((N, N), bool)
    for a, b in zip(pts[:-1], pts[1:]):  # fill the segment between consecutive points
        for t in np.linspace(0, 1, 4):
            m[int(round(a[0] + t * (b[0] - a[0]))), int(round(a[1] + t * (b[1] - a[1])))] = True
    return ndi.binary_dilation(m, iterations=int(rng.integers(2, 4))) & inner


def _score(shape, depth, thick, part, mu, prm, g, clean_b, trans_other):
    """Visibility score of one candidate, same formula as audit_visibility.py (on the image with this defect alone added)."""
    fac = np.where(shape, depth, 1.0).astype(np.float32)
    f = lambda t: ndi.gaussian_filter(t * prm["gain"] * (1 + prm["grad"] * g / 1.5), prm["blur"]) * 255
    trans_b = f(np.where(part, np.exp(-mu * thick * fac), 1.0))
    signal = trans_b - clean_b
    sg = prm["noise"] * (0.6 + 0.8 * trans_b[shape] / 255).mean() * 255
    ring = ndi.binary_dilation(shape, iterations=8) & ~ndi.binary_dilation(shape, iterations=2) & part
    cl = clean_b[ring].std() if ring.sum() > 8 else 0.0
    return float(signal[shape].mean() * np.sqrt(shape.sum()) / np.sqrt(sg ** 2 + cl ** 2))


def make_sample(src28, cls, rng):
    """Returns (image uint8, mask uint8 with 0/1/2, part mask, params, defect records) or None when the source is unusable."""
    prep = prepare_part(src28, cls)
    if prep is None:
        return None
    part, thick, inner = prep
    prm = dict(blur=rng.uniform(0.6, 1.4), noise=rng.uniform(0.01, 0.04), gain=rng.uniform(0.90, 1.0),
               grad=rng.uniform(0.0, 0.08), grad_angle=rng.uniform(0, 2 * np.pi))
    mu = float(rng.uniform(1.0, 2.0) / thick[part].mean())             # part transmits about exp(-1) to exp(-2) of the beam
    g = np.cos(prm["grad_angle"]) * (xx - N / 2) / (N / 2) + np.sin(prm["grad_angle"]) * (yy - N / 2) / (N / 2)
    f = lambda t: ndi.gaussian_filter(t * prm["gain"] * (1 + prm["grad"] * g / 1.5), prm["blur"]) * 255
    clean = np.where(part, np.exp(-mu * thick), 1.0)
    clean_b = f(clean)
    n_target = int(rng.choice([0, 1, 2, 3, 4, 5], p=[0.12, 0.25, 0.22, 0.18, 0.13, 0.10]))
    cy_c, cx_c = np.where(inner & (thick > 0.5))
    factor = np.ones((N, N), np.float32); mask = np.zeros((N, N), np.uint8); keep_out = np.zeros((N, N), bool)
    recs = []
    for _ in range(n_target):
        for _try in range(40):
            k = int(rng.integers(len(cy_c))); cy, cx = cy_c[k], cx_c[k]
            is_crack = rng.random() < 0.3
            shape = _crack(rng, cy, cx, inner) if is_crack else (_blob(rng, cy, cx) & inner)
            if shape is not None:                                       # one defect = one connected piece (a blob clipped by `inner` can split)
                lab, k = ndi.label(shape)
                if k > 1:
                    shape = lab == (1 + int(np.argmax(ndi.sum(shape, lab, range(1, k + 1)))))
            if shape is None or shape.sum() < (40 if is_crack else 30) or (shape & keep_out).any():
                continue
            depth = 1.0 - (rng.uniform(0.5, 0.9) if is_crack else float(np.exp(rng.uniform(np.log(0.3), np.log(0.85)))))
            score = _score(shape, depth, thick, part, mu, prm, g, clean_b, None)
            if score < T_VIS:
                continue
            factor = np.minimum(factor, np.where(shape, depth, 1.0).astype(np.float32))
            mask[shape] = 2 if is_crack else 1
            keep_out |= ndi.binary_dilation(shape, iterations=9)        # a gap of 9 px keeps defects separable after blur
            ys, xs = np.where(shape)
            recs.append(dict(type="crack" if is_crack else "void", area_px=int(shape.sum()), score=round(score, 2), contrast_removed=round(1 - depth, 3),
                             cy=float(ys.mean()), cx=float(xs.mean()), bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]))
            break
    trans = np.where(part, np.exp(-mu * thick * factor), 1.0)
    blurred = f(trans) / 255
    img = np.round(np.clip(blurred + rng.normal(0, prm["noise"] * (0.6 + 0.8 * blurred)), 0, 1) * 255).astype(np.uint8)
    prm.update(mu=mu, n_defects=len(recs), n_voids=sum(r["type"] == "void" for r in recs), n_cracks=sum(r["type"] == "crack" for r in recs),
               part_area=int(part.sum()), min_score=min((r["score"] for r in recs), default=0.0))
    return img, mask, part.astype(np.uint8), prm, recs
