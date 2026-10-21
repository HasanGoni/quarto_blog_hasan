"""Code that runs INSIDE the kernel (it is sent as text, then exec'd). Written for this series; numpy + scipy only.

show()  : lets the model look at an array. The host reads the saved PNG and attaches it to the next message.
The rest: a small inspection tool API (part_mask, label_blobs, overlay, crop). Primitives, not a detector:
the model still has to decide what counts as a void.
"""
import os as _os
import numpy as _np
import scipy.ndimage as _ndi
from PIL import Image as _Image

_SHOWN_DIR = "shown"
_n_shown = 0


def _to_uint8(x):
    x = _np.asarray(x)
    if x.dtype == bool:
        return (x * 255).astype(_np.uint8)
    if x.dtype != _np.uint8:
        x = x.astype(float)
        lo, hi = float(x.min()), float(x.max())
        x = (x - lo) / (hi - lo + 1e-9) * 255
        return x.astype(_np.uint8)
    return x


def show(x, title=""):
    """Look at an array: 2-D (grayscale) or H x W x 3 (RGB). Shown at 2x size. At most 2 images per turn."""
    global _n_shown
    a = _to_uint8(x)
    if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] != 3):
        raise ValueError(f"show() needs a 2-D or HxWx3 array, got shape {a.shape}")
    _os.makedirs(_SHOWN_DIR, exist_ok=True)
    _n_shown += 1
    h, w = a.shape[:2]
    _Image.fromarray(a).resize((w * 2, h * 2), _Image.NEAREST).save(f"{_SHOWN_DIR}/{_n_shown:03d}.png")
    with open(f"{_SHOWN_DIR}/{_n_shown:03d}.txt", "w") as f:
        f.write(str(title)[:80])
    print(f"[show #{_n_shown}] {str(title)[:80]}  (shape {tuple(a.shape)})")


def part_mask(image):
    """Boolean silhouette of the part (holes filled). Assumes the part is DARKER than the background (Otsu threshold)."""
    g = _np.asarray(image, dtype=float)
    hist, edges = _np.histogram(g, bins=64, range=(0, 255))
    p = hist / hist.sum()
    c = (edges[:-1] + edges[1:]) / 2
    w0, w1 = _np.cumsum(p), 1 - _np.cumsum(p)
    m0 = _np.cumsum(p * c) / _np.maximum(w0, 1e-9)
    m1 = (_np.sum(p * c) - _np.cumsum(p * c)) / _np.maximum(w1, 1e-9)
    t = c[_np.argmax(w0 * w1 * (m0 - m1) ** 2)]
    m = g < t
    lab, n = _ndi.label(m)
    if n == 0:
        return m
    sizes = _ndi.sum(m, lab, range(1, n + 1))
    return _ndi.binary_fill_holes(lab == (1 + int(_np.argmax(sizes))))


def label_blobs(mask, min_area=3):
    """Connected components of a boolean mask. Returns (label_image, list of dicts), biggest first.
    Each dict: id, area (pixels), cy, cx, bbox (y0, x0, y1, x1), elongation (1 = round, large = line-like)."""
    lab, n = _ndi.label(mask)
    out = []
    for i, sl in enumerate(_ndi.find_objects(lab), start=1):
        ys, xs = _np.nonzero(lab[sl] == i)
        if len(ys) < min_area:
            continue
        ys, xs = ys + sl[0].start, xs + sl[1].start
        ev = _np.linalg.eigvalsh(_np.cov(_np.vstack([ys, xs]))) if len(ys) > 2 else _np.array([1.0, 1.0])
        out.append(dict(id=i, area=int(len(ys)), cy=round(float(ys.mean()), 1), cx=round(float(xs.mean()), 1),
                        bbox=(int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1),
                        elongation=round(float(_np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9))), 2)))
    out.sort(key=lambda d: -d["area"])
    return lab, out


_COLORS = [(255, 60, 60), (60, 200, 60), (70, 120, 255), (255, 170, 0)]


def overlay(image, masks):
    """RGB picture: the grayscale image with the outline of each boolean mask in a colour. masks: list of masks."""
    rgb = _np.stack([_to_uint8(image)] * 3, axis=-1).copy()
    for k, m in enumerate(masks):
        edge = _np.asarray(m, dtype=bool) & ~_ndi.binary_erosion(_np.asarray(m, dtype=bool))
        rgb[edge] = _COLORS[k % len(_COLORS)]
    return rgb


def crop(image, cy, cx, r=24, zoom=4):
    """A (2r x 2r) window centred on (cy, cx), enlarged by `zoom`. Pass the result to show()."""
    y0, x0 = max(0, int(cy) - r), max(0, int(cx) - r)
    w = _np.asarray(image)[y0:int(cy) + r, x0:int(cx) + r]
    return _np.kron(w, _np.ones((zoom, zoom) + (1,) * (w.ndim - 2), dtype=w.dtype))
