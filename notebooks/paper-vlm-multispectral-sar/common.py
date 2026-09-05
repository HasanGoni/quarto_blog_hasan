"""Render So2Sat-LCZ42's real Sentinel-1 (SAR) + Sentinel-2 (multispectral) patches as
five named optical views + one named SAR view -- the paper's core trick: reuse a general VLM's
existing multi-image interface instead of building a dedicated multispectral/SAR encoder.
"""
import numpy as np
from PIL import Image

LCZ_CLASSES = [
    "Compact high-rise", "Compact midrise", "Compact low-rise",
    "Open high-rise", "Open midrise", "Open low-rise",
    "Lightweight low-rise", "Large low-rise", "Sparsely built", "Heavy industry",
    "Dense trees", "Scattered trees", "Bush and scrub", "Low plants",
    "Bare rock or paved", "Bare soil or sand", "Water",
]

# Sentinel-2 band order in sen2: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 (indices 0-9)
S2_VIEWS = [
    ("True color (B4-B3-B2)", [2, 1, 0]),
    ("Near-infrared false color (B8-B4-B3)", [6, 2, 1]),
    ("Short-wave infrared (B12-B11-B8A)", [9, 8, 7]),
    ("Red-edge composite (B7-B6-B5)", [5, 4, 3]),
    ("Vegetation red-edge (B8A)", [7, 7, 7]),
]


def _stretch(band: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(band, 2), np.percentile(band, 98)
    band = np.clip((band - lo) / (hi - lo + 1e-8), 0, 1)
    return (band * 255).astype(np.uint8)


def render_optical_views(sen2_patch: np.ndarray) -> list[tuple[str, Image.Image]]:
    """sen2_patch: (32, 32, 10) float array. Returns 5 named RGB composite views."""
    views = []
    for name, idxs in S2_VIEWS:
        rgb = np.stack([_stretch(sen2_patch[:, :, i]) for i in idxs], axis=-1)
        views.append((name, Image.fromarray(rgb).resize((128, 128), Image.LANCZOS)))
    return views


def render_sar_view(sen1_patch: np.ndarray) -> tuple[str, Image.Image]:
    """sen1_patch: (32, 32, 8) float array. Bands: VH_re,VH_im,VV_re,VV_im,VH_int,VV_int,cov_re,cov_im.
    Renders the standard SAR RGB convention: R=VH intensity, G=VV intensity, B=VH/VV ratio."""
    vh = 10 * np.log10(np.abs(sen1_patch[:, :, 4]) + 1e-6)
    vv = 10 * np.log10(np.abs(sen1_patch[:, :, 5]) + 1e-6)
    ratio = vh - vv  # log-domain ratio
    rgb = np.stack([_stretch(vh), _stretch(vv), _stretch(ratio)], axis=-1)
    img = Image.fromarray(rgb).resize((128, 128), Image.LANCZOS)
    return ("SAR intensity composite (VH-VV-VH/VV, dB)", img)


def render_all_views(sen1_patch: np.ndarray, sen2_patch: np.ndarray):
    return render_optical_views(sen2_patch) + [render_sar_view(sen1_patch)]
