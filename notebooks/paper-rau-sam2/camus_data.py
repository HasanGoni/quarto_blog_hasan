"""Loading for the real CAMUS-Lite data. Each file is a NIfTI half-cardiac-
cycle sequence (630x519x~22 frames per patient/view), not a flat PNG as the
dataset card's prose suggests — this module handles the real format.

Labels: 1 = left ventricular myocardium, 2 = left ventricle, 3 = left atrium.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

LABELS = [1, 2, 3]


def download_camus(cache_dir: Path) -> tuple[Path, Path]:
    images_zip = hf_hub_download("YongchengYAO/CAMUS-Lite", "Images.zip", repo_type="dataset")
    masks_zip = hf_hub_download("YongchengYAO/CAMUS-Lite", "Masks.zip", repo_type="dataset")
    img_dir = cache_dir / "Images"
    mask_dir = cache_dir / "Masks"
    if not img_dir.exists():
        with zipfile.ZipFile(images_zip) as z:
            z.extractall(cache_dir)
    if not mask_dir.exists():
        with zipfile.ZipFile(masks_zip) as z:
            z.extractall(cache_dir)
    return img_dir, mask_dir


def list_patients(img_dir: Path, mask_dir: Path) -> list[tuple[Path, Path, str]]:
    """Returns (image_path, mask_path, patient_view_id) for every patient/view
    that has both an image and a ground-truth volume."""
    out = []
    for img_path in sorted(img_dir.glob("*.nii.gz")):
        stem = img_path.name.replace(".nii.gz", "")
        mask_path = mask_dir / f"{stem}_gt.nii.gz"
        if mask_path.exists():
            out.append((img_path, mask_path, stem))
    return out


def load_frame_pair(image_path: Path, mask_path: Path, ref_frame: int = 0, target_frame: int = -1):
    """Loads two frames (by default first and last) from the same patient's
    half-sequence: a reference frame with its mask, and a target frame with
    its own (held-out at inference, used only for supervision at train time)
    mask."""
    img_vol = nib.load(str(image_path)).get_fdata()
    mask_vol = nib.load(str(mask_path)).get_fdata()
    n_frames = img_vol.shape[-1]
    ref_idx = ref_frame % n_frames
    tgt_idx = target_frame % n_frames

    ref_img = _to_pil(img_vol[..., ref_idx])
    ref_mask = mask_vol[..., ref_idx].astype(np.uint8).T  # match _to_pil's transpose
    tgt_img = _to_pil(img_vol[..., tgt_idx])
    tgt_mask = mask_vol[..., tgt_idx].astype(np.uint8).T
    return ref_img, ref_mask, tgt_img, tgt_mask


def _to_pil(frame_2d: np.ndarray) -> Image.Image:
    arr = frame_2d.astype(np.float32)
    if arr.max() > 0:
        arr = arr / arr.max() * 255.0
    arr = arr.astype(np.uint8).T  # nibabel returns (W, H); transpose to (H, W)
    return Image.fromarray(arr).convert("RGB")
