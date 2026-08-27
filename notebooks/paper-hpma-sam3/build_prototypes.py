"""Builds the real, frozen HPMA prototype memory bank from a subset of the
real EndoVis2017 dataset, using the real facebook/sam3 vision encoder.

For every (scale, category) pair: run the real SAM3 vision encoder on real
training images, masked-average-pool its FPN features over the real
ground-truth pixels for that category (Eq. 4-5 in the paper), collect one
vector per image, then K-Means (K=4) the collected vectors into frozen
prototypes. No labels beyond the dataset's own per-pixel category ids are
used — the HF mirror doesn't ship an authoritative id->name table, so
categories are referred to generically (class 1..7); the mechanism doesn't
need real names to work.

Run:
    uv run build_prototypes.py --n-images 200 --out prototypes.pt
"""
from __future__ import annotations

import argparse
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.cluster import KMeans
from transformers import Sam3Model, Sam3Processor

from hpma_modules import PrototypeBank

SCALES = ["local", "structural", "global"]  # FPN indices 0, 1, 2 respectively


def load_endovis_subset(n_images: int, split: str = "train") -> pd.DataFrame:
    parts = []
    n_loaded = 0
    part_idx = 0
    while n_loaded < n_images:
        try:
            path = hf_hub_download(
                "tyluan/Endovis2017", f"data/{split}-{part_idx:05d}-of-00004.parquet", repo_type="dataset"
            )
        except Exception:
            break
        df = pd.read_parquet(path)
        parts.append(df)
        n_loaded += len(df)
        part_idx += 1
        if part_idx > 4:
            break
    full = pd.concat(parts, ignore_index=True)
    return full.iloc[:n_images]


@torch.no_grad()
def extract_fpn(model: Sam3Model, processor: Sam3Processor, image: Image.Image, device) -> list[torch.Tensor]:
    inputs = processor(images=image, return_tensors="pt").to(device)
    vout = model.vision_encoder(inputs.pixel_values)
    return [vout.fpn_hidden_states[i][0] for i in range(3)]  # (d, H, W) each, drop batch dim


def masked_avg_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    """feat: (d, H, W). mask: (H, W) bool, already resized to feat's resolution."""
    if mask.sum() == 0:
        return None
    return feat[:, mask].mean(dim=1)  # (d,)


def build(n_images: int, out_path: Path, device: str = "cuda"):
    model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    df = load_endovis_subset(n_images)
    print(f"Loaded {len(df)} real EndoVis2017 images")

    collected: dict[str, dict[int, list[np.ndarray]]] = {s: defaultdict(list) for s in SCALES}

    for i, row in df.iterrows():
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        label = Image.open(io.BytesIO(row["label"]["bytes"]))
        label_arr = np.array(label)
        categories = [c for c in np.unique(label_arr) if c != 0]
        if not categories:
            continue

        fpn = extract_fpn(model, processor, image, device)

        for scale_idx, scale_name in enumerate(SCALES):
            feat = fpn[scale_idx]
            h, w = feat.shape[1], feat.shape[2]
            mask_resized = np.array(Image.fromarray(label_arr).resize((w, h), Image.NEAREST))
            for c in categories:
                m = torch.from_numpy(mask_resized == c).to(device)
                pooled = masked_avg_pool(feat, m)
                if pooled is not None:
                    collected[scale_name][int(c)].append(pooled.cpu().numpy())

        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(df)} images")

    bank = PrototypeBank()
    for scale_name in SCALES:
        bank.vectors[scale_name] = {}
        for c, vecs in collected[scale_name].items():
            arr = np.stack(vecs)
            k = min(4, len(arr))
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(arr)
            bank.vectors[scale_name][c] = torch.tensor(km.cluster_centers_, dtype=torch.float32)
            print(f"  {scale_name} class {c}: {len(arr)} samples -> {k} prototypes")

    bank.save(out_path)
    print(f"Saved prototype bank -> {out_path}")
    return bank


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("prototypes.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    build(args.n_images, args.out, args.device)
