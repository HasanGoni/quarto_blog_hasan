"""Loader for the synthetic void dataset that runs through the whole series.

The data is built by `notebooks/agentic-factory-segmentation/make_dataset.py` (see the post
"A Synthetic Void Dataset With Exact Masks, Built From Fashion-MNIST"). Protocol used in every
part of this series:

    val  (600 images) = DEV   -- every choice (a threshold, a prompt) is made here
    test (600 images) = HELD-OUT -- looked at once per question, to report the number

`train` (2,800 images) is not used until a part trains something.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = (Path(__file__).resolve().parents[1]
            / "agentic-factory-segmentation" / "data" / "void_dataset_v1")


@lru_cache(maxsize=None)
def _arrays(split: str):
    d = np.load(DATA_DIR / f"{split}.npz")
    return d["images"], d["void_masks"], d["part_masks"]  # load each array once


@lru_cache(maxsize=None)
def _voids_by_id():
    by_id: dict[str, list[dict]] = {}
    for line in (DATA_DIR / "voids.jsonl").read_text().splitlines():
        v = json.loads(line)
        by_id.setdefault(v["id"], []).append(v)
    return by_id


@lru_cache(maxsize=None)
def _metadata():
    import csv
    with open(DATA_DIR / "metadata.csv") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def split_ids(split: str) -> list[str]:
    n = len(_arrays(split)[0])
    return [f"{split}_{i:05d}" for i in range(n)]


def load_image(split: str, i: int) -> np.ndarray:
    """The X-ray-like image, uint8, 224x224. Part is dark, air is bright, voids are brighter."""
    return _arrays(split)[0][i]


def load_masks(split: str, i: int):
    """Exact (void_mask, part_mask), both uint8 0/1. Never shown to a detector or a model."""
    _, vm, pm = _arrays(split)
    return vm[i], pm[i]


def load_voids(split: str, i: int) -> list[dict]:
    """Ground-truth voids of one image: bbox [x1, y1, x2, y2], area_px, eq_diam_px, contrast."""
    return _voids_by_id().get(f"{split}_{i:05d}", [])


def load_meta(split: str, i: int) -> dict:
    return _metadata()[f"{split}_{i:05d}"]
