"""Two data protocols for the same pipeline.

holdout  (setup A): katiehahm/mvtec_ad mirror ships only the TEST split, so k 'good' test images per
                    category are held out as the few-shot support set; the rest is evaluated.
official (setup B): the official MVTec-AD layout (foersben/mvtec-ad mirror): support set drawn from
                    train/good, evaluation on the full official test set (good + defective).
Each protocol returns, per category: pool (keys to draw support from), cands (keys to evaluate on,
support excluded), label[key], and load(key) -> PIL image.
"""
import os
from PIL import Image

OFFICIAL_ROOT = "data/mvtec_ad"


def load_mvtec_ds():
    from datasets import load_dataset
    return load_dataset("katiehahm/mvtec_ad", split="test")


def holdout_classes(ds):
    return sorted(set(ds["class"]))


def holdout_class(ds, cls):
    idxs = [i for i, c in enumerate(ds["class"]) if c == cls]
    label = {i: int(ds[i]["gt_label"]) for i in idxs}
    return dict(pool=[i for i in idxs if label[i] == 0], cands=idxs, label=label,
                load=lambda k: ds[k]["image"])


def official_classes(root=OFFICIAL_ROOT):
    return sorted(d for d in os.listdir(root) if os.path.isdir(f"{root}/{d}/train/good"))


def official_class(cls, root=OFFICIAL_ROOT):
    pool = sorted(f"{root}/{cls}/train/good/{f}" for f in os.listdir(f"{root}/{cls}/train/good"))
    cands, label = [], {}
    for defect in sorted(os.listdir(f"{root}/{cls}/test")):
        for f in sorted(os.listdir(f"{root}/{cls}/test/{defect}")):
            p = f"{root}/{cls}/test/{defect}/{f}"
            cands.append(p); label[p] = 0 if defect == "good" else 1
    for p in pool: label[p] = 0
    return dict(pool=pool, cands=cands, label=label, load=lambda k: Image.open(k).convert("RGB"))
