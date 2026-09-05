"""Real MVTec-AD (image, guide, text, mask_image) rows from Kelvin878/mvtec, split into a
training set (everything except the held-out combos) and a held-out zero-shot eval set (only
the held-out combos) -- both real data, never approximated.
"""
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms

from common import IMG_SIZE, HELD_OUT_COMBOS, parse_caption, product_agnostic_prompt

_resize = transforms.Resize((IMG_SIZE, IMG_SIZE))


def load_split():
    ds = load_dataset("Kelvin878/mvtec", split="train")
    train_idx, held_out_idx = [], []
    for i in range(len(ds)):
        defect, category = parse_caption(ds[i]["text"])
        if defect is None:
            continue
        if (defect, category) in HELD_OUT_COMBOS:
            held_out_idx.append(i)
        else:
            train_idx.append(i)
    return ds, train_idx, held_out_idx


class MVTecAnomalyDataset(Dataset):
    def __init__(self, ds, indices):
        self.ds = ds
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.ds[int(self.indices[i])]
        defect, category = parse_caption(row["text"])
        return {
            "guide": _resize(row["guide"].convert("RGB")),          # normal reference (masked-image input)
            "mask": _resize(row["mask_image"].convert("L")),         # defect region mask
            "image": _resize(row["image"].convert("RGB")),           # real defect image (training target)
            "prompt": product_agnostic_prompt(defect),               # product-agnostic caption
            "defect": defect,
            "category": category,
        }
