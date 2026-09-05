"""Real Kvasir-SEG polyp segmentation data (kowndinya23/Kvasir-SEG on the HF Hub -- 880 train /
120 validation image+mask pairs, CC BY 4.0), turned into (original, edited-target, instruction)
triples for instruction-driven diffusion fine-tuning.
"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset

from common import IMG_SIZE, INSTRUCTION, render_overlay

_resize = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
])


class KvasirEditDataset(Dataset):
    def __init__(self, split: str, tokenizer):
        self.ds = load_dataset("kowndinya23/Kvasir-SEG", split=split)
        self.tokenizer = tokenizer
        self.token_ids = tokenizer(
            INSTRUCTION, padding="max_length", truncation=True,
            max_length=tokenizer.model_max_length, return_tensors="pt",
        ).input_ids[0]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        image = _resize(row["image"].convert("RGB"))
        mask = row["annotation"]
        target = render_overlay(image, mask)

        to_tensor = transforms.ToTensor()
        original = to_tensor(image) * 2 - 1     # [-1, 1] for VAE
        target_t = to_tensor(target) * 2 - 1
        original_0_1 = to_tensor(image)          # [0, 1] for DINO normalization

        return {
            "original": original,
            "original_0_1": original_0_1,
            "target": target_t,
            "input_ids": self.token_ids,
        }
