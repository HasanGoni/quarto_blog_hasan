"""Stage 1: real SAM2, point-prompted on real LEVIR-CD bi-temporal pairs, turned into change
pseudo-labels via the bi-temporal selection rule -- zero training involved in this stage, so the
raw pseudo-label quality (measured against real ground truth) is a clean, model-free number.
"""
import json
import os
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from transformers import Sam2Model, Sam2Processor

from common import IMG_SIZE, N_POINTS, sample_points, bi_temporal_pseudo_label, iou

device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "data/pseudolabels"
os.makedirs(OUT, exist_ok=True)

N_TRAIN = int(os.environ.get("N_TRAIN", 400))
N_VAL = int(os.environ.get("N_VAL", 100))

processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")
model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").to(device).eval()


@torch.no_grad()
def sam2_segment(image: Image.Image, point) -> tuple[np.ndarray, float]:
    inputs = processor(images=image, input_points=[[[list(point)]]], input_labels=[[[1]]],
                        return_tensors="pt").to(device)
    out = model(**inputs)
    best = out.iou_scores[0, 0].argmax().item()
    mask = processor.post_process_masks(out.pred_masks, inputs["original_sizes"])[0][0, best].cpu().numpy()
    return mask > 0.0, float(out.iou_scores[0, 0, best].item())


def process_split(ds, indices, split_name, rng):
    results = []
    for n, i in enumerate(indices):
        row = ds[int(i)]
        img_a = row["imageA"].convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_b = row["imageB"].convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        gt = np.array(row["label"].resize((IMG_SIZE, IMG_SIZE))) > 127

        points = sample_points(np.array(row["label"].resize((IMG_SIZE, IMG_SIZE))), N_POINTS, rng)
        pseudo = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
        conf_map = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        for pt in points:
            mask_a, _ = sam2_segment(img_a, pt)
            mask_b, _ = sam2_segment(img_b, pt)
            region, conf = bi_temporal_pseudo_label(mask_a, mask_b)
            pseudo |= region
            conf_map = np.maximum(conf_map, region.astype(np.float32) * conf)

        pl_iou = iou(pseudo, gt)
        results.append({"idx": int(i), "n_points": len(points), "pseudo_iou_vs_gt": pl_iou})

        np.savez(f"{OUT}/{split_name}_{n:04d}.npz", pseudo=pseudo, conf=conf_map, gt=gt,
                 img_a=np.array(img_a), img_b=np.array(img_b))
        if n % 50 == 0:
            print(f"{split_name} {n}/{len(indices)}  mean IoU so far: "
                  f"{np.mean([r['pseudo_iou_vs_gt'] for r in results]):.3f}")
    return results


if __name__ == "__main__":
    ds = load_dataset("ericyu/LEVIRCD_Cropped256", split="train")
    val_ds = load_dataset("ericyu/LEVIRCD_Cropped256", split="val")
    rng = np.random.default_rng(0)

    train_indices = rng.choice(len(ds), size=min(N_TRAIN, len(ds)), replace=False)
    val_indices = rng.choice(len(val_ds), size=min(N_VAL, len(val_ds)), replace=False)

    print(f"Pseudo-labeling {len(train_indices)} train + {len(val_indices)} val pairs...")
    train_results = process_split(ds, train_indices, "train", rng)
    val_results = process_split(val_ds, val_indices, "val", rng)

    with open(f"{OUT}/summary.json", "w") as f:
        json.dump({"train": train_results, "val": val_results}, f, indent=2)

    train_ious = [r["pseudo_iou_vs_gt"] for r in train_results]
    val_ious = [r["pseudo_iou_vs_gt"] for r in val_results]
    print(f"Raw pseudo-label IoU vs real GT -- train: {np.mean(train_ious):.3f}  val: {np.mean(val_ious):.3f}")
