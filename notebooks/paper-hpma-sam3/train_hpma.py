"""Real, reduced training loop for the HPMA adapters against real SAM3 +
real EndoVis2017 data. SAM3 stays 100% frozen; only the two coupling
adapters (global cross-attention, structural query bias) are trained.

Losses implemented for real: pixel BCE (L_mask) + Dice (L_dice) on the
best-matching query's predicted mask, plus the local alignment loss (Eq. 7).
Not implemented: the classification/box/giou/presence terms and full
DETR Hungarian matching across all 200 queries per image — those are
orthogonal to what HPMA itself contributes (the adapters + alignment loss),
and full bipartite matching is significant extra machinery for a demo. We
pick the single highest-confidence query per (image, category) instead of
matching against a full ground-truth instance set.

Run:
    uv run train_hpma.py --prototypes prototypes.pt --steps 60
"""
from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image

from hpma_modules import PrototypeBank, dice_loss, local_alignment_loss
from hpma_sam3 import HPMASam3

# Standard EndoVis2017 instrument-type numbering (widely cited convention for
# this challenge's label maps; the HF mirror's dataset card doesn't ship an
# explicit id->name table itself, so this is the commonly-used convention,
# not a value confirmed directly from tyluan/Endovis2017's own metadata).
CATEGORY_NAMES = {
    1: "bipolar forceps",
    2: "prograsp forceps",
    3: "large needle driver",
    4: "vessel sealer",
    5: "grasping retractor",
    6: "monopolar curved scissors",
    7: "ultrasound probe",
}


def load_split(split: str, n: int) -> pd.DataFrame:
    path = hf_hub_download("tyluan/Endovis2017", f"data/{split}-00000-of-00002.parquet"
                            if split == "val" else f"data/{split}-00000-of-00004.parquet",
                            repo_type="dataset")
    df = pd.read_parquet(path)
    return df.iloc[:n]


def decode_row(row):
    image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    label = np.array(Image.open(io.BytesIO(row["label"]["bytes"])))
    return image, label


def train(prototypes_path: Path, steps: int, out_dir: Path, device: str):
    bank = PrototypeBank.load(prototypes_path).to(device)
    categories = sorted(bank.vectors["global"].keys())

    model = HPMASam3().to(device)
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=3e-4)

    df = load_split("val", 60)  # distinct from the images used to build prototypes
    examples = []
    for _, row in df.iterrows():
        image, label = decode_row(row)
        present = [c for c in np.unique(label) if c in categories]
        for c in present:
            examples.append((image, label, int(c)))
    random.Random(0).shuffle(examples)
    print(f"{len(examples)} (image, category) training examples from real val split")

    fixed_image, fixed_label, fixed_cat = next((e for e in examples if e[2] == categories[0]), examples[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    gif_frames = []
    losses = []

    for step in range(steps):
        image, label, cat = examples[step % len(examples)]
        model.set_active_category(bank, cat)

        inputs = model.processor(images=image, text=CATEGORY_NAMES.get(cat, f"class {cat}"), return_tensors="pt").to(device)
        vout = model.sam3.vision_encoder(inputs.pixel_values)
        f0 = vout.fpn_hidden_states[0]  # (1, d, 288, 288), local scale

        outputs = model.sam3(
            vision_embeds=vout, input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
        )

        gt_288 = torch.from_numpy(
            np.array(Image.fromarray((label == cat).astype(np.uint8) * 255).resize((288, 288), Image.NEAREST))
        ).to(device).float() / 255.0

        best_q = outputs.pred_logits[0].argmax().item()
        pred_mask_logits = outputs.pred_masks[0, best_q]  # (288, 288)

        bce = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_288)
        dice = dice_loss(pred_mask_logits.unsqueeze(0), gt_288.unsqueeze(0))
        align = local_alignment_loss(f0, (gt_288 > 0.5).unsqueeze(0), bank.vectors["local"][cat])
        loss = bce + dice + 0.01 * align

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 5 == 0 or step == steps - 1:
            print(f"step {step:3d}/{steps}  loss {loss.item():.4f}  (bce {bce.item():.4f} dice {dice.item():.4f} align {align.item():.4f})  cat {cat}")

        if step % max(1, steps // 24) == 0 or step == steps - 1:
            gif_frames.append(_render_frame(model, bank, fixed_image, fixed_label, fixed_cat, device, step))

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("training step")
    plt.ylabel("loss (bce + dice + 0.01*align)")
    plt.title("HPMA adapter training loss (real SAM3, real EndoVis2017 subset)")
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=130)

    gif_frames[0].save(
        out_dir / "training_progress.gif", save_all=True,
        append_images=gif_frames[1:] + [gif_frames[-1]] * 4, duration=400, loop=0,
    )
    import json
    (out_dir / "losses.json").write_text(json.dumps(losses))
    print(f"Saved loss curve + training_progress.gif -> {out_dir}")


@torch.no_grad()
def _render_frame(model, bank, image, label, cat, device, step) -> Image.Image:
    model.set_active_category(bank, cat)
    inputs = model.processor(images=image, text=CATEGORY_NAMES.get(cat, f"class {cat}"), return_tensors="pt").to(device)
    outputs = model.sam3(pixel_values=inputs.pixel_values, input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    best_q = outputs.pred_logits[0].argmax().item()
    pred = (outputs.pred_masks[0, best_q].sigmoid() > 0.5).cpu().numpy()

    base = np.array(image.resize((288, 288))).astype(np.float32)
    overlay = base.copy()
    overlay[pred] = overlay[pred] * 0.4 + np.array([255, 60, 60]) * 0.6
    frame = Image.fromarray(overlay.astype(np.uint8))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(frame)
    ax.set_title(f"step {step}", fontsize=10)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prototypes", type=Path, default=Path("prototypes.pt"))
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out", type=Path, default=Path("train_out"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    train(args.prototypes, args.steps, args.out, args.device)
