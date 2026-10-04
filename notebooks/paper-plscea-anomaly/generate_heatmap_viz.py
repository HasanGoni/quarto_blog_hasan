"""Real visualization: per-patch anomaly maps from the corrected pipeline, original QK attention vs.
PL-SCEA (gamma=2), on real defective MVTec-AD images of one category.

Uses the exact same support-set construction as run_eval.py (official protocol / setup B, seed 0, 8-shot,
same permutation of the category's train/good images), and the same LNM-VAE head, so the maps correspond to the seed-0 numbers.

usage: python generate_heatmap_viz.py [CLASS] [N_FRAMES]

Produces:
  - plscea-heatmap-{TARGET_CLASS}-video.mp4: cycles through real defective images, QK vs PL-SCEA side by side.
  - plscea-heatmap-{TARGET_CLASS}-before.jpg / plscea-heatmap-{TARGET_CLASS}-after.jpg: one clean image pair for an interactive
    before/after slider (QK map vs PL-SCEA map on the SAME defective image, shared color scale).
"""
import sys

import imageio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import data
from common import Radio, fit_lnm, patch_score_map, IMG_SIZE, GAMMA

K_SHOT = 8
SEED = 0
IMG_OUT = "../../posts/series/papers/images"
TARGET_CLASS = sys.argv[1] if len(sys.argv) > 1 else "capsule"
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 6
device = "cuda" if torch.cuda.is_available() else "cpu"


def overlay(img: Image.Image, score_map: np.ndarray, vmin: float, vmax: float) -> Image.Image:
    g = np.clip((score_map - vmin) / (vmax - vmin + 1e-8), 0, 1)
    heat = Image.fromarray((cm.inferno(g)[:, :, :3] * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    base = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return Image.blend(base, heat, alpha=0.5)


if __name__ == "__main__":
    d = data.official_class(TARGET_CLASS)
    radio = Radio(device)
    order = list(np.random.default_rng(SEED).permutation(d["pool"]))
    support = order[:K_SHOT]
    all_defects = [k for k in d["cands"] if d["label"][k] == 1]
    defects = [all_defects[int(j)] for j in np.linspace(0, len(all_defects) - 1, N_FRAMES)]  # spread over defect types

    heads = {}
    for name, mode in [("qk", "qk"), ("plscea", "plscea")]:
        sup = torch.cat([radio.patch_tokens(d["load"](i), mode, GAMMA) for i in support], dim=0)
        heads[name] = (mode, *fit_lnm(sup, seed=SEED))
        print(f"fitted {name} head on {len(support)} real '{TARGET_CLASS}' shots", flush=True)

    frames = []
    for j, i in enumerate(defects):
        img = d["load"](i)
        maps = {n: patch_score_map(m, mean, std, radio.patch_tokens(img, mode, GAMMA), "paper")
                for n, (mode, m, mean, std) in heads.items()}
        vmin = min(mp.min() for mp in maps.values()); vmax = max(mp.max() for mp in maps.values())
        ov = {n: overlay(img, mp, vmin, vmax) for n, mp in maps.items()}
        if j == 0:
            ov["qk"].save(f"{IMG_OUT}/plscea-heatmap-{TARGET_CLASS}-before.jpg", quality=88)
            ov["plscea"].save(f"{IMG_OUT}/plscea-heatmap-{TARGET_CLASS}-after.jpg", quality=90)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
        axes[0].imshow(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))); axes[0].set_title(f"real defective {TARGET_CLASS}", fontsize=9)
        axes[1].imshow(ov["qk"]); axes[1].set_title("original QK attention", fontsize=9)
        axes[2].imshow(ov["plscea"]); axes[2].set_title("PL-SCEA (gamma = 2)", fontsize=9)
        for a in axes: a.axis("off")
        plt.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
        print(f"  frame {j}: image idx {i} done", flush=True)

    imageio.mimsave(f"{IMG_OUT}/plscea-heatmap-{TARGET_CLASS}-video.mp4", frames, fps=1, macro_block_size=16)
    print(f"Saved heatmap video + before/after for {TARGET_CLASS}")
