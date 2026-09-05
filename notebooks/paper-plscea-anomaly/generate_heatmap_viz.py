"""Real visualization: per-patch VAE reconstruction-error heatmaps, raw DINOv2 features vs.
PL-SCEA-reconfigured features, on real defective 'bottle' images -- the category where SCEA
hurt AUROC the most dramatically (0.947 raw -> 0.675 SCEA). Reuses the exact same support-set
construction (same class-iteration order + seed) as run_eval.py so the two scripts' 'bottle'
support set and fitted VAEs match.

Produces:
  - plscea-heatmap-video.mp4: cycles through several real defective bottle images, raw vs SCEA
    heatmap side by side.
  - plscea-heatmap-before.jpg / plscea-heatmap-after.jpg: one clean image pair for an interactive
    before/after slider in the post (raw heatmap vs SCEA heatmap on the SAME defective image).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio
from PIL import Image

from common import load_mvtec, resize, sceareconfigure, IMG_SIZE, K_SHOT, GAMMA

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
IMG_OUT = "../../posts/series/papers/images"
TARGET_CLASS = "bottle"
GRID = 16  # DINOv2-small, patch size 14, 224/14 = 16 -> 256 patches

DINO_ID = "facebook/dinov2-small"
processor = AutoImageProcessor.from_pretrained(DINO_ID)
dino = AutoModel.from_pretrained(DINO_ID).to(device).eval()


@torch.no_grad()
def extract_patch_tokens(img):
    inputs = processor(images=resize(img), return_tensors="pt").to(device)
    out = dino(**inputs).last_hidden_state[0, 1:]
    return F.normalize(out, dim=-1)


class TinyVAE(nn.Module):
    def __init__(self, dim, latent=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, latent * 2))
        self.dec = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, dim))
        self.latent = latent

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = h[:, :self.latent], h[:, self.latent:]
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        recon = self.dec(z)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)
        return recon, kl


def fit_vae(support_features, steps=400):
    vae = TinyVAE(support_features.shape[1]).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for _ in range(steps):
        recon, kl = vae(support_features)
        loss = F.mse_loss(recon, support_features) + 1e-3 * kl.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    vae.eval()
    return vae


@torch.no_grad()
def per_patch_error_map(vae, features):
    recon, _ = vae(features)
    per_patch = (recon - features).pow(2).mean(-1).cpu().numpy()  # (256,)
    grid = per_patch.reshape(GRID, GRID)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    return grid


def overlay_heatmap(img: Image.Image, grid: np.ndarray) -> Image.Image:
    heat = Image.fromarray((cm.inferno(grid)[:, :, :3] * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    base = resize(img)
    blended = Image.blend(base, heat, alpha=0.5)
    return blended


if __name__ == "__main__":
    ds = load_mvtec()
    classes = sorted(set(ds["class"]))
    rng = np.random.default_rng(0)

    for cls in classes:  # replicate run_eval.py's exact iteration to reproduce the same rng state
        idxs = [i for i, c in enumerate(ds["class"]) if c == cls]
        good_idxs = [i for i in idxs if ds[i]["gt_label"] == 0]
        if len(good_idxs) <= K_SHOT:
            continue
        rng.shuffle(good_idxs)
        if cls == TARGET_CLASS:
            support_idxs = good_idxs[:K_SHOT]
            defect_idxs = [i for i in idxs if ds[i]["gt_label"] == 1][:6]
            break

    print(f"Refitting VAEs on '{TARGET_CLASS}' support set ({len(support_idxs)} shots) ...")
    support_raw, support_scea = [], []
    for i in support_idxs:
        feats = extract_patch_tokens(ds[i]["image"])
        support_raw.append(feats)
        support_scea.append(sceareconfigure(feats, GAMMA))
    vae_raw = fit_vae(torch.cat(support_raw, dim=0))
    vae_scea = fit_vae(torch.cat(support_scea, dim=0))

    frames = []
    for j, i in enumerate(defect_idxs):
        img = ds[i]["image"]
        feats = extract_patch_tokens(img)
        raw_grid = per_patch_error_map(vae_raw, feats)
        scea_grid = per_patch_error_map(vae_scea, sceareconfigure(feats, GAMMA))

        raw_overlay = overlay_heatmap(img, raw_grid)
        scea_overlay = overlay_heatmap(img, scea_grid)

        if j == 0:
            raw_overlay.save(f"{IMG_OUT}/plscea-heatmap-before.jpg", quality=90)
            scea_overlay.save(f"{IMG_OUT}/plscea-heatmap-after.jpg", quality=90)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
        axes[0].imshow(resize(img)); axes[0].set_title("real defective bottle", fontsize=9); axes[0].axis("off")
        axes[1].imshow(raw_overlay); axes[1].set_title("raw DINOv2 error map", fontsize=9); axes[1].axis("off")
        axes[2].imshow(scea_overlay); axes[2].set_title("PL-SCEA error map", fontsize=9); axes[2].axis("off")
        plt.tight_layout()
        fig.canvas.draw()
        frame_img = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                                      fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")
        frames.append(np.array(frame_img))
        plt.close(fig)
        print(f"  frame {j}: image idx {i} done")

    imageio.mimsave(f"{IMG_OUT}/plscea-heatmap-video.mp4", frames, fps=1, macro_block_size=16)
    print("Saved plscea-heatmap-video.mp4, plscea-heatmap-before.jpg, plscea-heatmap-after.jpg")
