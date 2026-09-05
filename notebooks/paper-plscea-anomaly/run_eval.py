"""Real evaluation: does PL-SCEA's attention reconfiguration actually improve few-shot anomaly
detection over raw frozen DINOv2 features, on real MVTec-AD?

Pipeline, all real:
1. Frozen DINOv2-small extracts real patch tokens for every real MVTec-AD test image.
2. PL-SCEA reconfigures those tokens (common.sceareconfigure) -- zero new trainable params.
3. A tiny VAE is fit per category on k=8 real "good" images' (reconfigured) patch tokens.
4. Anomaly score = per-patch VAE reconstruction error, aggregated to an image-level score.
5. Real AUROC, WITH vs. WITHOUT the SCEA reconfiguration, same VAE architecture and k-shot
   support set both times -- a clean, real ablation.
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from common import load_mvtec, resize, sceareconfigure, IMG_SIZE, K_SHOT, GAMMA

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
IMG_OUT = "../../posts/series/papers/images"
OUT = "out"
os.makedirs(OUT, exist_ok=True)

DINO_ID = "facebook/dinov2-small"
processor = AutoImageProcessor.from_pretrained(DINO_ID)
dino = AutoModel.from_pretrained(DINO_ID).to(device).eval()
DINO_DIM = dino.config.hidden_size


@torch.no_grad()
def extract_patch_tokens(img) -> torch.Tensor:
    """Real image -> (N, D) L2-normalized patch tokens (CLS dropped), frozen DINOv2."""
    inputs = processor(images=resize(img), return_tensors="pt").to(device)
    out = dino(**inputs).last_hidden_state[0, 1:]  # drop CLS
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


def fit_vae(support_features: torch.Tensor, steps: int = 400) -> TinyVAE:
    """support_features: (K*N, D) real patch tokens from k normal shots."""
    vae = TinyVAE(support_features.shape[1]).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for _ in range(steps):
        recon, kl = vae(support_features)
        loss = F.mse_loss(recon, support_features) + 1e-3 * kl.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    vae.eval()
    return vae


@torch.no_grad()
def anomaly_score(vae: TinyVAE, features: torch.Tensor) -> float:
    recon, _ = vae(features)
    per_patch = (recon - features).pow(2).mean(-1)
    return per_patch.topk(max(1, len(per_patch) // 10)).values.mean().item()  # top-10% patches


if __name__ == "__main__":
    ds = load_mvtec()
    classes = sorted(set(ds["class"]))
    print(f"{len(ds)} real MVTec-AD test images, {len(classes)} categories")

    results = {}
    rng = np.random.default_rng(0)
    for cls in classes:
        idxs = [i for i, c in enumerate(ds["class"]) if c == cls]
        good_idxs = [i for i in idxs if ds[i]["gt_label"] == 0]
        if len(good_idxs) <= K_SHOT:
            print(f"skip {cls}: only {len(good_idxs)} good images")
            continue
        rng.shuffle(good_idxs)
        support_idxs = good_idxs[:K_SHOT]
        eval_idxs = [i for i in idxs if i not in support_idxs]

        support_raw, support_scea = [], []
        for i in support_idxs:
            feats = extract_patch_tokens(ds[i]["image"])
            support_raw.append(feats)
            support_scea.append(sceareconfigure(feats, GAMMA))
        support_raw = torch.cat(support_raw, dim=0)
        support_scea = torch.cat(support_scea, dim=0)

        vae_raw = fit_vae(support_raw)
        vae_scea = fit_vae(support_scea)

        scores_raw, scores_scea, labels = [], [], []
        for i in eval_idxs:
            feats = extract_patch_tokens(ds[i]["image"])
            scores_raw.append(anomaly_score(vae_raw, feats))
            scores_scea.append(anomaly_score(vae_scea, sceareconfigure(feats, GAMMA)))
            labels.append(int(ds[i]["gt_label"]))

        if len(set(labels)) < 2:
            print(f"skip {cls}: only one label present in eval set")
            continue
        auroc_raw = roc_auc_score(labels, scores_raw)
        auroc_scea = roc_auc_score(labels, scores_scea)
        results[cls] = {"auroc_raw": auroc_raw, "auroc_scea": auroc_scea, "n_eval": len(eval_idxs)}
        print(f"{cls:12s} n={len(eval_idxs):3d}  raw AUROC {auroc_raw:.3f}  SCEA AUROC {auroc_scea:.3f}")

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    mean_raw = np.mean([r["auroc_raw"] for r in results.values()])
    mean_scea = np.mean([r["auroc_scea"] for r in results.values()])
    print(f"\nMean AUROC across {len(results)} categories -- raw: {mean_raw:.3f}  SCEA: {mean_scea:.3f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    cats = list(results.keys())
    x = np.arange(len(cats))
    ax.bar(x - 0.2, [results[c]["auroc_raw"] for c in cats], width=0.4, label=f"Raw DINOv2 (mean {mean_raw:.3f})")
    ax.bar(x + 0.2, [results[c]["auroc_scea"] for c in cats], width=0.4, label=f"PL-SCEA (mean {mean_scea:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylabel(f"AUROC ({K_SHOT}-shot, real MVTec-AD)"); ax.legend()
    ax.set_title("Real few-shot anomaly detection: raw features vs. PL-SCEA reconfiguration")
    plt.tight_layout()
    plt.savefig(f"{IMG_OUT}/plscea-auroc-comparison.png", dpi=130)
    print("Saved plscea-auroc-comparison.png")
