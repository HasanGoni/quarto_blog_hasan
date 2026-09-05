"""Real task-free continual anomaly detection: stream real MVTec-AD categories one at a time,
no announced task boundaries, and measure real forgetting -- NC-TFAD's fixed simplex-ETF
prototypes vs. a naive baseline whose "prototypes" are just each category's own running-mean
embedding (which drifts as the shared head keeps training on later categories).
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

from common import (load_mvtec, resize, build_simplex_etf, cutpaste, focal_nc_loss,
                     IMG_SIZE, K_SHOT, PROJ_DIM, N_PROTOTYPES)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
IMG_OUT = "../../posts/series/papers/images"
OUT = "out"
os.makedirs(OUT, exist_ok=True)

DINO_ID = "facebook/dinov2-small"
processor = AutoImageProcessor.from_pretrained(DINO_ID)
dino = AutoModel.from_pretrained(DINO_ID).to(device).eval()
DINO_DIM = dino.config.hidden_size
STEPS_PER_CATEGORY = 150


@torch.no_grad()
def backbone_embed(img) -> torch.Tensor:
    inputs = processor(images=resize(img), return_tensors="pt").to(device)
    return dino(**inputs).last_hidden_state[0, 0]  # CLS token, (D,)


class ProjHead(nn.Module):
    def __init__(self, dim_in=DINO_DIM, dim_out=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_in, 128), nn.ReLU(), nn.Linear(128, dim_out))

    def forward(self, x):
        return self.net(x)


def run_stream(ds, classes, method: str, rng: np.random.Generator):
    """method: 'nctfad' (fixed ETF prototypes) or 'baseline' (drifting running-mean prototypes)."""
    head = ProjHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    etf = build_simplex_etf(N_PROTOTYPES, PROJ_DIM).to(device)

    prototypes = {}  # category -> (PROJ_DIM,) tensor
    support_embeds_cache = {}  # cache raw backbone embeddings so re-eval doesn't recompute
    curve = []  # (n_categories_seen, mean_auroc_so_far)

    for step_cat, cls in enumerate(classes):
        idxs = [i for i, c in enumerate(ds["class"]) if c == cls]
        good_idxs = [i for i in idxs if ds[i]["gt_label"] == 0]
        if len(good_idxs) <= K_SHOT:
            continue
        rng.shuffle(good_idxs)
        support_idxs = good_idxs[:K_SHOT]

        support_raw = [backbone_embed(ds[i]["image"]) for i in support_idxs]
        support_raw = torch.stack(support_raw)
        neg_raw = [backbone_embed(cutpaste(resize(ds[i]["image"]), rng)) for i in support_idxs]
        neg_raw = torch.stack(neg_raw)
        support_embeds_cache[cls] = (support_idxs, support_raw)

        if method == "nctfad":
            target = etf[step_cat]
        else:
            with torch.no_grad():
                target = F.normalize(head(support_raw).mean(0), dim=0)

        prior_protos = torch.stack([prototypes[c] for c in prototypes]) if prototypes else torch.zeros(0, PROJ_DIM, device=device)

        for _ in range(STEPS_PER_CATEGORY):
            pos_emb = head(support_raw)
            neg_emb = head(neg_raw)
            all_emb = torch.cat([pos_emb, neg_emb], dim=0)
            loss = focal_nc_loss(pos_emb, target.detach(), prior_protos.detach())
            loss = loss + F.relu(F.normalize(neg_emb, dim=-1) @ target.detach() + 0.1).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            prototypes[cls] = F.normalize(head(support_raw).mean(0), dim=0) if method == "baseline" else target

        # --- evaluate on every category seen so far (real forgetting curve) ---
        aurocs = []
        with torch.no_grad():
            for seen_cls in prototypes:
                seen_idxs = [i for i, c in enumerate(ds["class"]) if c == seen_cls]
                seen_support_idxs, _ = support_embeds_cache[seen_cls]
                eval_idxs = [i for i in seen_idxs if i not in seen_support_idxs]
                labels, scores = [], []
                for i in eval_idxs:
                    emb = F.normalize(head(backbone_embed(ds[i]["image"]).unsqueeze(0)), dim=-1)[0]
                    score = 1 - (emb @ prototypes[seen_cls]).item()
                    scores.append(score); labels.append(int(ds[i]["gt_label"]))
                if len(set(labels)) < 2:
                    continue
                aurocs.append(roc_auc_score(labels, scores))
        mean_auroc = float(np.mean(aurocs)) if aurocs else float("nan")
        curve.append((step_cat + 1, mean_auroc))
        print(f"[{method}] after '{cls}' ({step_cat+1}/{len(classes)} categories seen): "
              f"mean AUROC over all seen so far = {mean_auroc:.3f}")

    return curve


if __name__ == "__main__":
    ds = load_mvtec()
    classes = sorted(set(ds["class"]))
    print(f"{len(ds)} real MVTec-AD test images, streaming {len(classes)} categories, "
          f"no announced task boundaries")

    rng1 = np.random.default_rng(0)
    curve_nctfad = run_stream(ds, classes, "nctfad", rng1)
    rng2 = np.random.default_rng(0)
    curve_baseline = run_stream(ds, classes, "baseline", rng2)

    with open(f"{OUT}/results.json", "w") as f:
        json.dump({"nctfad": curve_nctfad, "baseline": curve_baseline}, f, indent=2)

    plt.figure(figsize=(7, 4.5))
    plt.plot([c[0] for c in curve_nctfad], [c[1] for c in curve_nctfad], marker="o", label="NC-TFAD (fixed ETF prototypes)")
    plt.plot([c[0] for c in curve_baseline], [c[1] for c in curve_baseline], marker="s", label="Baseline (drifting mean prototypes)")
    plt.xlabel("categories streamed so far (no task boundary announced)")
    plt.ylabel("mean AUROC over all categories seen so far")
    plt.title("Real task-free continual anomaly detection on MVTec-AD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{IMG_OUT}/nctfad-forgetting-curve.png", dpi=130)
    print("Saved nctfad-forgetting-curve.png")
