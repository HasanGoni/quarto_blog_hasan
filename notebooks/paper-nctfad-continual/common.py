"""Shared utilities: real MVTec-AD loading, a real simplex ETF prototype construction (the
"neural collapse" target geometry), real CutPaste-style synthetic anomaly generation, and the
focal neural-collapse contrastive loss.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMG_SIZE = 224
K_SHOT = 8
PROJ_DIM = 64
N_PROTOTYPES = 20  # generous fixed allocation -- task-free means the model doesn't know in
                    # advance how many categories it will ever see


def build_simplex_etf(n_prototypes: int, dim: int, seed: int = 0) -> torch.Tensor:
    """Real simplex Equiangular Tight Frame construction: n_prototypes unit vectors in `dim`
    dimensions, pairwise cosine similarity = -1/(n_prototypes-1) for every pair -- the exact
    geometry neural-collapse theory predicts a well-trained classifier's last-layer class means
    converge to. Fixed once, never updated by gradient descent (unlike a naive running-mean
    "prototype")."""
    g = torch.Generator().manual_seed(seed)
    rand = torch.randn(dim, n_prototypes, generator=g)
    u, _ = torch.linalg.qr(rand)  # (dim, n_prototypes) orthonormal columns (dim >= n_prototypes)
    identity = torch.eye(n_prototypes)
    ones = torch.ones(n_prototypes, n_prototypes) / n_prototypes
    m = (n_prototypes / (n_prototypes - 1)) ** 0.5 * u @ (identity - ones)
    return F.normalize(m.T, dim=-1)  # (n_prototypes, dim), each row unit norm


def cutpaste(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Real CutPaste augmentation (Li et al., 2021): crop a random rectangular patch from the
    image and paste it at a different random location, optionally jittered -- a real, standard,
    label-free way to synthesize a local structural anomaly from normal images alone."""
    w, h = img.size
    pw, ph = rng.integers(w // 10, w // 4), rng.integers(h // 10, h // 4)
    x0, y0 = rng.integers(0, w - pw), rng.integers(0, h - ph)
    patch = img.crop((x0, y0, x0 + pw, y0 + ph))
    x1, y1 = rng.integers(0, w - pw), rng.integers(0, h - ph)
    out = img.copy()
    out.paste(patch, (x1, y1))
    return out


def focal_nc_loss(embeddings: torch.Tensor, target_proto: torch.Tensor,
                   negative_protos: torch.Tensor, focal_gamma: float = 2.0) -> torch.Tensor:
    """Focal Neural Collapse Contrastive loss: pull `embeddings` toward `target_proto` (cosine
    alignment), push away from every prototype in `negative_protos`, with a focal-loss-style
    (1 - p)^gamma weight down-weighting examples already well-aligned with their target --
    concentrating gradient on the harder, not-yet-aligned examples."""
    embeddings = F.normalize(embeddings, dim=-1)
    pos_sim = embeddings @ target_proto  # (B,)
    pos_prob = (pos_sim + 1) / 2  # rescale cosine [-1,1] -> [0,1] as a pseudo-probability
    focal_weight = (1 - pos_prob).clamp(min=0).pow(focal_gamma)
    pull_loss = (focal_weight * (1 - pos_sim)).mean()

    if negative_protos.shape[0] > 0:
        neg_sim = embeddings @ negative_protos.T  # (B, n_neg)
        push_loss = F.relu(neg_sim + 0.1).mean()  # margin: push below -0.1 similarity
    else:
        push_loss = torch.tensor(0.0, device=embeddings.device)
    return pull_loss + push_loss


def load_mvtec():
    from datasets import load_dataset
    return load_dataset("katiehahm/mvtec_ad", split="test")


def resize(img):
    return img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
