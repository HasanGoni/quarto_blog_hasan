"""Shared utilities: real MVTec-AD loading (via katiehahm/mvtec_ad, image+gt_label+class,
CC-BY-NC-4.0), the frozen C-RADIOv3-B encoder with the real PL-SCEA attention swapped into its
final block (arXiv:2609.03655, Eq. 1-10), and the paper's Latent Normality Modeling VAE head
(Eq. 11-13).

Revision note (2026-10-03): the first version of this reimplementation was wrong in several ways,
as pointed out in a GitHub issue by the paper's author -- it used DINOv2-small instead of
RADIOv3-B, gamma=4 instead of the paper's default gamma=2, raw cosine similarity instead of the
per-token standardized correlation, and applied the reweighting to the block's *output* instead
of inside its attention. This file is the corrected version.

Honest scoping note: this mirror only ships MVTec-AD's *test* split (both "good" and defective
images per category, 1725 real images total). The official protocol trains on a separate
normal-only train split this mirror doesn't provide, so this reimplementation instead holds out
k "good"-labeled test images per category as the few-shot support set and evaluates on the
remaining (real, never-seen) test images -- a legitimate few-shot construction, just not the
paper's exact official split, so absolute numbers are not interchangeable with the paper's.
"""
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

IMG_SIZE = 448  # 448/16 = 28 -> 784 patch tokens; AnomalyDINO-style resolution
PATCH = 16
GRID = IMG_SIZE // PATCH
GAMMA = 2.0  # paper default (Fig. 5 ablation peaks here)
RADIO_ID = "nvidia/C-RADIOv3-B"
EPS = 1e-6


# --------------------------------------------------------------------------- data
def load_mvtec():
    from datasets import load_dataset
    return load_dataset("katiehahm/mvtec_ad", split="test")


def to_tensor(img):
    """PIL -> (3, H, W) float in [0, 1]. RADIO's input_conditioner normalizes internally."""
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0


# --------------------------------------------------------------------------- PL-SCEA
def plscea_weights(y: torch.Tensor, gamma: float) -> torch.Tensor:
    """Paper Eq. 3-9. y: (B, H, N, d) contextualized values Y = softmax(QK^T/sqrt(d)) V of the
    PATCH tokens only. Returns (B, H, N, N) row-stochastic PL-SCEA attention weights."""
    y_n = F.normalize(y, dim=-1, eps=EPS)                      # Eq. 3: L2-normalize tokens
    s = y_n @ y_n.transpose(-2, -1)                            # Eq. 4: cosine self-correlation
    mu = s.mean(dim=-1, keepdim=True)                          # Eq. 5: row-wise mean / std
    sigma = s.std(dim=-1, keepdim=True, unbiased=False)
    s_hat = (s - mu) / (sigma + EPS)                           # Eq. 6: token-adaptive standardization
    r = F.relu(s_hat).pow(gamma)                               # Eq. 7: keep above-average, power-law
    d = r.sum(dim=-1, keepdim=True)                            # Eq. 8: row-normalize...
    eye = torch.eye(r.shape[-1], device=r.device, dtype=r.dtype)
    return torch.where(d > 0, r / d.clamp_min(1e-12), eye)     # ...with identity fallback


def _attn_forward(self, x, attn_mask=None, is_causal=False):
    """Drop-in replacement for timm Attention.forward. mode 'qk' = the original attention;
    mode 'plscea' = prefix tokens keep the original QK attention, patch tokens are re-aggregated
    with PL-SCEA weights over the original values (Eq. 10)."""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    y = attn @ v                                               # Eq. 1-2: Y = A_qk V
    if self.scea_mode == "plscea":
        p = N - GRID * GRID                                    # number of prefix (cls/register) tokens
        a_pl = plscea_weights(y[:, :, p:], self.scea_gamma)
        o_patch = a_pl @ v[:, :, p:]                           # Eq. 10: O = A_PL-SCEA V
        y = torch.cat([y[:, :, :p], o_patch], dim=2)
    out = y.transpose(1, 2).reshape(B, N, self.attn_dim)
    return self.proj_drop(self.proj(self.norm(out)))


class Radio:
    """Frozen C-RADIOv3-B whose final-block attention can be switched between the original QK
    attention ('qk') and PL-SCEA ('plscea') -- same weights, zero new parameters."""

    def __init__(self, device):
        from transformers import AutoModel
        self.device = device
        self.model = AutoModel.from_pretrained(RADIO_ID, trust_remote_code=True).to(device).eval()
        attn = self.model.model.blocks[-1].attn
        attn.scea_mode, attn.scea_gamma = "qk", GAMMA
        attn.forward = types.MethodType(_attn_forward, attn)
        self.attn = attn

    @torch.no_grad()
    def patch_tokens(self, img, mode: str, gamma: float = GAMMA) -> torch.Tensor:
        """PIL image -> (N, D) L2-normalized spatial tokens (prefix tokens excluded)."""
        self.attn.scea_mode, self.attn.scea_gamma = mode, gamma
        _, feats = self.model(to_tensor(img)[None].to(self.device))
        assert feats.shape[1] == GRID * GRID, feats.shape
        return F.normalize(feats[0], dim=-1)


# --------------------------------------------------------------------------- LNM (VAE head)
class LNM(nn.Module):
    """Paper's Latent Normality Modeling: lightweight VAE over L2-normalized patch features,
    64-d latent. Hidden width, noise std and beta are not given in the paper -- my choices."""

    def __init__(self, dim, hidden=512, latent=64, noise=0.05, beta=1e-3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, latent * 2))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.latent, self.noise, self.beta = latent, noise, beta

    def loss(self, x_bar):
        x_in = F.normalize(x_bar + self.noise * torch.randn_like(x_bar), dim=-1)  # noise, renormalize
        mu, logvar = self.enc(x_in).chunk(2, dim=-1)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        recon = self.dec(z)
        l_cos = (1 - F.cosine_similarity(recon, x_bar, dim=-1)).mean()           # Eq. 11
        l_kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)).mean()
        return l_cos + self.beta * l_kl

    @torch.no_grad()
    def raw_score(self, x_bar):
        """Eq. 12: deterministic -- encode the clean feature, decode the posterior mean."""
        mu, _ = self.enc(x_bar).chunk(2, dim=-1)
        recon = F.normalize(self.dec(mu), dim=-1)
        return 1 - F.cosine_similarity(x_bar, recon, dim=-1)


def fit_lnm(support: torch.Tensor, epochs=600, bs=512, lr=8e-4, wd=1e-5, seed=0):
    """AdamW, lr 8e-4, wd 1e-5, batch 512, up to 600 epochs (paper's settings).
    Returns (model, mean, std) where mean/std standardize raw scores on the normal support patches."""
    torch.manual_seed(seed)
    m = LNM(support.shape[1]).to(support.device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    n = support.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=support.device)
        for i in range(0, n, bs):
            loss = m.loss(support[perm[i:i + bs]])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval()
    a = m.raw_score(support)
    return m, a.mean().item(), a.std().item()


def patch_score_map(m, mean, std, feats: torch.Tensor, smooth: str = "grid") -> np.ndarray:
    """Eq. 13: standardize with normal-support statistics, then Gaussian-smooth.
    smooth='grid'  : sigma=1 on the 28x28 patch grid (setup A).
    smooth='paper' : bilinear-upsample to IMG_SIZE and Gaussian sigma=4 px, as in AnomalyDINO (setup B)."""
    s = ((m.raw_score(feats) - mean) / (std + EPS)).reshape(GRID, GRID)
    if smooth == "grid":
        return gaussian_filter(s.cpu().numpy(), sigma=1.0)
    up = F.interpolate(s[None, None], size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
    return gaussian_filter(up.cpu().numpy(), sigma=4.0)


def image_score(score_map: np.ndarray) -> float:
    """Mean of the top 1% scores (patches on the grid map, pixels on the upsampled map)."""
    flat = np.sort(score_map.ravel())[::-1]
    return float(flat[: max(1, int(round(0.01 * flat.size)))].mean())
