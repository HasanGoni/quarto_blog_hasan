"""HPMA core modules — real reimplementation of the paper's equations 1-7.

Hierarchical Prototype-Memory Adaptation of SAM for Surgical Instrument
Segmentation (arXiv:2608.24541). Wired against the real `facebook/sam3`
architecture (see hpma_sam3.py): three FPN levels are used as the three
scales — index 0 (288x288, highest-res) is "local" (the paper's F0), index 1
(144x144) is "structural", index 2 (72x72, the level SAM3's own DETR
encoder/decoder actually consumes) is "global".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PrototypeBank:
    """Frozen prototype memory. `vectors[scale][category]` -> (K, d) tensor.

    Corresponds to the paper's memory bank P in R^(3 x |C| x K x d).
    """
    vectors: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)

    def to(self, device):
        for scale in self.vectors:
            for c in self.vectors[scale]:
                self.vectors[scale][c] = self.vectors[scale][c].to(device)
        return self

    def save(self, path):
        torch.save({s: {c: v.cpu() for c, v in d.items()} for s, d in self.vectors.items()}, path)

    @classmethod
    def load(cls, path):
        raw = torch.load(path, weights_only=True)
        return cls(vectors=raw)


class MLPProjection(nn.Module):
    """Phi: LayerNorm + two-layer MLP with GELU (paper, Sec. on coupling adapters)."""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))


class GlobalCouplingAdapter(nn.Module):
    """Eq: T~c = Tc + alpha_g * (A_c^g . Vc) . WO, cross-attention with text
    tokens Tc as queries and the (projected) global prototypes as K/V."""

    def __init__(self, dim: int = 256, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.phi = MLPProjection(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, text_tokens: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """text_tokens: (B, S, d). prototypes: (K, d) for the active category."""
        b, s, d = text_tokens.shape
        proto = self.phi(prototypes).unsqueeze(0).expand(b, -1, -1)  # (B, K, d)

        q = self.wq(text_tokens).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(proto).view(b, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(proto).view(b, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, s, d)
        return text_tokens + self.alpha * self.wo(out)


class StructuralCouplingAdapter(nn.Module):
    """Eq: Q~c = Q + alpha_p * Phi_p(Pc^p), broadcast-added to every decoder
    object query for the active category."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.phi = MLPProjection(dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, query_embeds: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """query_embeds: (B, num_queries, d). prototypes: (K, d) -> mean-pooled to (d,)."""
        delta = self.phi(prototypes.mean(dim=0, keepdim=True))  # (1, d)
        return query_embeds + self.alpha * delta.unsqueeze(0)


def local_alignment_loss(f0: torch.Tensor, mask: torch.Tensor, local_prototypes: torch.Tensor) -> torch.Tensor:
    """Eq 7: L_align = mean_i [ min_k (1 - cos(phi_i, p_{c_i,k}^l)) ].

    f0: (B, d, H, W) highest-res FPN feature map (local scale).
    mask: (B, H, W) boolean mask (already resized to f0's spatial size) of the
    target-category pixels for that image.
    local_prototypes: (K, d) frozen local-scale prototypes for that category.
    """
    losses = []
    for b in range(f0.shape[0]):
        m = mask[b]
        if m.sum() == 0:
            continue
        feats = f0[b][:, m]  # (d, N)
        feats = feats.transpose(0, 1)  # (N, d)
        feats = F.normalize(feats, dim=-1)
        protos = F.normalize(local_prototypes, dim=-1)  # (K, d)
        cos_sim = feats @ protos.T  # (N, K)
        best = cos_sim.max(dim=-1).values  # (N,) -> max cos sim = min cos distance
        losses.append((1 - best).mean())
    if not losses:
        return f0.new_tensor(0.0, requires_grad=True)
    return torch.stack(losses).mean()


def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    pred = pred_logits.sigmoid().flatten(1)
    target = target.flatten(1).float()
    intersection = (pred * target).sum(-1)
    union = pred.sum(-1) + target.sum(-1)
    return (1 - (2 * intersection + eps) / (union + eps)).mean()
