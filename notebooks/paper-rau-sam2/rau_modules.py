"""RAU core modules — real reimplementation of the paper's Eq. 6-7 (Section
3.3) plus the DINOv2 reference-retrieval step (Fig. 2a).

Reference-based Anatomical Understanding with Vision Language Models
(arXiv:2509.22404). SAM2 backbone: the paper builds its "memory bank" from
SAM2's own video-native memory encoder, populated from a reference mask. We
build the functional equivalent with SAM2's plain image encoder instead —
masked-average-pooled vision features per labeled region of the retrieved
reference image — and feed the fused query into SAM2's mask decoder through
its public, documented `target_embedding` parameter (the same interface SAM2
exposes for PerSAM-style semantic prompting). Equations 6-7 are implemented
exactly; the memory *construction* substitutes SAM2's video/session memory
machinery for a simpler, non-video masked-pooling step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReferenceBank:
    """One entry per reference image: DINOv2 global embedding (for
    retrieval, Fig. 2a) + per-label SAM2 memory vectors (for Eq. 7)."""
    dino_embeds: torch.Tensor = None  # (N, d_dino)
    memory_vectors: list[dict[int, torch.Tensor]] = field(default_factory=list)  # per-image {label: (d_sam2,)}
    images: list = field(default_factory=list)
    masks: list = field(default_factory=list)

    def to(self, device):
        self.dino_embeds = self.dino_embeds.to(device)
        self.memory_vectors = [{k: v.to(device) for k, v in d.items()} for d in self.memory_vectors]
        return self

    def save(self, path):
        torch.save({
            "dino_embeds": self.dino_embeds.cpu(),
            "memory_vectors": [{k: v.cpu() for k, v in d.items()} for d in self.memory_vectors],
        }, path)

    @classmethod
    def load(cls, path):
        raw = torch.load(path, weights_only=True)
        return cls(dino_embeds=raw["dino_embeds"], memory_vectors=raw["memory_vectors"])


def retrieve_reference(target_dino_embed: torch.Tensor, bank: ReferenceBank) -> int:
    """Fig. 2a: cosine similarity retrieval over the reference bank."""
    sims = F.cosine_similarity(target_dino_embed.unsqueeze(0), bank.dino_embeds, dim=-1)
    return sims.argmax().item()


class SegQueryProjection(nn.Module):
    """Eq. 6: q_i = MLP(h_i^<Seg>). Projects the VLM's <SEG> hidden state
    into SAM2's mask-decoder embedding space (dim=256)."""

    def __init__(self, vlm_dim: int, sam2_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vlm_dim, sam2_dim),
            nn.GELU(),
            nn.Linear(sam2_dim, sam2_dim),
        )

    def forward(self, h_seg: torch.Tensor) -> torch.Tensor:
        return self.net(h_seg)


def memory_attention(query: torch.Tensor, memory_vectors: dict[int, torch.Tensor]) -> torch.Tensor:
    """Eq. 7: dot-product attention of the projected query over the
    retrieved reference's per-label memory slots {m_j}.

    query: (d,). memory_vectors: {label: (d,)}. Returns fused (d,) z_i.
    """
    labels = sorted(memory_vectors.keys())
    m = torch.stack([memory_vectors[l] for l in labels])  # (K, d)
    scores = (query.unsqueeze(0) * m).sum(-1) / (query.shape[-1] ** 0.5)  # (K,)
    attn = scores.softmax(dim=-1)
    return (attn.unsqueeze(-1) * m).sum(0)  # (d,)


def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    pred = pred_logits.sigmoid().flatten(1)
    target = target.flatten(1).float()
    intersection = (pred * target).sum(-1)
    union = pred.sum(-1) + target.sum(-1)
    return (1 - (2 * intersection + eps) / (union + eps)).mean()
