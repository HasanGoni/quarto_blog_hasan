"""Shared utilities: real MVTec-AD loading (via katiehahm/mvtec_ad, image+gt_label+class,
CC-BY-NC-4.0) and the PL-SCEA attention-reconfiguration mechanism itself.

Honest scoping note: this mirror only ships MVTec-AD's *test* split (both "good" and defective
images per category, 1725 real images total). The official protocol trains on a separate
normal-only train split this mirror doesn't provide, so this reimplementation instead holds out
k "good"-labeled test images per category as the few-shot support set and evaluates on the
remaining (real, never-seen) test images -- a legitimate few-shot construction, just not the
paper's exact official split.
"""
import numpy as np
import torch
import torch.nn.functional as F

IMG_SIZE = 224  # DINOv2-small's native/comfortable resolution
K_SHOT = 8
GAMMA = 4.0  # power-law exponent


def sceareconfigure(features: torch.Tensor, gamma: float = GAMMA) -> torch.Tensor:
    """The real PL-SCEA mechanism, applied post-hoc to frozen ViT patch features (standing in
    for the paper's "contextualized value features", since hooking DINOv2's internal per-layer
    value projections would need surgery on the model's forward pass -- documented simplification,
    not a hidden one): token-adaptive self-correlation, positive-correlation filtering, power-law
    reweighting, all with zero new trainable parameters.

    features: (N, D) patch tokens for ONE image (already L2-normalized).
    Returns: (N, D) reconfigured features -- each token replaced by a correlation-weighted
    average of every OTHER token whose value-feature is positively, strongly self-similar to it.
    """
    sim = features @ features.T  # (N, N) cosine similarity, since features are L2-normalized
    sim = sim - torch.eye(sim.shape[0], device=sim.device) * 2  # mask self-similarity out
    sim_pos = F.relu(sim)  # positive-correlation filtering
    weights = sim_pos.pow(gamma)  # power-law reweighting: emphasize only strong correlations
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    return weights @ features


def load_mvtec():
    from datasets import load_dataset
    return load_dataset("katiehahm/mvtec_ad", split="test")


def resize(img):
    return img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
