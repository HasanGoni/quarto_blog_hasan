#!/usr/bin/env python3
"""Minimal asset download for Marigold V2 inference only (no training data,
no text encoder -- the inference pipeline never loads it, confirmed against
marigoldv2/experiments/20260316_qwen_depth/component_loader.py)."""

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent / "repo"
CKPT_DIR = REPO_ROOT / "assets" / "checkpoints"

print("[1/2] Qwen-Image-Edit-2509 base (transformer + vae only)", flush=True)
snapshot_download(
    repo_id="Qwen/Qwen-Image-Edit-2509",
    repo_type="model",
    local_dir=str(CKPT_DIR / "Qwen-Image-Edit-2509"),
    allow_patterns=[
        "model_index.json",
        "scheduler/*",
        "transformer/*",
        "vae/*",
    ],
    max_workers=1,
)

print("[2/2] Marigold-V2 LoRA checkpoints (depth + normals + albedo)", flush=True)
snapshot_download(
    repo_id="huawei-bayerlab/marigold-v2-0",
    repo_type="model",
    local_dir=str(CKPT_DIR / "Marigold-V2"),
    allow_patterns=[
        "depth/Log-stage2/*",
        "normals/*",
        "albedo/*",
        "qwen_text_embeddings/*",
        "manifest.json",
    ],
    max_workers=1,
)

print("[done]", flush=True)
