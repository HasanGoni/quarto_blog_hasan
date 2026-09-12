#!/usr/bin/env python3
"""Same Marigold V2 inference graph (single-step, fixed t=0.499, precomputed
depth prompt embedding) but with NO LoRA checkpoint loaded -- i.e. the raw,
un-fine-tuned Qwen-Image-Edit-2509 base model asked to do the exact same
single-step depth guess. This is the "normal checkpoint" baseline requested,
paired against the real Marigold V2 (LoRA fine-tuned) result in output/depth.

Reuses infer.py's own config builder so the two runs are identical except
for --checkpoint, which is simply omitted here (evaluate_pipeline.py's own
documented behavior: "Omit --checkpoint to run with pretrained weights only").
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "repo" / "scripts"))
from infer import _build_config, REPO_ROOT  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

image_dir = Path("xray_inputs").resolve()
output_dir = Path("output/depth_base").resolve()
output_dir.mkdir(parents=True, exist_ok=True)

config = _build_config(
    image_dir=image_dir,
    output_dir=output_dir,
    modality="depth",
    checkpoint=str(REPO_ROOT / "assets" / "checkpoints" / "Marigold-V2" / "depth" / "Log-stage2"),
    width=None,
    height=None,
    seed=2025,
)
config_path = output_dir / "config.yaml"
OmegaConf.save(config=OmegaConf.create(config), f=str(config_path))

print("Running base Qwen-Image-Edit-2509 with NO Marigold LoRA (pretrained weights only)...")
subprocess.run(
    [
        sys.executable, "-m", "evaluation.depth.evaluate_pipeline",
        "--config", str(config_path),
        "--output_dir", str(output_dir),
        # --checkpoint deliberately omitted
    ],
    cwd=str(REPO_ROOT),
    check=True,
)
