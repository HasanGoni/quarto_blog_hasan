#!/usr/bin/env python3
"""Same as run_base_only.py but for the normals modality."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "repo" / "scripts"))
from infer import _build_config, REPO_ROOT  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

image_dir = Path("xray_inputs").resolve()
output_dir = Path("output/normals_base").resolve()
output_dir.mkdir(parents=True, exist_ok=True)

config = _build_config(
    image_dir=image_dir,
    output_dir=output_dir,
    modality="normals",
    checkpoint=str(REPO_ROOT / "assets" / "checkpoints" / "Marigold-V2" / "normals"),
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
    ],
    cwd=str(REPO_ROOT),
    check=True,
)
