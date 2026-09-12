#!/usr/bin/env python3
"""Run Marigold V2 and save each modality in its own output folder.

Reads raw images from ``--image_dir`` and writes PNG visualizations to sibling
folders under ``--output_root`` (default: parent of image_dir):

  <output_root>/depth/<stem>.png
  <output_root>/normals/<stem>.png
  <output_root>/albedo/<stem>.png

Resume-safe: skips inputs that already have an output PNG. Optional vLLM
stop/start for GPU memory.

Example:
  .venv/bin/python run_infer_modality_folders.py \\
    --image_dir /path/to/train/images \\
    --stop-vllm --restart-vllm
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "repo" / "scripts"))
from infer import MODALITIES, REPO_ROOT, _build_config  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MODALITY_DIRS = ("depth", "normals", "albedo")
VIZ_SUBDIRS = {
    "depth": "images/visualizations/depth_spectral",
    "normals": "images/visualizations/normals",
    "albedo": "images/visualizations/albedo",
}
VLLM_UNIT = "vllm-cursor-router.service"


def _input_images(image_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _output_path(modality_dir: Path, stem: str) -> Path:
    return modality_dir / f"{stem}.png"


def _pending(modality_dir: Path, inputs: list[Path]) -> list[Path]:
    return [p for p in inputs if not _output_path(modality_dir, p.stem).exists()]


def _stop_vllm() -> None:
    print(f"stopping {VLLM_UNIT} to free GPU memory...", flush=True)
    subprocess.run(["systemctl", "--user", "stop", VLLM_UNIT], check=False)


def _start_vllm() -> None:
    print(f"starting {VLLM_UNIT}...", flush=True)
    subprocess.run(["systemctl", "--user", "start", VLLM_UNIT], check=False)


def _run_modality(
    output_root: Path,
    modality: str,
    seed: int,
    pending: list[Path],
) -> None:
    modality_dir = output_root / modality
    modality_dir.mkdir(parents=True, exist_ok=True)

    if not pending:
        print(f"[{modality}] all outputs already present — skip", flush=True)
        return

    checkpoint = str(MODALITIES[modality]["checkpoint"])
    with tempfile.TemporaryDirectory(prefix=f"marigold_{modality}_") as tmp:
        tmp_root = Path(tmp)
        tmp_in = tmp_root / "inputs"
        tmp_out = tmp_root / "outputs"
        tmp_in.mkdir()
        tmp_out.mkdir()

        for src in pending:
            (tmp_in / src.name).symlink_to(src.resolve())

        config = _build_config(
            image_dir=tmp_in,
            output_dir=tmp_out,
            modality=modality,
            checkpoint=checkpoint,
            width=None,
            height=None,
            seed=seed,
        )
        config_path = tmp_out / "config.yaml"
        OmegaConf.save(config=OmegaConf.create(config), f=str(config_path))

        print(
            f"[{modality}] {len(pending)} image(s) -> {modality_dir}",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.depth.evaluate_pipeline",
                "--config",
                str(config_path),
                "--output_dir",
                str(tmp_out),
                "--checkpoint",
                checkpoint,
            ],
            cwd=str(REPO_ROOT),
            check=True,
        )

        viz_dir = tmp_out / VIZ_SUBDIRS[modality]
        if not viz_dir.is_dir():
            raise FileNotFoundError(f"No visualizations under {viz_dir}")

        for src in sorted(viz_dir.glob("*.png")):
            dst = _output_path(modality_dir, src.stem)
            shutil.copy2(src, dst)
            print(f"  -> {modality}/{dst.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Folder of input images (e.g. .../train/images).",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Parent for depth/normals/albedo folders (default: image_dir.parent).",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--stop-vllm",
        action="store_true",
        help=f"Stop {VLLM_UNIT} before inference.",
    )
    parser.add_argument(
        "--restart-vllm",
        action="store_true",
        help=f"Start {VLLM_UNIT} again after all modalities finish.",
    )
    args = parser.parse_args()

    image_dir = args.image_dir.resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")

    output_root = (args.output_root or image_dir.parent).resolve()
    inputs = _input_images(image_dir)
    if not inputs:
        raise ValueError(f"No input images found in {image_dir}")

    if args.stop_vllm:
        _stop_vllm()

    print(f"{len(inputs)} input image(s) in {image_dir}", flush=True)
    print(f"output folders under {output_root}", flush=True)
    try:
        for modality in MODALITY_DIRS:
            modality_dir = output_root / modality
            pending = _pending(modality_dir, inputs)
            _run_modality(output_root, modality, args.seed, pending)
    finally:
        if args.restart_vllm:
            _start_vllm()

    print("done", flush=True)


if __name__ == "__main__":
    main()
