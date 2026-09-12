#!/usr/bin/env python3
"""Run Marigold V2 (main LoRA checkpoints) and save maps next to each input image.

Company / local-only data: outputs land in the *same folder* as the inputs, with
suffix names — nothing under output/depth|normals|albedo in this repo.

The Marigold authors' pipeline already walks every image in a folder per modality
(batch size 1 — safest for a 21B 4-bit model). This script adds resume, skips
outputs when re-run, and optional vLLM stop/start so GPU memory isn't contested.

Example:
  .venv/bin/python run_infer_colocated.py --image_dir images/train --stop-vllm
  .venv/bin/python run_infer_colocated.py --image_dir /data/company_xrays --stop-vllm --restart-vllm

For each ``foo.png`` you get (same folder):
  foo_marigold_depth.png
  foo_marigold_normals.png
  foo_marigold_albedo.png
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
VIZ_SUBDIRS = {
    "depth": "images/visualizations/depth_spectral",
    "normals": "images/visualizations/normals",
    "albedo": "images/visualizations/albedo",
}
SUFFIXES = {
    "depth": "_marigold_depth.png",
    "normals": "_marigold_normals.png",
    "albedo": "_marigold_albedo.png",
}
VLLM_UNIT = "vllm-cursor-router.service"


def _input_images(image_dir: Path) -> list[Path]:
    """Raw inputs only — ignore colocated Marigold outputs."""
    return sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and "_marigold_" not in p.stem
    )


def _output_path(image_dir: Path, stem: str, modality: str) -> Path:
    return image_dir / f"{stem}{SUFFIXES[modality]}"


def _pending_for_modality(image_dir: Path, inputs: list[Path], modality: str) -> list[Path]:
    return [p for p in inputs if not _output_path(image_dir, p.stem, modality).exists()]


def _stop_vllm() -> None:
    print(f"stopping {VLLM_UNIT} to free GPU memory...", flush=True)
    subprocess.run(
        ["systemctl", "--user", "stop", VLLM_UNIT],
        check=False,
    )


def _start_vllm() -> None:
    print(f"starting {VLLM_UNIT}...", flush=True)
    subprocess.run(
        ["systemctl", "--user", "start", VLLM_UNIT],
        check=False,
    )


def _run_modality(
    image_dir: Path,
    modality: str,
    seed: int,
    pending: list[Path],
) -> None:
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
            f"[{modality}] {len(pending)} image(s); checkpoint: {checkpoint}",
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

        suffix = SUFFIXES[modality]
        for src in sorted(viz_dir.glob("*.png")):
            dst = image_dir / f"{src.stem}{suffix}"
            shutil.copy2(src, dst)
            print(f"  -> {dst.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=Path("images/train"),
        help="Folder of input images (default: images/train). Outputs written here.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--stop-vllm",
        action="store_true",
        help=f"Stop {VLLM_UNIT} before inference (frees ~60%% GPU for Marigold).",
    )
    parser.add_argument(
        "--restart-vllm",
        action="store_true",
        help=f"Start {VLLM_UNIT} again after all modalities finish.",
    )
    args = parser.parse_args()

    image_dir = args.image_dir.resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"image_dir does not exist: {image_dir}\n"
            "Create it and drop X-ray PNGs/JPEGs there (gitignored)."
        )

    inputs = _input_images(image_dir)
    if not inputs:
        raise ValueError(f"No input images found in {image_dir}")

    if args.stop_vllm:
        _stop_vllm()

    print(f"{len(inputs)} input image(s) in {image_dir}", flush=True)
    try:
        for modality in ("depth", "normals", "albedo"):
            pending = _pending_for_modality(image_dir, inputs, modality)
            _run_modality(image_dir, modality, args.seed, pending)
    finally:
        if args.restart_vllm:
            _start_vllm()

    print("done", flush=True)


if __name__ == "__main__":
    main()
