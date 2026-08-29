# generate_diffusion_comparison.py
"""Generate the physics-vs-diffusion-vs-real comparison images for Part 2 of the
Synthetic Data series, by running the LoRA-fine-tuned SD-inpainting pipeline on
the *exact same* synthetic crack mask Part 1's physics simulation used.

Reuses Part 1's own deterministic setup (`generate_examples.py`): same base
image, same `start` location, same `generate_crack_mask(..., rng=default_rng(0))`
call -- so the only thing that differs between Part 1's already-saved
`synth-xray-defect-synthetic.png` and this script's output is the generation
method, not the mask or location.
"""
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from peft import LoraConfig
from PIL import Image

from synth_xray.data import download_and_extract, find_groundtruth_file, find_images_in_series, find_series_dirs
from synth_xray.defect_edit import generate_crack_mask
from synth_xray.groundtruth import parse_gdxray_bboxes
from train_diffusion_lora import MODEL_ID, PROMPT

DATA_DIR = Path(__file__).parent / ".data_cache"
LORA_DIR = Path(__file__).parent / "diffusion_out"
OUT_DIR = Path(__file__).parents[2] / "posts" / "series" / "synthetic-data" / "images"

# Top-left-aligned so this single 512x512 window contains both of Part 1's
# display crops verbatim: the wide crop `base_image[150:450, 0:500]` and the
# zoom crop `base_image[220:340, 0:200]` -- letting every downstream crop below
# reuse Part 1's exact same row/col indices with no re-offsetting.
WINDOW = (0, 512, 0, 512)


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float64)


def save_gray(array: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(path)


def image_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    group_dir = download_and_extract("Welds", DATA_DIR)
    series_dirs = [d for d in find_series_dirs(group_dir) if find_groundtruth_file(d) is not None]
    series_dir = series_dirs[0]
    gt_file = find_groundtruth_file(series_dir)
    boxes_by_image = parse_gdxray_bboxes(gt_file)
    images = find_images_in_series(series_dir)

    # Same two images, same roles, as generate_examples.py.
    defect_path = images[0]
    base_path = images[1] if len(images) > 1 else images[0]
    defect_image = load_gray(defect_path)
    base_image = load_gray(base_path)

    box = boxes_by_image[image_id_from_path(defect_path)][0]
    scale_row = base_image.shape[0] / defect_image.shape[0]
    scale_col = base_image.shape[1] / defect_image.shape[1]
    start = (int(box[0] * scale_row), int(box[1] * scale_col))

    rng = np.random.default_rng(0)
    synthetic_mask = generate_crack_mask(base_image.shape, start=start, length=40, thickness=2, rng=rng)

    r0, r1, c0, c1 = WINDOW
    window_image = base_image[r0:r1, c0:c1]
    window_mask = synthetic_mask[r0:r1, c0:c1]
    if window_mask.sum() == 0:
        raise RuntimeError("synthetic crack mask has no overlap with the comparison window -- WINDOW needs updating")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL_ID, dtype=torch.float32, safety_checker=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    pipe.unet.add_adapter(lora_config)
    lora_state_dict = torch.load(LORA_DIR / "lora_weights.pt", map_location=device, weights_only=True)
    missing, unexpected = pipe.unet.load_state_dict(lora_state_dict, strict=False)
    assert not unexpected, f"unexpected keys loading LoRA weights: {unexpected}"
    print(f"loaded {len(lora_state_dict)} real fine-tuned LoRA tensors into the UNet")

    image_pil = Image.fromarray(np.stack([np.clip(window_image, 0, 255).astype(np.uint8)] * 3, axis=-1))
    mask_pil = Image.fromarray((window_mask.astype(np.uint8) * 255))

    with torch.no_grad():
        result = pipe(prompt=PROMPT, image=image_pil, mask_image=mask_pil, num_inference_steps=30, guidance_scale=1.0).images[0]
    generated = np.array(result.convert("L"), dtype=np.float64)
    # Composite explicitly: keep the real input pixels everywhere outside the
    # mask (StableDiffusionInpaintPipeline's raw output differs from the input
    # everywhere, not just under the mask, because the whole image round-trips
    # through the VAE's lossy encode/decode -- only the masked region should
    # actually be replaced).
    diffusion_output = np.where(window_mask, generated, window_image)

    # Same indices as generate_examples.py -- valid unchanged since WINDOW starts at (0, 0).
    diffusion_crop = diffusion_output[150:450, 0:500]
    diffusion_zoom = diffusion_output[220:340, 0:200]
    save_gray(diffusion_crop, OUT_DIR / "synth-xray-defect-diffusion.png")
    save_gray(diffusion_zoom, OUT_DIR / "synth-xray-defect-diffusion-zoom.png")

    print(f"series: {series_dir.name}")
    print(f"base image (masked + inpainted): {base_path.name}")
    print(f"crack mask pixels: {int(synthetic_mask.sum())}")
    print("Saved synth-xray-defect-diffusion.png + -zoom.png")


if __name__ == "__main__":
    main()
