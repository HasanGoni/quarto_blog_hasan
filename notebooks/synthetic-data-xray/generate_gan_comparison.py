# generate_gan_comparison.py
"""Generate the physics-vs-diffusion-vs-ControlNet-vs-Flux-vs-GAN-vs-real comparison
images for Part 7, by running the trained conditional GAN generator on the *exact
same* synthetic crack mask Parts 1-4 all used.

Reuses Part 1's own deterministic setup (`generate_examples.py`): same base image,
same `start` location, same `generate_crack_mask(..., rng=default_rng(0))` call --
so the only thing that differs across every part's saved comparison image is the
generation method, never the mask, location, or crop window.

Real, disclosed difference from Parts 2-4: the GAN was trained from scratch at
128x128 (`train_gan.py`'s RESOLUTION), not natively at 512 like the pretrained
diffusion checkpoints. The masked window is downsampled to 128 for the generator,
and its output is upsampled back to 512 (bicubic) purely so it can be composited
into the same real 512 base image and shown at the same crop/zoom windows as every
other part -- a real resizing step, not hidden.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from synth_xray.data import download_and_extract, find_groundtruth_file, find_images_in_series, find_series_dirs
from synth_xray.defect_edit import generate_crack_mask
from synth_xray.groundtruth import parse_gdxray_bboxes
from train_gan import RESOLUTION, Generator

DATA_DIR = Path(__file__).parent / ".data_cache"
GAN_DIR = Path(__file__).parent / "gan_out"
OUT_DIR = Path(__file__).parents[2] / "posts" / "series" / "synthetic-data" / "images"

# Same window Parts 2-4 used.
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator(noise_dim=32).to(device)
    G.load_state_dict(torch.load(GAN_DIR / "gan_generator.pt", map_location=device, weights_only=True))
    G.eval()
    print(f"loaded real trained generator from {GAN_DIR / 'gan_generator.pt'}")

    masked_window = window_image.copy()
    masked_window[window_mask] = 0

    masked_pil = Image.fromarray(np.clip(masked_window, 0, 255).astype(np.uint8)).resize(
        (RESOLUTION, RESOLUTION), Image.LANCZOS
    )
    mask_pil = Image.fromarray(window_mask.astype(np.uint8) * 255).resize((RESOLUTION, RESOLUTION), Image.NEAREST)

    masked_t = torch.from_numpy(np.array(masked_pil, dtype=np.float32) / 127.5 - 1.0).view(1, 1, RESOLUTION, RESOLUTION).to(device)
    mask_t = torch.from_numpy(np.array(mask_pil, dtype=np.float32) / 127.5 - 1.0).view(1, 1, RESOLUTION, RESOLUTION).to(device)
    noise = torch.randn(1, 32, device=device, generator=torch.Generator(device=device).manual_seed(0))

    with torch.no_grad():
        fake = G(masked_t, mask_t, noise)
    fake_np = ((fake[0, 0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    fake_upsampled = np.array(
        Image.fromarray(fake_np).resize((window_image.shape[1], window_image.shape[0]), Image.BICUBIC),
        dtype=np.float64,
    )

    # Composite: only paste the GAN's generated content inside the mask, matching every
    # other part's convention -- everything outside the mask stays exactly the real pixels.
    gan_output = np.where(window_mask, fake_upsampled, window_image)

    gan_crop = gan_output[150:450, 0:500]
    gan_zoom = gan_output[220:340, 0:200]
    save_gray(gan_crop, OUT_DIR / "synth-xray-defect-gan.png")
    save_gray(gan_zoom, OUT_DIR / "synth-xray-defect-gan-zoom.png")

    print(f"series: {series_dir.name}")
    print(f"base image (masked + generated): {base_path.name}")
    print(f"crack mask pixels: {int(synthetic_mask.sum())}")
    print("Saved synth-xray-defect-gan.png + -zoom.png")


if __name__ == "__main__":
    main()
