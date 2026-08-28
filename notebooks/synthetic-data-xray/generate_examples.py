# generate_examples.py
"""Generate the real before/after example images used in Part 1 of the Synthetic Data series."""
from pathlib import Path

import numpy as np
from PIL import Image

from synth_xray.data import download_and_extract, find_series_dirs, find_images_in_series, find_groundtruth_file
from synth_xray.groundtruth import parse_gdxray_bboxes, bbox_to_mask, refine_mask_otsu
from synth_xray.calibration import estimate_I0, fit_defect_attenuation_depth, fit_noise_model, fit_blur_sigma
from synth_xray.defect_edit import generate_crack_mask
from synth_xray.pipeline import synthesize_defect_image

DATA_DIR = Path(__file__).parent / ".data_cache"
OUT_DIR = Path(__file__).parents[2] / "posts" / "series" / "synthetic-data" / "images"


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float64)


def save_gray(array: np.ndarray, path: Path) -> None:
    normalized = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(normalized).save(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    group_dir = download_and_extract("Welds", DATA_DIR)
    series_dirs = [d for d in find_series_dirs(group_dir) if find_groundtruth_file(d) is not None]
    series_dir = series_dirs[0]
    gt_file = find_groundtruth_file(series_dir)
    boxes = parse_gdxray_bboxes(gt_file)
    images = find_images_in_series(series_dir)

    defect_image = load_gray(images[0])
    box = boxes[0]
    roi_mask = bbox_to_mask(defect_image.shape, box)
    defect_mask = refine_mask_otsu(defect_image, roi_mask)
    background_mask = ~roi_mask

    I0 = estimate_I0(defect_image)
    delta = fit_defect_attenuation_depth(defect_image, defect_mask, background_mask, I0)

    # NOTE: this weld image has a strong large-scale brightness gradient (the bright
    # weld-bead band sweeping across an otherwise dark plate) -- the brief's original
    # 2x2-quadrant "flat" patches each still span most of that gradient, so their
    # variance is dominated by the gradient itself rather than pixel-level photon
    # noise. Measured directly: real image flat-region std is ~4 (at mean ~45), but
    # fitting on quadrant patches gave a noise model predicting std ~15-20 there --
    # a visibly over-grainy synthetic image once applied everywhere. Using a dense
    # grid of small (20x20) patches instead keeps each patch inside one locally-flat
    # region, and the fitted model then matches the real image's actual local noise
    # level (predicted std ~6-8 across the observed brightness range).
    patch_h, patch_w = 20, 20
    patch_masks = []
    for r in range(0, defect_image.shape[0] - patch_h, patch_h):
        for c in range(0, defect_image.shape[1] - patch_w, patch_w):
            m = np.zeros(defect_image.shape, dtype=bool)
            m[r:r + patch_h, c:c + patch_w] = True
            if (m & roi_mask).any():
                continue
            patch_masks.append(m)
    a, b = fit_noise_model(defect_image, patch_masks)

    row_c, col_c = box[0] + (box[2] - box[0]) // 2, box[1] + (box[3] - box[1]) // 2
    edge_profile = defect_image[row_c, max(col_c - 15, 0):col_c + 15]
    blur_sigma = fit_blur_sigma(edge_profile) if edge_profile.size >= 10 else 1.5
    # NOTE: this image has no genuinely sharp step edge near the chosen defect --
    # every profile tried (through the defect box, and through the weld-bead band
    # edge elsewhere in the image) is dominated by the same slow, large-scale
    # brightness gradient described above, not a local detector-blur-scale
    # transition. `fit_blur_sigma` fits *some* sigma to that slope regardless
    # (10-40 px, depending on where the profile is taken), but a blur that large
    # applied to the ~11x11 px synthetic crack washes it out to invisibility. Real
    # detector/geometric unsharpness for this modality/resolution is a few pixels
    # at most, so treat a fitted value above that as gradient contamination rather
    # than a real blur measurement and fall back to a small, physically-plausible
    # constant.
    if not (0 < blur_sigma <= 5):
        blur_sigma = 1.5

    clean_image = load_gray(images[1]) if len(images) > 1 else defect_image.copy()
    rng = np.random.default_rng(0)
    synthetic_mask = generate_crack_mask(clean_image.shape, start=box[:2], length=40, thickness=2, rng=rng)
    synthetic_image, _ = synthesize_defect_image(
        clean_image, I0=I0, defect_mask=synthetic_mask, delta=delta,
        noise_gain=max(a, 0.1), blur_sigma=blur_sigma, rng=rng,
    )

    save_gray(clean_image, OUT_DIR / "synth-xray-clean.png")
    save_gray(defect_image, OUT_DIR / "synth-xray-defect-real.png")
    save_gray(synthetic_image, OUT_DIR / "synth-xray-defect-synthetic.png")

    print(f"series: {series_dir.name}")
    print(f"fitted I0: {I0:.2f}")
    print(f"fitted attenuation-depth delta: {delta:.4f}")
    print(f"fitted noise model: variance = {a:.3f} * mean + {b:.3f}")
    print(f"fitted blur sigma: {blur_sigma:.3f} px")


if __name__ == "__main__":
    main()
