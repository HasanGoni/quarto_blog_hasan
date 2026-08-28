# generate_examples.py
"""Generate the real before/after example images used in Part 1 of the Synthetic Data series."""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from synth_xray.data import download_and_extract, find_series_dirs, find_images_in_series, find_groundtruth_file
from synth_xray.groundtruth import parse_gdxray_bboxes, bbox_to_mask, refine_mask_otsu
from synth_xray.calibration import estimate_I0, fit_defect_attenuation_depth, fit_noise_model, fit_blur_sigma
from synth_xray.defect_edit import generate_crack_mask
from synth_xray.pipeline import synthesize_defect_image

DATA_DIR = Path(__file__).parent / ".data_cache"
OUT_DIR = Path(__file__).parents[2] / "posts" / "series" / "synthetic-data" / "images"

# Margin (px) grown around the defect ROI to get a *local* background ring for the
# attenuation-depth fit. See the NOTE in main() for why a whole-image background
# is wrong for these panoramas.
BACKGROUND_RING_PX = 25


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float64)


def save_gray(array: np.ndarray, path: Path) -> None:
    normalized = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(normalized).save(path)


def image_id_from_path(path: Path) -> int:
    """Map a GDXray series image filename to its ground-truth `image_id`.

    GDXray series images are named `<SERIES>_<NNNN>.png` (e.g. `W0001_0002.png`),
    and the ground-truth file's leading image_id column counts from 1 in that same
    order -- verified on the real Welds/W0001 data by grouping ground_truth.txt's
    641 boxes by image_id and checking each group's coordinate extents against the
    corresponding numbered PNG's actual (width, height): every one of the 10 ids
    fits inside its same-numbered file's bounds, several of them tightly.
    """
    return int(path.stem.split("_")[-1])


def dense_patch_masks(shape: tuple[int, int], patch: int = 20) -> list[np.ndarray]:
    """Tile an image into small square masks for a photon-transfer-curve fit.

    NOTE: patch size matters more than it looks. These weld panoramas have a strong
    large-scale brightness gradient (the bright weld-bead band sweeping across an
    otherwise dark plate), so the obvious choice of a 2x2 grid of quadrant "flat"
    patches has each patch spanning most of that gradient -- its variance is then
    dominated by the gradient itself rather than by pixel-level photon noise.
    Measured, that gave a noise model predicting a standard deviation roughly an
    order of magnitude above the image's real local std, i.e. a visibly
    over-grainy synthetic image. Small patches
    keep each one inside a locally-flat region, so the fitted model matches the
    image's actual local noise level.
    """
    masks = []
    for r in range(0, shape[0] - patch, patch):
        for c in range(0, shape[1] - patch, patch):
            m = np.zeros(shape, dtype=bool)
            m[r:r + patch, c:c + patch] = True
            masks.append(m)
    return masks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    group_dir = download_and_extract("Welds", DATA_DIR)
    series_dirs = [d for d in find_series_dirs(group_dir) if find_groundtruth_file(d) is not None]
    series_dir = series_dirs[0]
    gt_file = find_groundtruth_file(series_dir)
    boxes_by_image = parse_gdxray_bboxes(gt_file)
    images = find_images_in_series(series_dir)

    # Two different real radiographs, doing two different jobs:
    #   defect_image (images[0]) -- has annotated ground-truth defects, so it is what
    #     the *defect physics* (the attenuation-depth signature `delta`) is fit from.
    #   base_image  (images[1]) -- the radiograph the synthetic crack is injected into,
    #     so it is what the *sensor/acquisition* characteristics (noise model, blur)
    #     are fit from. Those are properties of a specific capture -- exposure, dose,
    #     detector settings -- not of defect physics, so fitting them on defect_image
    #     and applying them to base_image made the output several times grainier than
    #     its own source. Fit them where they will be applied.
    defect_path = images[0]
    base_path = images[1] if len(images) > 1 else images[0]
    defect_image = load_gray(defect_path)
    base_image = load_gray(base_path)

    # Select the boxes belonging to *this* image by its ground-truth image_id, rather
    # than trusting the ground-truth file's line order to line up with the image list.
    box = boxes_by_image[image_id_from_path(defect_path)][0]
    roi_mask = bbox_to_mask(defect_image.shape, box)
    defect_mask = refine_mask_otsu(defect_image, roi_mask)
    # NOTE: the background reference for the defect fit is a *local ring* around the
    # ROI, not the whole-image complement. These panoramas have a strong large-scale
    # brightness gradient, so a whole-image background median averages over the dark
    # plate and the bright weld bead alike and has nothing to do with the material
    # immediately surrounding this defect. Measured on W0001_0001, delta by background
    # choice: whole image -0.4925, 25 px ring -0.3494 (used here), 10 px ring -0.2412,
    # 5 px ring -0.1974. Same (negative) polarity throughout; the whole-image version
    # inflates the magnitude. There is no single "true" delta -- only a stated
    # background reference. 25 px is the compromise: local enough to share the
    # defect's gradient context, wide enough (~2700 px) for a stable median.
    dilated_mask = binary_dilation(roi_mask, structure=np.ones((3, 3), bool), iterations=BACKGROUND_RING_PX)
    background_mask = dilated_mask & ~roi_mask

    # I0 is fit from defect_image, which is harmless here: `render_xray` and
    # `invert_to_attenuation_depth` are exact algebraic inverses
    # (I0 * exp(-(-log(x/I0))) == x), so I0 cancels out of the round trip and only
    # scales the intermediate depth map. delta is likewise a *difference* of two
    # log-depths, so the I0 term cancels there too.
    I0 = estimate_I0(defect_image)
    delta = fit_defect_attenuation_depth(defect_image, defect_mask, background_mask, I0)

    # Noise and blur are fit from base_image -- the image actually being synthesized
    # onto. Simplification worth stating: base_image is not defect-free either (no
    # image in this GDXray series is -- it carries 37 of its own annotated defects),
    # and these patches do not exclude them. They cover a tiny fraction of the frame,
    # so their effect on a median-dominated dense-grid noise fit is negligible.
    patch_masks = dense_patch_masks(base_image.shape)
    a, b = fit_noise_model(base_image, patch_masks)

    # `box` is in defect_image's (images[0]) coordinate frame, but the crack is drawn
    # on base_image (images[1]) -- a different real GDXray image with its own
    # resolution and aspect ratio (e.g. here 366x3512 vs 835x4919). Copying box[:2]
    # as a raw pixel coordinate would be meaningless on base_image (it'd only be
    # in-bounds by luck, and wouldn't correspond to anything). Scale it
    # proportionally into base_image's shape instead, so the start point still
    # lands "near where a real defect was" in relative terms.
    scale_row = base_image.shape[0] / defect_image.shape[0]
    scale_col = base_image.shape[1] / defect_image.shape[1]
    start = (int(box[0] * scale_row), int(box[1] * scale_col))

    # Blur is measured on base_image too, along a profile through the site the crack
    # is about to be injected into -- the place where the blur actually has to be right.
    edge_profile = base_image[start[0], max(start[1] - 15, 0):start[1] + 15]
    try:
        blur_sigma = fit_blur_sigma(edge_profile) if edge_profile.size >= 10 else float("nan")
    except RuntimeError:
        blur_sigma = float("nan")
    # NOTE: these images contain no genuinely sharp step edge to measure. Every
    # profile tried is either dominated by the same slow, large-scale brightness
    # gradient the noise patches ran into, or is locally flat enough that the
    # step-edge fit degenerates. `fit_blur_sigma` returns *some* sigma either way --
    # 10-40 px on a gradient-riding profile, or a near-zero, non-identifiable sigma
    # (scipy cannot even estimate its covariance) on a flat one. Neither is a real
    # blur measurement: a 10-40 px blur would wash the ~11x11 px synthetic crack out
    # to invisibility, and a ~0 px blur means no detector unsharpness at all. Real
    # detector/geometric unsharpness for this modality/resolution is a sub-pixel to
    # few-pixel affair, so anything outside [0.5, 5] px is treated as a failed fit
    # and falls back to a small, physically-plausible constant.
    if not (0.5 <= blur_sigma <= 5):
        blur_sigma = 1.5

    rng = np.random.default_rng(0)
    synthetic_mask = generate_crack_mask(base_image.shape, start=start, length=40, thickness=2, rng=rng)
    # `a` is the photon-transfer slope: `apply_photon_noise` draws Poisson counts at
    # signal/gain, so gain == a reproduces variance = a*mean. The floor here only
    # guards against a degenerate non-positive fit (which would divide by zero);
    # it is deliberately far below any plausible fitted slope, because clamping it
    # up to a "safe-looking" constant would silently throw the calibration away and
    # re-introduce exactly the over-grainy output this pipeline is trying to avoid.
    noise_gain = a if a > 1e-3 else 1e-3
    synthetic_image, _ = synthesize_defect_image(
        base_image, I0=I0, defect_mask=synthetic_mask, delta=delta,
        noise_gain=noise_gain, blur_sigma=blur_sigma, rng=rng,
    )

    save_gray(base_image, OUT_DIR / "synth-xray-clean.png")
    save_gray(defect_image, OUT_DIR / "synth-xray-defect-real.png")
    save_gray(synthetic_image, OUT_DIR / "synth-xray-defect-synthetic.png")

    print(f"series: {series_dir.name}")
    print(f"defect image (delta fit from): {defect_path.name}")
    print(f"base image (noise/blur fit from, synthesized onto): {base_path.name}")
    print(f"fitted I0: {I0:.2f}")
    print(f"fitted attenuation-depth delta: {delta:.4f}")
    print(f"fitted noise model: variance = {a:.3f} * mean + {b:.3f}")
    print(f"fitted blur sigma: {blur_sigma:.3f} px")


if __name__ == "__main__":
    main()
