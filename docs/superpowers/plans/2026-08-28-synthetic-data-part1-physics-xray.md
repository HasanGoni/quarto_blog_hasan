# Synthetic Data Series — Part 1 (Physics-Based X-ray Simulation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a physics-based X-ray defect synthesis pipeline — calibrated from real GDXray images since no imaging-equipment access exists — that turns a real clean X-ray image plus a mask into a synthetic defect image with matching ground-truth mask, and publish it as Part 1 of the new "Synthetic Data" blog series.

**Architecture:** All numerical work lives in a small, pure-function Python package (`notebooks/synthetic-data-xray/src/synth_xray/`) with no framework dependencies beyond numpy/scipy/scikit-image — each module is independently unit-testable without network access or real images. A single `data.py` module handles the one network-dependent step (downloading real GDXray images). A pipeline module composes the pure functions into the end-to-end synthesis used to generate the blog post's real output.

**Tech Stack:** Python 3.12, `uv` for environment management, numpy, scipy, scikit-image, matplotlib (including `plt.xkcd()` for the whiteboard-style diagram), Pillow, pytest, requests.

**Spec:** `docs/superpowers/specs/2026-08-28-synthetic-data-series-design.md`

## Global Constraints

- Data anchor is GDXray (Welds group primary — confirmed pixel-level ground truth for porosity defects; Castings group optional/secondary), downloaded from the current maintained source: `https://github.com/computervision-xray-testing/GDXray` (Dropbox-hosted zips), not the stale `dmery.sitios.ing.uc.cl` academic mirror referenced in older scripts.
- No equipment access — every physical parameter (attenuation, noise, blur) is fit from real GDXray pixel statistics, never hardcoded from a spec sheet.
- Work in **attenuation-depth units** (μ·t, i.e. `-ln(I/I0)`), never claim to separate μ and t — a single 2D projection cannot do that without spectral or multi-view data.
- New subproject at `notebooks/synthetic-data-xray/`, `uv`-managed, `pyproject.toml` with `[tool.uv] package = false`, `.venv` gitignored — same pattern as `paper-rau-sam2`/`paper-hpma-sam3`.
- GIF/diagram deliverables: whiteboard-style diagram must be reproducible code (matplotlib `plt.xkcd()`), not a manually-drawn one-off asset — keeps it consistent and regenerable for Parts 2-5.
- Always confirm with the user before running `quarto publish` (live deploy).

---

## File Structure

```
notebooks/synthetic-data-xray/
├── pyproject.toml
├── .gitignore                          # .venv/, .data_cache/
├── src/synth_xray/
│   ├── __init__.py
│   ├── physics.py                      # Beer-Lambert forward/inverse, blur, noise
│   ├── calibration.py                  # fit attenuation depth, noise model, blur sigma from real images
│   ├── groundtruth.py                  # GDXray bbox parsing, Otsu mask refinement
│   ├── defect_edit.py                  # crack-mask generation, attenuation-depth editing
│   ├── pipeline.py                     # end-to-end synthesize_defect_image()
│   └── data.py                         # GDXray download/cache/loading
├── tests/
│   ├── test_physics.py
│   ├── test_calibration.py
│   ├── test_groundtruth.py
│   ├── test_defect_edit.py
│   └── test_pipeline.py
├── generate_examples.py                # produces blog post's before/after PNGs
├── generate_diagram.py                 # produces the xkcd-style whiteboard diagram PNG
└── .data_cache/                        # gitignored, downloaded GDXray zips extract here
```

```
posts/series/synthetic-data/
├── index.qmd
├── 01-physics-based-xray-simulation.qmd
└── images/
    ├── synth-xray-eli5-sketch.png
    ├── synth-xray-clean.png
    ├── synth-xray-defect-real.png
    └── synth-xray-defect-synthetic.png
```

---

### Task 1: Scaffold the `synthetic-data-xray` uv project

**Files:**
- Create: `notebooks/synthetic-data-xray/pyproject.toml`
- Create: `notebooks/synthetic-data-xray/src/synth_xray/__init__.py`
- Create: `notebooks/synthetic-data-xray/tests/test_environment.py`
- Modify: `.gitignore:1` (repo root — add the new venv/cache paths)

**Interfaces:**
- Produces: an importable `synth_xray` package that every later task's modules live in.

- [ ] **Step 1: Create the project directory and `pyproject.toml`**

```toml
[project]
name = "synth-xray"
version = "0.1.0"
description = "Physics-based and generative synthetic X-ray defect image generation (Synthetic Data blog series)"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.0",
    "scipy>=1.13",
    "scikit-image>=0.24",
    "matplotlib>=3.9",
    "pillow>=10.0",
    "requests>=2.32",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[tool.uv]
package = false

[tool.pytest.ini_options]
pythonpath = ["src"]
```

- [ ] **Step 2: Create the package `__init__.py`**

```python
"""Physics-based and generative synthetic X-ray defect image generation."""
```

- [ ] **Step 3: Add gitignore entries**

Append to the repo-root `.gitignore`:

```
notebooks/synthetic-data-xray/.venv/
notebooks/synthetic-data-xray/.data_cache/
```

- [ ] **Step 4: Sync the environment**

Run: `cd notebooks/synthetic-data-xray && uv sync --group dev`
Expected: creates `.venv/` and `uv.lock` with no errors.

- [ ] **Step 5: Write and run the environment sanity test**

```python
# tests/test_environment.py
import numpy as np
import scipy
import skimage
import synth_xray


def test_dependencies_import():
    assert np.__version__
    assert scipy.__version__
    assert skimage.__version__


def test_package_importable():
    assert synth_xray is not None
```

Run: `cd notebooks/synthetic-data-xray && uv run pytest tests/test_environment.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add notebooks/synthetic-data-xray/pyproject.toml notebooks/synthetic-data-xray/src notebooks/synthetic-data-xray/tests/test_environment.py notebooks/synthetic-data-xray/uv.lock .gitignore
git commit -m "Scaffold synthetic-data-xray uv project"
```

---

### Task 2: GDXray ground-truth parsing and mask refinement

**Files:**
- Create: `notebooks/synthetic-data-xray/src/synth_xray/groundtruth.py`
- Test: `notebooks/synthetic-data-xray/tests/test_groundtruth.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure numpy/skimage).
- Produces:
  - `parse_gdxray_bboxes(txt_path: pathlib.Path) -> list[tuple[int, int, int, int]]` — each tuple is `(row_min, col_min, row_max, col_max)`.
  - `bbox_to_mask(image_shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray` — bool array, `True` inside the box.
  - `refine_mask_otsu(image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray` — bool array, the minority-area Otsu cluster within `roi_mask`.

GDXray ground-truth files are whitespace-separated text, one defect per row, with the format `<image_id> <row_min> <col_min> <row_max> <col_max>` (1-indexed in the original files; this parser converts to 0-indexed).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_groundtruth.py
import numpy as np
import pytest
from synth_xray.groundtruth import parse_gdxray_bboxes, bbox_to_mask, refine_mask_otsu


def test_parse_gdxray_bboxes(tmp_path):
    gt_file = tmp_path / "ground_truth_C0001.txt"
    gt_file.write_text("1 10 10 20 20\n1 30 5 45 15\n2 1 1 5 5\n")
    boxes = parse_gdxray_bboxes(gt_file)
    assert boxes == [(9, 9, 19, 19), (29, 4, 44, 14), (0, 0, 4, 4)]


def test_bbox_to_mask_shape_and_bounds():
    mask = bbox_to_mask((50, 50), (9, 9, 19, 19))
    assert mask.shape == (50, 50)
    assert mask.dtype == bool
    assert mask[9:20, 9:20].all()
    assert mask.sum() == 11 * 11
    assert not mask[8, 9]
    assert not mask[9, 8]


def test_refine_mask_otsu_recovers_bright_blob():
    image = np.full((40, 40), 100.0)
    roi = bbox_to_mask((40, 40), (10, 10, 29, 29))
    image[15:25, 15:25] = 220.0  # small bright defect inside the ROI
    refined = refine_mask_otsu(image, roi)
    assert refined.shape == (40, 40)
    assert refined.dtype == bool
    assert refined[18, 18]
    assert not refined[11, 11]
    assert not refined.any() or refined[roi].sum() < roi.sum()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd notebooks/synthetic-data-xray && uv run pytest tests/test_groundtruth.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` for `synth_xray.groundtruth`.

- [ ] **Step 3: Implement `groundtruth.py`**

```python
# src/synth_xray/groundtruth.py
"""Parsing and refining GDXray defect ground truth."""
import pathlib

import numpy as np
from skimage.filters import threshold_otsu


def parse_gdxray_bboxes(txt_path: pathlib.Path) -> list[tuple[int, int, int, int]]:
    """Parse a GDXray ground-truth file into 0-indexed (row_min, col_min, row_max, col_max) boxes."""
    boxes = []
    for line in pathlib.Path(txt_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        _, row_min, col_min, row_max, col_max = (int(float(v)) for v in line.split())
        boxes.append((row_min - 1, col_min - 1, row_max - 1, col_max - 1))
    return boxes


def bbox_to_mask(image_shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Convert a (row_min, col_min, row_max, col_max) box into a boolean mask, inclusive bounds."""
    row_min, col_min, row_max, col_max = bbox
    mask = np.zeros(image_shape, dtype=bool)
    mask[row_min:row_max + 1, col_min:col_max + 1] = True
    return mask


def refine_mask_otsu(image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Split the ROI into two Otsu clusters and return the minority-area one as the defect mask.

    GDXray's shipped ground truth is a bounding box, not a pixel mask; defects are
    the minority-area intensity cluster within that box in every case observed in
    the Welds/Castings groups (a defect is a small region, not most of the box).
    """
    roi_values = image[roi_mask]
    threshold = threshold_otsu(roi_values)
    high_cluster = roi_mask & (image >= threshold)
    low_cluster = roi_mask & (image < threshold)
    return high_cluster if high_cluster.sum() <= low_cluster.sum() else low_cluster
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd notebooks/synthetic-data-xray && uv run pytest tests/test_groundtruth.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add notebooks/synthetic-data-xray/src/synth_xray/groundtruth.py notebooks/synthetic-data-xray/tests/test_groundtruth.py
git commit -m "Add GDXray ground-truth parsing and Otsu mask refinement"
```

---

### Task 3: Beer-Lambert forward/inverse model, blur, and noise

**Files:**
- Create: `notebooks/synthetic-data-xray/src/synth_xray/physics.py`
- Test: `notebooks/synthetic-data-xray/tests/test_physics.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `render_xray(attenuation_depth: np.ndarray, I0: float) -> np.ndarray`
  - `invert_to_attenuation_depth(image: np.ndarray, I0: float, eps: float = 1e-6) -> np.ndarray`
  - `apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray`
  - `apply_photon_noise(image: np.ndarray, gain: float, rng: np.random.Generator | None = None) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_physics.py
import numpy as np
from synth_xray.physics import (
    render_xray,
    invert_to_attenuation_depth,
    apply_gaussian_blur,
    apply_photon_noise,
)


def test_render_xray_zero_depth_returns_i0():
    depth = np.zeros((10, 10))
    image = render_xray(depth, I0=1000.0)
    assert np.allclose(image, 1000.0)


def test_render_xray_monotonic_decrease_with_depth():
    depth = np.array([[0.0, 0.5, 1.0, 2.0]])
    image = render_xray(depth, I0=1000.0)
    assert np.all(np.diff(image[0]) < 0)


def test_render_and_invert_roundtrip():
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.0, 3.0, size=(20, 20))
    image = render_xray(depth, I0=500.0)
    recovered = invert_to_attenuation_depth(image, I0=500.0)
    assert np.allclose(recovered, depth, atol=1e-6)


def test_apply_gaussian_blur_preserves_flat_image():
    image = np.full((15, 15), 42.0)
    blurred = apply_gaussian_blur(image, sigma=2.0)
    assert np.allclose(blurred, 42.0, atol=1e-6)


def test_apply_gaussian_blur_smooths_a_step_edge():
    image = np.zeros((20, 20))
    image[:, 10:] = 100.0
    blurred = apply_gaussian_blur(image, sigma=2.0)
    assert 0.0 < blurred[10, 10] < 100.0


def test_apply_photon_noise_variance_scales_with_signal():
    rng = np.random.default_rng(0)
    dim_signal = np.full((256, 256), 50.0)
    bright_signal = np.full((256, 256), 500.0)
    dim_noisy = apply_photon_noise(dim_signal, gain=1.0, rng=rng)
    bright_noisy = apply_photon_noise(bright_signal, gain=1.0, rng=rng)
    assert dim_noisy.var() < bright_noisy.var()
    assert np.isclose(dim_noisy.mean(), 50.0, rtol=0.05)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd notebooks/synthetic-data-xray && uv run pytest tests/test_physics.py -v`
Expected: FAIL with `ModuleNotFoundError` for `synth_xray.physics`.

- [ ] **Step 3: Implement `physics.py`**

```python
# src/synth_xray/physics.py
"""Beer-Lambert forward/inverse rendering, blur, and photon-noise models."""
import numpy as np
from scipy.ndimage import gaussian_filter


def render_xray(attenuation_depth: np.ndarray, I0: float) -> np.ndarray:
    """Forward-render an attenuation-depth map (mu*t, in attenuation lengths) to intensity."""
    return I0 * np.exp(-attenuation_depth)


def invert_to_attenuation_depth(image: np.ndarray, I0: float, eps: float = 1e-6) -> np.ndarray:
    """Invert Beer-Lambert to recover the attenuation-depth map implied by a real image."""
    ratio = np.clip(image / I0, eps, None)
    return -np.log(ratio)


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply detector/geometric-unsharpness blur as a Gaussian filter."""
    if sigma <= 0:
        return image
    return gaussian_filter(image, sigma=sigma)


def apply_photon_noise(
    image: np.ndarray, gain: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Apply photon-limited (Poisson) noise: variance scales with signal, per `gain` electrons/count."""
    rng = rng if rng is not None else np.random.default_rng()
    photon_counts = np.clip(image, 0, None) / gain
    noisy_counts = rng.poisson(photon_counts).astype(np.float64)
    return noisy_counts * gain
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd notebooks/synthetic-data-xray && uv run pytest tests/test_physics.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add notebooks/synthetic-data-xray/src/synth_xray/physics.py notebooks/synthetic-data-xray/tests/test_physics.py
git commit -m "Add Beer-Lambert forward/inverse model, blur, and photon-noise functions"
```

---

### Task 4: Calibrate attenuation depth, noise model, and blur from real images

**Files:**
- Create: `notebooks/synthetic-data-xray/src/synth_xray/calibration.py`
- Test: `notebooks/synthetic-data-xray/tests/test_calibration.py`

**Interfaces:**
- Consumes: `render_xray`, `invert_to_attenuation_depth`, `apply_gaussian_blur`, `apply_photon_noise` from `synth_xray.physics` (Task 3).
- Produces:
  - `estimate_I0(image: np.ndarray, percentile: float = 99.0) -> float`
  - `fit_defect_attenuation_depth(image: np.ndarray, defect_mask: np.ndarray, background_mask: np.ndarray, I0: float) -> float`
  - `fit_noise_model(image: np.ndarray, patch_masks: list[np.ndarray]) -> tuple[float, float]` — returns `(a, b)` s.t. `variance ≈ a * mean + b`.
  - `fit_blur_sigma(edge_profile: np.ndarray) -> float` — takes a 1D intensity profile crossing an edge, returns the Gaussian sigma (in pixels) of its line-spread function.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_calibration.py
import numpy as np
from synth_xray.physics import render_xray, apply_gaussian_blur
from synth_xray.calibration import (
    estimate_I0,
    fit_defect_attenuation_depth,
    fit_noise_model,
    fit_blur_sigma,
)


def test_estimate_i0_uses_bright_percentile():
    image = np.full((50, 50), 200.0)
    image[0, 0] = 1000.0  # single unattenuated pixel, e.g. beam edge
    i0 = estimate_I0(image, percentile=99.9)
    assert i0 > 200.0


def test_fit_defect_attenuation_depth_recovers_known_void_depth():
    background = np.zeros((30, 30))
    background_mask = np.zeros((30, 30), dtype=bool)
    background_mask[:10, :] = True
    defect_mask = np.zeros((30, 30), dtype=bool)
    defect_mask[20:, :] = True

    true_depth = np.full((30, 30), 2.0)
    true_depth[defect_mask] = 0.8  # void: less material, lower attenuation depth
    image = render_xray(true_depth, I0=1000.0)

    fitted_delta = fit_defect_attenuation_depth(image, defect_mask, background_mask, I0=1000.0)
    assert np.isclose(fitted_delta, 2.0 - 0.8, atol=1e-3)


def test_fit_noise_model_recovers_positive_slope():
    rng = np.random.default_rng(0)
    means = [20.0, 100.0, 400.0]
    patches = []
    image = np.zeros((90, 30))
    for i, m in enumerate(means):
        photon_counts = rng.poisson(m, size=(30, 30)).astype(float)
        image[i * 30:(i + 1) * 30, :] = photon_counts
        mask = np.zeros((90, 30), dtype=bool)
        mask[i * 30:(i + 1) * 30, :] = True
        patches.append(mask)

    a, b = fit_noise_model(image, patches)
    assert a > 0.5  # Poisson noise: variance ~= mean, slope near 1


def test_fit_blur_sigma_recovers_known_sigma():
    profile = np.zeros(100)
    profile[50:] = 100.0
    blurred = apply_gaussian_blur(profile.reshape(1, -1), sigma=3.0).ravel()
    sigma = fit_blur_sigma(blurred)
    assert 2.0 < sigma < 4.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd notebooks/synthetic-data-xray && uv run pytest tests/test_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError` for `synth_xray.calibration`.

- [ ] **Step 3: Implement `calibration.py`**

```python
# src/synth_xray/calibration.py
"""Fitting physics-model parameters from real GDXray image statistics."""
import numpy as np
from scipy.optimize import curve_fit

from synth_xray.physics import invert_to_attenuation_depth


def estimate_I0(image: np.ndarray, percentile: float = 99.0) -> float:
    """Estimate the unattenuated beam intensity as a high percentile of the image."""
    return float(np.percentile(image, percentile))


def fit_defect_attenuation_depth(
    image: np.ndarray, defect_mask: np.ndarray, background_mask: np.ndarray, I0: float
) -> float:
    """Fit the attenuation-depth difference (background - defect) a real defect represents.

    Positive for a void/crack (material missing -> less attenuation -> brighter pixels).
    """
    depth = invert_to_attenuation_depth(image, I0)
    background_depth = float(np.median(depth[background_mask]))
    defect_depth = float(np.median(depth[defect_mask]))
    return background_depth - defect_depth


def fit_noise_model(image: np.ndarray, patch_masks: list[np.ndarray]) -> tuple[float, float]:
    """Fit variance = a*mean + b (photon-transfer curve) across flat patches."""
    means = np.array([image[mask].mean() for mask in patch_masks])
    variances = np.array([image[mask].var() for mask in patch_masks])
    a, b = np.polyfit(means, variances, deg=1)
    return float(a), float(b)


def _gaussian_cdf_edge(x: np.ndarray, amplitude: float, center: float, sigma: float, offset: float) -> np.ndarray:
    from scipy.special import erf
    return offset + amplitude * 0.5 * (1 + erf((x - center) / (sigma * np.sqrt(2))))


def fit_blur_sigma(edge_profile: np.ndarray) -> float:
    """Fit a Gaussian-blurred step edge to a 1D intensity profile; return sigma in pixels."""
    x = np.arange(edge_profile.size, dtype=float)
    amplitude_guess = edge_profile[-1] - edge_profile[0]
    center_guess = x[np.argmin(np.abs(edge_profile - edge_profile.mean()))]
    p0 = [amplitude_guess, center_guess, 2.0, edge_profile[0]]
    popt, _ = curve_fit(_gaussian_cdf_edge, x, edge_profile, p0=p0, maxfev=5000)
    return float(abs(popt[2]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd notebooks/synthetic-data-xray && uv run pytest tests/test_calibration.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add notebooks/synthetic-data-xray/src/synth_xray/calibration.py notebooks/synthetic-data-xray/tests/test_calibration.py
git commit -m "Add attenuation depth, noise model, and blur calibration from real images"
```

---

### Task 5: Defect editing — crack-mask generation and attenuation-depth edits

**Files:**
- Create: `notebooks/synthetic-data-xray/src/synth_xray/defect_edit.py`
- Test: `notebooks/synthetic-data-xray/tests/test_defect_edit.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure numpy).
- Produces:
  - `generate_crack_mask(image_shape: tuple[int, int], start: tuple[int, int], length: int, thickness: int = 1, rng: np.random.Generator | None = None) -> np.ndarray`
  - `edit_attenuation_depth(attenuation_depth: np.ndarray, mask: np.ndarray, delta: float) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_defect_edit.py
import numpy as np
from synth_xray.defect_edit import generate_crack_mask, edit_attenuation_depth


def test_generate_crack_mask_shape_and_length():
    rng = np.random.default_rng(0)
    mask = generate_crack_mask((100, 100), start=(50, 50), length=30, thickness=1, rng=rng)
    assert mask.shape == (100, 100)
    assert mask.dtype == bool
    assert mask[50, 50]
    assert mask.sum() >= 30
    assert mask.sum() < 100 * 100 * 0.05


def test_generate_crack_mask_thickness_widens_mask():
    rng = np.random.default_rng(0)
    thin = generate_crack_mask((100, 100), start=(50, 50), length=30, thickness=1, rng=rng)
    rng = np.random.default_rng(0)
    thick = generate_crack_mask((100, 100), start=(50, 50), length=30, thickness=3, rng=rng)
    assert thick.sum() > thin.sum()


def test_edit_attenuation_depth_reduces_depth_under_mask_only():
    depth = np.full((10, 10), 2.0)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:6, 3:6] = True
    edited = edit_attenuation_depth(depth, mask, delta=1.2)
    assert np.allclose(edited[mask], 0.8)
    assert np.allclose(edited[~mask], 2.0)


def test_edit_attenuation_depth_clips_at_zero():
    depth = np.full((5, 5), 0.5)
    mask = np.ones((5, 5), dtype=bool)
    edited = edit_attenuation_depth(depth, mask, delta=10.0)
    assert np.all(edited >= 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd notebooks/synthetic-data-xray && uv run pytest tests/test_defect_edit.py -v`
Expected: FAIL with `ModuleNotFoundError` for `synth_xray.defect_edit`.

- [ ] **Step 3: Implement `defect_edit.py`**

```python
# src/synth_xray/defect_edit.py
"""Synthetic defect shape generation and attenuation-depth map editing."""
import numpy as np
from skimage.draw import disk


def generate_crack_mask(
    image_shape: tuple[int, int],
    start: tuple[int, int],
    length: int,
    thickness: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a thin random-walk crack-shaped mask starting at `start`."""
    rng = rng if rng is not None else np.random.default_rng()
    mask = np.zeros(image_shape, dtype=bool)
    row, col = start
    radius = max(thickness - 1, 0)
    for _ in range(length):
        row = int(np.clip(row + rng.integers(-1, 2), 0, image_shape[0] - 1))
        col = int(np.clip(col + rng.integers(-1, 2), 0, image_shape[1] - 1))
        if radius > 0:
            rr, cc = disk((row, col), radius + 1, shape=image_shape)
            mask[rr, cc] = True
        else:
            mask[row, col] = True
    return mask


def edit_attenuation_depth(attenuation_depth: np.ndarray, mask: np.ndarray, delta: float) -> np.ndarray:
    """Reduce attenuation depth under `mask` by `delta` (a void/crack: material is missing)."""
    edited = attenuation_depth.copy()
    edited[mask] = np.clip(edited[mask] - delta, 0.0, None)
    return edited
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd notebooks/synthetic-data-xray && uv run pytest tests/test_defect_edit.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add notebooks/synthetic-data-xray/src/synth_xray/defect_edit.py notebooks/synthetic-data-xray/tests/test_defect_edit.py
git commit -m "Add crack-mask generation and attenuation-depth editing"
```

---

### Task 6: End-to-end synthesis pipeline

**Files:**
- Create: `notebooks/synthetic-data-xray/src/synth_xray/pipeline.py`
- Test: `notebooks/synthetic-data-xray/tests/test_pipeline.py`

**Interfaces:**
- Consumes:
  - `render_xray`, `invert_to_attenuation_depth`, `apply_gaussian_blur`, `apply_photon_noise` from `synth_xray.physics` (Task 3)
  - `edit_attenuation_depth` from `synth_xray.defect_edit` (Task 5)
- Produces: `synthesize_defect_image(clean_image: np.ndarray, I0: float, defect_mask: np.ndarray, delta: float, noise_gain: float, blur_sigma: float, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]` — returns `(synthetic_image, defect_mask)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
import numpy as np
from synth_xray.pipeline import synthesize_defect_image


def test_synthesize_defect_image_output_shape_and_range():
    rng = np.random.default_rng(0)
    clean = np.full((60, 60), 400.0)
    mask = np.zeros((60, 60), dtype=bool)
    mask[25:35, 25:35] = True

    synthetic, returned_mask = synthesize_defect_image(
        clean, I0=1000.0, defect_mask=mask, delta=1.0, noise_gain=1.0, blur_sigma=1.5, rng=rng
    )

    assert synthetic.shape == clean.shape
    assert np.array_equal(returned_mask, mask)
    assert np.all(synthetic >= 0.0)


def test_synthesize_defect_image_is_brighter_under_mask_on_average():
    rng = np.random.default_rng(0)
    clean = np.full((60, 60), 400.0)
    mask = np.zeros((60, 60), dtype=bool)
    mask[25:35, 25:35] = True

    synthetic, _ = synthesize_defect_image(
        clean, I0=1000.0, defect_mask=mask, delta=1.0, noise_gain=1.0, blur_sigma=0.0, rng=rng
    )

    assert synthetic[mask].mean() > synthetic[~mask].mean()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd notebooks/synthetic-data-xray && uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError` for `synth_xray.pipeline`.

- [ ] **Step 3: Implement `pipeline.py`**

```python
# src/synth_xray/pipeline.py
"""End-to-end physics-based synthetic defect image generation."""
import numpy as np

from synth_xray.physics import (
    render_xray,
    invert_to_attenuation_depth,
    apply_gaussian_blur,
    apply_photon_noise,
)
from synth_xray.defect_edit import edit_attenuation_depth


def synthesize_defect_image(
    clean_image: np.ndarray,
    I0: float,
    defect_mask: np.ndarray,
    delta: float,
    noise_gain: float,
    blur_sigma: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert a real clean image to attenuation depth, inject a defect, and re-render.

    Steps: invert -> edit under mask -> forward render -> blur -> photon noise.
    Returns (synthetic_image, defect_mask) so the mask travels with its image as
    free ground truth.
    """
    depth = invert_to_attenuation_depth(clean_image, I0)
    edited_depth = edit_attenuation_depth(depth, defect_mask, delta)
    rendered = render_xray(edited_depth, I0)
    blurred = apply_gaussian_blur(rendered, blur_sigma)
    noisy = apply_photon_noise(blurred, gain=noise_gain, rng=rng)
    return noisy, defect_mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd notebooks/synthetic-data-xray && uv run pytest tests/test_pipeline.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the full test suite**

Run: `cd notebooks/synthetic-data-xray && uv run pytest -v`
Expected: all tests from Tasks 1-6 pass (19 tests).

- [ ] **Step 6: Commit**

```bash
git add notebooks/synthetic-data-xray/src/synth_xray/pipeline.py notebooks/synthetic-data-xray/tests/test_pipeline.py
git commit -m "Add end-to-end physics-based defect synthesis pipeline"
```

---

### Task 7: GDXray data acquisition

**Files:**
- Create: `notebooks/synthetic-data-xray/src/synth_xray/data.py`
- Test: `notebooks/synthetic-data-xray/tests/test_data.py`

**Interfaces:**
- Consumes: `parse_gdxray_bboxes` from `synth_xray.groundtruth` (Task 2, used by `find_groundtruth_file`'s caller in Task 8's example script — not imported directly here).
- Produces:
  - `GDXRAY_URLS: dict[str, str]`
  - `download_and_extract(group: str, dest_dir: pathlib.Path) -> pathlib.Path` — returns the path to the extracted group directory, skips download if already present.
  - `find_series_dirs(group_dir: pathlib.Path) -> list[pathlib.Path]`
  - `find_images_in_series(series_dir: pathlib.Path) -> list[pathlib.Path]`
  - `find_groundtruth_file(series_dir: pathlib.Path) -> pathlib.Path | None`

The real download is 209MB (Welds) and network-dependent; only the extraction/discovery logic is exercised with a real network call, isolated into one explicitly slow test.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data.py
import zipfile
import pytest
from synth_xray.data import (
    GDXRAY_URLS,
    download_and_extract,
    find_series_dirs,
    find_images_in_series,
    find_groundtruth_file,
)


def _make_fake_group_zip(zip_path):
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("Welds/W0001/W0001_0001.png", b"fake-png-bytes")
        zf.writestr("Welds/W0001/W0001_0002.png", b"fake-png-bytes")
        zf.writestr("Welds/W0001/ground_truth_W0001.txt", "1 10 10 20 20\n")
        zf.writestr("Welds/W0002/W0002_0001.png", b"fake-png-bytes")


def test_download_and_extract_skips_existing(tmp_path, monkeypatch):
    dest = tmp_path / "cache"
    dest.mkdir()
    group_dir = dest / "Welds"
    group_dir.mkdir()
    (group_dir / "already_here.txt").write_text("x")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not attempt download when group dir already exists")

    monkeypatch.setattr("synth_xray.data.requests.get", fail_if_called)
    result = download_and_extract("Welds", dest)
    assert result == group_dir


def test_find_series_and_images_and_groundtruth(tmp_path):
    zip_path = tmp_path / "Welds.zip"
    _make_fake_group_zip(zip_path)
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    group_dir = extract_dir / "Welds"

    series_dirs = find_series_dirs(group_dir)
    assert {d.name for d in series_dirs} == {"W0001", "W0002"}

    w0001 = group_dir / "W0001"
    images = find_images_in_series(w0001)
    assert len(images) == 2
    assert all(p.suffix == ".png" for p in images)

    gt = find_groundtruth_file(w0001)
    assert gt is not None
    assert gt.name == "ground_truth_W0001.txt"

    w0002 = group_dir / "W0002"
    assert find_groundtruth_file(w0002) is None


def test_gdxray_urls_are_https_dropbox_links():
    assert "Welds" in GDXRAY_URLS
    assert GDXRAY_URLS["Welds"].startswith("https://www.dropbox.com/")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd notebooks/synthetic-data-xray && uv run pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError` for `synth_xray.data`.

- [ ] **Step 3: Implement `data.py`**

```python
# src/synth_xray/data.py
"""Downloading and locating real GDXray images and ground truth."""
import pathlib
import zipfile

import requests

GDXRAY_URLS = {
    "Welds": "https://www.dropbox.com/scl/fi/im896nbhllnbnol585fsq/Welds.zip?rlkey=u584im2jtrdxzmhrg2lcqtavv&st=sswtvnmd&dl=1",
    "Castings": "https://www.dropbox.com/scl/fi/5e7m31ri5grvnl7kxt8mx/Castings.zip?rlkey=zetfg5g337d2ip265a16rcviw&st=gjub7hr3&dl=1",
}


def download_and_extract(group: str, dest_dir: pathlib.Path) -> pathlib.Path:
    """Download and extract a GDXray group zip into `dest_dir`, skipping if already extracted."""
    dest_dir = pathlib.Path(dest_dir)
    group_dir = dest_dir / group
    if group_dir.exists():
        return group_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{group}.zip"
    response = requests.get(GDXRAY_URLS[group], stream=True, timeout=300)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()
    return group_dir


def find_series_dirs(group_dir: pathlib.Path) -> list[pathlib.Path]:
    """Find every series subdirectory in an extracted GDXray group."""
    return sorted(p for p in pathlib.Path(group_dir).iterdir() if p.is_dir())


def find_images_in_series(series_dir: pathlib.Path) -> list[pathlib.Path]:
    """Find every image file in a series directory."""
    return sorted(pathlib.Path(series_dir).glob("*.png"))


def find_groundtruth_file(series_dir: pathlib.Path) -> pathlib.Path | None:
    """Find the GDXray ground-truth text file in a series directory, if present."""
    matches = sorted(pathlib.Path(series_dir).glob("ground_truth_*.txt"))
    return matches[0] if matches else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd notebooks/synthetic-data-xray && uv run pytest tests/test_data.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the real download once (manual, slow — not part of the default test suite)**

Run:
```bash
cd notebooks/synthetic-data-xray
uv run python -c "
from pathlib import Path
from synth_xray.data import download_and_extract, find_series_dirs
group_dir = download_and_extract('Welds', Path('.data_cache'))
series = find_series_dirs(group_dir)
print(f'{len(series)} series found, first: {series[0].name}')
"
```
Expected: downloads `.data_cache/Welds.zip` (~209MB), extracts it, prints a series count > 0. If the Dropbox link has expired, re-fetch the current link from `https://github.com/computervision-xray-testing/GDXray` and update `GDXRAY_URLS` before re-running.

- [ ] **Step 6: Commit**

```bash
git add notebooks/synthetic-data-xray/src/synth_xray/data.py notebooks/synthetic-data-xray/tests/test_data.py
git commit -m "Add GDXray download, extraction, and discovery utilities"
```

---

### Task 8: Generate real before/after example images for the blog post

**Files:**
- Create: `notebooks/synthetic-data-xray/generate_examples.py`
- Modify (create dir): `posts/series/synthetic-data/images/`

**Interfaces:**
- Consumes: `download_and_extract`, `find_series_dirs`, `find_images_in_series`, `find_groundtruth_file` from `synth_xray.data` (Task 7); `parse_gdxray_bboxes`, `bbox_to_mask`, `refine_mask_otsu` from `synth_xray.groundtruth` (Task 2); `estimate_I0`, `fit_defect_attenuation_depth`, `fit_noise_model`, `fit_blur_sigma` from `synth_xray.calibration` (Task 4); `generate_crack_mask` from `synth_xray.defect_edit` (Task 5); `synthesize_defect_image` from `synth_xray.pipeline` (Task 6).
- Produces (files, not importable symbols): `posts/series/synthetic-data/images/synth-xray-clean.png`, `synth-xray-defect-real.png`, `synth-xray-defect-synthetic.png`, plus a printed calibration report (fitted I0, attenuation delta, noise (a, b), blur sigma) captured into the blog post text.

This is a script, not a unit-tested module — its job is producing real captured output, which is itself the verification (visual + printed numbers), same as the RAU/HPMA posts' example-generation scripts.

- [ ] **Step 1: Write `generate_examples.py`**

```python
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

    flat_h, flat_w = defect_image.shape[0] // 4, defect_image.shape[1] // 4
    patch_masks = []
    for r in range(2):
        for c in range(2):
            m = np.zeros(defect_image.shape, dtype=bool)
            m[r * flat_h:(r + 1) * flat_h, c * flat_w:(c + 1) * flat_w] = True
            m &= ~roi_mask
            patch_masks.append(m)
    a, b = fit_noise_model(defect_image, patch_masks)

    row_c, col_c = box[0] + (box[2] - box[0]) // 2, box[1]
    edge_profile = defect_image[row_c, max(col_c - 15, 0):col_c + 15]
    blur_sigma = fit_blur_sigma(edge_profile) if edge_profile.size >= 10 else 1.5

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
```

- [ ] **Step 2: Run it**

Run: `cd notebooks/synthetic-data-xray && uv run python generate_examples.py`
Expected: prints the calibration report and writes 3 PNGs to `posts/series/synthetic-data/images/`. Capture the printed numbers — they go verbatim into the blog post's "real captured output" section.

- [ ] **Step 3: Visually inspect the output**

Open the 3 generated PNGs and confirm: the synthetic image looks like a plausible weld X-ray with a crack, not obviously broken (e.g. not all-black, not all-white, crack visible but not absurdly bright/dark). If it looks wrong, adjust `delta`/`blur_sigma`/`noise_gain` scaling in `generate_examples.py` and re-run before proceeding — this is a real judgment call the post needs to be honest about either way.

- [ ] **Step 4: Commit**

```bash
git add notebooks/synthetic-data-xray/generate_examples.py posts/series/synthetic-data/images/synth-xray-clean.png posts/series/synthetic-data/images/synth-xray-defect-real.png posts/series/synthetic-data/images/synth-xray-defect-synthetic.png
git commit -m "Generate real before/after example images for Synthetic Data Part 1"
```

---

### Task 9: Generate the whiteboard-style ELI5 diagram

**Files:**
- Create: `notebooks/synthetic-data-xray/generate_diagram.py`

**Interfaces:**
- Consumes: nothing (matplotlib only).
- Produces (file): `posts/series/synthetic-data/images/synth-xray-eli5-sketch.png`.

- [ ] **Step 1: Write `generate_diagram.py`**

```python
# generate_diagram.py
"""Generate the whiteboard-style (matplotlib xkcd) ELI5 diagram for Part 1."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_PATH = Path(__file__).parents[2] / "posts" / "series" / "synthetic-data" / "images" / "synth-xray-eli5-sketch.png"

STEPS = [
    "Real clean\nX-ray image",
    "Invert physics\n(Beer-Lambert)",
    "Thickness map",
    "Edit inside\na mask",
    "Re-render\n+ noise + blur",
    "Synthetic image\n+ free mask",
]


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.set_xlim(0, len(STEPS))
        ax.set_ylim(0, 2)
        ax.axis("off")

        fig.text(0.5, 0.95, "Physics-Based Synthetic X-ray", fontsize=20, fontweight="bold", ha="center")

        box_w, box_h = 0.8, 0.9
        for i, label in enumerate(STEPS):
            x = i + 0.5
            box = FancyBboxPatch(
                (x - box_w / 2, 1.0 - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.05,rounding_size=0.08",
                linewidth=2, edgecolor="black", facecolor="white",
            )
            ax.add_patch(box)
            ax.text(x, 1.0, label, ha="center", va="center", fontsize=10)
            if i < len(STEPS) - 1:
                arrow = FancyArrowPatch(
                    (x + box_w / 2, 1.0), (x + 1 - box_w / 2, 1.0),
                    arrowstyle="-|>", mutation_scale=20, linewidth=2, color="black",
                )
                ax.add_patch(arrow)

        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(OUT_PATH, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd notebooks/synthetic-data-xray && uv run python generate_diagram.py`
Expected: writes `posts/series/synthetic-data/images/synth-xray-eli5-sketch.png`.

- [ ] **Step 3: Visually inspect**

Open the PNG, confirm it reads clearly at blog-post display width (boxes and labels legible, arrows connect left-to-right in the right order).

- [ ] **Step 4: Commit**

```bash
git add notebooks/synthetic-data-xray/generate_diagram.py posts/series/synthetic-data/images/synth-xray-eli5-sketch.png
git commit -m "Generate whiteboard-style ELI5 diagram for Synthetic Data Part 1"
```

---

### Task 10: Write the series index and Part 1 post, wire into the site

**Files:**
- Create: `posts/series/synthetic-data/index.qmd`
- Create: `posts/series/synthetic-data/01-physics-based-xray-simulation.qmd`
- Modify: `_quarto.yml` (navbar Series dropdown + sidebar contents)
- Modify: `README.md` (Series list)

**Interfaces:**
- Consumes: the printed calibration numbers from Task 8's run, and the 4 images from Tasks 8-9.

- [ ] **Step 1: Write `posts/series/synthetic-data/index.qmd`**

```markdown
---
title: "Synthetic Data Series"
description: "Generating synthetic X-ray defect images for industrial inspection when real defective examples are scarce -- physics-based simulation, generative fine-tuning, GANs, and the classic cut-paste baseline, measured against each other."
author: "Hasan Goni"
date: last-modified
categories: [series, computer-vision, synthetic-data, X-ray, semiconductor]
image: images/synth-xray-eli5-sketch.png
toc: true
---

## Series Overview

Real defect examples are chronically scarce in industrial X-ray inspection -- especially for
rare defect classes like voids, cracks, and bent leads. The usual manual fallback is
cut-and-paste: crop a real defect, paste it into a good image. This series explores whether
there's something better, working through the space of synthetic-defect-generation methods
from first-principles physics to learned generative models, using the real, public
[GDXray](https://github.com/computervision-xray-testing/GDXray) dataset throughout.

Every post ends with an honest limitation, not just a win -- same ethos as the
[Local-AI Defect Inspection](../local-ai-defect-inspection/index.qmd) series, which this one
complements: that series evaluates and fine-tunes models on real defect data; this one is about
generating more of it.

### Posts in this series

1. **Part 1** -- [Physics-Based X-ray Simulation](01-physics-based-xray-simulation.qmd). No
   imaging-equipment access, so every physical parameter is fit directly from real GDXray pixel
   statistics: an attenuation-depth calibration, a photon-noise model, a blur kernel -- then a
   real clean image is inverted, edited inside a mask, and re-rendered into a synthetic defect
   image with its ground-truth mask for free.
2. **Part 2** -- Mask-conditioned diffusion fine-tuning. Coming soon.
3. **Part 3** -- GAN-based synthesis. Coming soon.
4. **Part 4** -- Cut-paste + Poisson blending baseline. Coming soon.
5. **Part 5** -- Compare everything. Coming soon.
```

- [ ] **Step 2: Write `posts/series/synthetic-data/01-physics-based-xray-simulation.qmd`**

```markdown
---
title: "Physics-Based X-ray Simulation"
description: "No access to the X-ray equipment that shot our images? Invert the physics on a real image instead of simulating from a CAD model that doesn't exist."
author: "Hasan Goni"
date: last-modified
categories: [computer-vision, synthetic-data, X-ray, physics]
image: images/synth-xray-eli5-sketch.png
toc: true
series:
  name: "Synthetic Data Series"
  number: 1
---

:::{.callout-tip}
This post is part of the [Synthetic Data Series](index.qmd) series.
:::

## The problem

<!-- Motivation: scarce real defect examples, manual cut-paste workflow, why physics first. -->

## Explain it like I'm five

![Physics-based synthesis, sketched out.](images/synth-xray-eli5-sketch.png)

<!-- ELI5 narrative matching the diagram's 6 steps. -->

## Explain it like I'm a data scientist

<!-- Beer-Lambert law, why mu and t aren't separable from one projection, why we work in
     attenuation-depth (mu*t) units instead. -->

## Jargon buster

| Term | Plain-English meaning |
|---|---|
| Attenuation depth (mu·t) | How much an X-ray beam is dimmed passing through material -- the only thing a single 2D X-ray image actually tells you, since attenuation coefficient (mu) and thickness (t) can't be separated without spectral or multi-view data |
| Beer-Lambert law | The physics equation for that dimming: I = I0 * exp(-mu*t) |
| Photon-transfer curve | Plotting noise variance against signal mean to characterize a photon-limited (Poisson) sensor |
| Line-spread function | How a sharp edge gets smeared out by geometric and detector blur -- fitting a Gaussian to it recovers the blur width |
| GDXray | Public X-ray dataset of castings, welds, baggage, and more, with defect ground truth |

## Architecture

<!-- Mermaid diagram traced from the actual pipeline module structure (physics -> calibration
     -> defect_edit -> pipeline), matching the RAU/HPMA post convention. -->

## Calibrating from real images

<!-- Paste the printed calibration report from Task 8's generate_examples.py run:
     fitted I0, attenuation-depth delta, noise model (a, b), blur sigma -- with the real
     GDXray series name it came from. -->

## Real code

<!-- Embed or link the synth_xray package modules: physics.py, calibration.py,
     defect_edit.py, pipeline.py. -->

## Results

::: {layout-ncol=3}
![Real clean image](images/synth-xray-clean.png)

![Real defect image](images/synth-xray-defect-real.png)

![Synthetic defect image](images/synth-xray-defect-synthetic.png)
:::

<!-- Honest assessment: does the synthetic image actually look plausible? Where does the
     physics-only approach visibly fall short -- this is the baseline Part 2's diffusion
     fine-tune needs to beat. -->

## What's next

Part 2 fine-tunes a mask-conditioned diffusion model on the same real GDXray defect crops --
does learning the texture from data beat re-rendering it from physics?
```

- [ ] **Step 3: Fill in the `<!-- -->` placeholders with real prose and a real Mermaid diagram**

This step is writing, not code — draft the ELI5/data-scientist explanations, the architecture Mermaid diagram (trace it from `synth_xray/pipeline.py`'s actual function calls), and the honest results assessment once Task 8's images and numbers are in hand. Follow the two-reading-level + jargon-table + Mermaid-diagram structure the RAU/HPMA posts already established.

- [ ] **Step 4: Render the post**

Run: `quarto render posts/series/synthetic-data/index.qmd posts/series/synthetic-data/01-physics-based-xray-simulation.qmd`
Expected: no `ERROR`/`WARN` lines in the output.

- [ ] **Step 5: Wire into `_quarto.yml`**

Add a "Synthetic Data" entry to `website.navbar.left`'s Series dropdown (pointing at `posts/series/synthetic-data/index.qmd`) and a new `section:` block under `website.sidebar.contents` listing `index.qmd` and `01-physics-based-xray-simulation.qmd`, matching the existing entries' format exactly.

- [ ] **Step 6: Add to `README.md`'s Series list**

Add a `**Synthetic Data Series** — generating synthetic X-ray defect images for industrial inspection when real examples are scarce (physics, diffusion, GANs, cut-paste, compared)` line alongside the existing series bullets.

- [ ] **Step 7: Render the full site**

Run: `quarto render`
Expected: no `ERROR`/`WARN` lines.

- [ ] **Step 8: Commit**

```bash
git add posts/series/synthetic-data/index.qmd posts/series/synthetic-data/01-physics-based-xray-simulation.qmd _quarto.yml README.md
git commit -m "Add Synthetic Data series index and Part 1 (physics-based X-ray simulation)"
```

- [ ] **Step 9: Confirm with the user before publishing**

Show the rendered post locally (`quarto preview` or the rendered HTML). Only run `quarto publish quarto-pub --no-prompt` after the user explicitly confirms — this is a live deploy to the public site.

---

## Self-Review Notes

- **Spec coverage:** Global Constraints (attenuation-depth framing, verified data source, `uv` project pattern, GDXray anchor, deliverable bar) are each implemented in Tasks 1, 3, 4, 7, 8-9. Part 1's spec section (equipment-info list, approximation approach, thickness-map editing) maps directly to Tasks 2-6. Parts 2-5 are explicitly out of scope for this plan (separate plans per the brainstorming decomposition).
- **Type consistency checked:** `attenuation_depth` arrays flow `physics.invert_to_attenuation_depth` → `calibration.fit_defect_attenuation_depth` / `defect_edit.edit_attenuation_depth` → `physics.render_xray` with consistent shapes/dtypes (`np.ndarray`, float) across Tasks 3-6; `synthesize_defect_image`'s signature in Task 6 matches its use in Task 8's `generate_examples.py`.
- **No placeholders in code** — the only `<!-- -->` comments are in Task 10's `.qmd` prose sections, which Step 3 explicitly calls out as a writing step to complete with real content once Task 8's real numbers/images exist (can't be written before that data exists, unlike the Python tasks).
