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
