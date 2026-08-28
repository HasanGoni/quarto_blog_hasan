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
