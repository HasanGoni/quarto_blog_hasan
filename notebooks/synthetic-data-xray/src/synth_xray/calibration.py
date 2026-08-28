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
