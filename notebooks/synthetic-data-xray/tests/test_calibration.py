import numpy as np
from synth_xray.physics import render_xray, apply_gaussian_blur
from synth_xray.calibration import (
    estimate_I0,
    fit_defect_attenuation_depth,
    fit_noise_model,
    fit_blur_sigma,
)


def test_estimate_i0_uses_bright_percentile():
    image = np.full((10, 10), 200.0)
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
