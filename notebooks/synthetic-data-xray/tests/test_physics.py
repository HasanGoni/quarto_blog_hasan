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
