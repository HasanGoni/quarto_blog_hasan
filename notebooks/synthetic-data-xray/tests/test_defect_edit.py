import numpy as np
from synth_xray.defect_edit import generate_crack_mask, edit_attenuation_depth


def test_generate_crack_mask_shape_and_length():
    rng = np.random.default_rng(0)
    mask = generate_crack_mask((100, 100), start=(50, 50), length=30, thickness=1, rng=rng)
    assert mask.shape == (100, 100)
    assert mask.dtype == bool
    assert mask[50, 50]
    assert mask.sum() >= 20  # Random walk with seed 0 produces ~22 unique pixels (9 revisits out of 30 steps)
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
