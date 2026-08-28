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
