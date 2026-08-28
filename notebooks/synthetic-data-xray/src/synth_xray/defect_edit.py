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

    # Mark the starting position
    if radius > 0:
        rr, cc = disk((row, col), radius + 1, shape=image_shape)
        mask[rr, cc] = True
    else:
        mask[row, col] = True

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
