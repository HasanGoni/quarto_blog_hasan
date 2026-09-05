"""Builds the two fixed benchmark pairs used across every post in the series, and saves
the figure that opens Part 1 (reference vs. drifted, side by side).

Scenario A ("stage drift"): pure rigid/affine misalignment -- what phase correlation, ECC,
SIFT+RANSAC, the CNN affine regressor, and LoFTR+homography are all able to fix in principle.

Scenario B ("thermal drift"): the same affine drift *plus* a smooth local elastic warp --
only a deformable (non-rigid) method can fully correct this one; every affine-only method
is expected to leave residual error, which is the actual motivating case for Part 3.
"""
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

from common import (
    SIZE, make_semiconductor_pattern, control_points, affine_matrix,
    elastic_field, warp_affine, warp_points_affine, RNG_SEED,
)

OUT = "data"
import os
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(RNG_SEED)

reference = make_semiconductor_pattern(SIZE, seed=RNG_SEED)
pts_ref = control_points(SIZE)
center = (SIZE / 2, SIZE / 2)

# --- Scenario A: rigid/affine stage drift ---
M_gt = affine_matrix(angle_deg=4.5, tx=9.0, ty=-6.0, scale=1.02, center=center)
moving_affine = warp_affine(reference, M_gt, SIZE)
pts_moving_affine = warp_points_affine(pts_ref, M_gt)

# --- Scenario B: affine drift + smooth elastic (thermal) warp ---
dx, dy, map_x, map_y = elastic_field(SIZE, rng, alpha=6.0, sigma=18.0)
moving_affine_elastic = cv2.remap(moving_affine, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                   borderValue=40)
# ground-truth landmark positions after affine THEN elastic warp: sample the elastic
# field's inverse displacement at each affine-warped point (small-field approx: add local dx,dy)
pts_moving_elastic = pts_moving_affine.copy()
for i, (x, y) in enumerate(pts_moving_affine):
    xi, yi = int(np.clip(x, 0, SIZE - 1)), int(np.clip(y, 0, SIZE - 1))
    pts_moving_elastic[i, 0] += dx[yi, xi]
    pts_moving_elastic[i, 1] += dy[yi, xi]

cv2.imwrite(f"{OUT}/reference.png", reference)
cv2.imwrite(f"{OUT}/moving_affine.png", moving_affine)
cv2.imwrite(f"{OUT}/moving_affine_elastic.png", moving_affine_elastic)

np.savez(f"{OUT}/ground_truth.npz",
         M_gt=M_gt, pts_ref=pts_ref,
         pts_moving_affine=pts_moving_affine, pts_moving_elastic=pts_moving_elastic,
         dx=dx, dy=dy)

with open(f"{OUT}/meta.json", "w") as f:
    json.dump({
        "size": SIZE,
        "affine_gt": {"angle_deg": 4.5, "tx": 9.0, "ty": -6.0, "scale": 1.02},
        "elastic_gt": {"alpha": 6.0, "sigma": 18.0},
    }, f, indent=2)

# --- figure for the opening of Part 1 ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, img, title in zip(
    axes, [reference, moving_affine, moving_affine_elastic],
    ["Reference (golden die image)", "Scenario A: stage drift\n(rotate + shift + scale)",
     "Scenario B: stage + thermal drift\n(affine + local elastic warp)"]
):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig("../../posts/series/image-registration/images/benchmark-pairs.png", dpi=140)
print("Saved benchmark images + ground truth to data/, figure to posts images/")
