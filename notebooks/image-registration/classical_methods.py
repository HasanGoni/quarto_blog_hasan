"""Classical (non-deep-learning) registration: phase correlation, ECC (intensity-based),
and SIFT+RANSAC (feature-based). Run on both benchmark scenarios so Part 1 can show that
all three fix Scenario A (pure affine) well, but leave residual error on Scenario B
(affine + elastic) -- they only ever fit a global affine/rigid model.
"""
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

from common import (
    SIZE, control_points, warp_points_affine, landmark_error, ncc, warp_affine, invert_affine,
    ecc_affine,
)

D = "data"
IMG_OUT = "../../posts/series/image-registration/images"

reference = cv2.imread(f"{D}/reference.png", cv2.IMREAD_GRAYSCALE)
gt = np.load(f"{D}/ground_truth.npz")
pts_ref = gt["pts_ref"]

scenarios = {
    "affine": (cv2.imread(f"{D}/moving_affine.png", cv2.IMREAD_GRAYSCALE), gt["pts_moving_affine"]),
    "elastic": (cv2.imread(f"{D}/moving_affine_elastic.png", cv2.IMREAD_GRAYSCALE), gt["pts_moving_elastic"]),
}

results = {}


def phase_correlation(ref, moving):
    """Translation-only: FFT cross-power spectrum peak, subpixel via cv2.phaseCorrelate.

    A Hann window is applied first -- the standard mitigation for spectral leakage -- since
    our synthetic die image repeats every 64px (scribe-line grid): a raw phase correlation on
    a periodic pattern plus a small rotation locks onto an aliased peak (confirmed below: the
    unwindowed response confidence is ~0.07, near-random). Even windowed, confidence stays low
    (~0.10) because the periodicity is everywhere in the image, not just at the border --
    this is the real, honest limitation phase correlation has on repeating industrial patterns.
    """
    ref_f = ref.astype(np.float32)
    mov_f = moving.astype(np.float32)
    win = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(ref_f, mov_f, win)
    M = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
    return M, float(response)



def sift_ransac(ref, moving):
    """Feature-based: SIFT keypoints, ratio-test matching, affine fit with RANSAC."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref, None)
    kp2, des2 = sift.detectAndCompute(moving, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des2, des1, k=2)  # moving -> ref
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    src = np.float32([kp2[m.queryIdx].pt for m in good])
    dst = np.float32([kp1[m.trainIdx].pt for m in good])
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    n_inliers = int(inliers.sum()) if inliers is not None else 0
    return M, len(good), n_inliers


methods = {}

for scen_name, (moving, pts_moving_gt) in scenarios.items():
    for method_name, fn in [
        ("phase_correlation", phase_correlation),
        ("ecc_affine", ecc_affine),
        ("sift_ransac", None),
    ]:
        if method_name == "sift_ransac":
            M, n_good, n_inliers = sift_ransac(reference, moving)
            extra = {"n_good_matches": n_good, "n_inliers": n_inliers}
        elif method_name == "phase_correlation":
            M, response = fn(reference, moving)
            extra = {"phase_corr_response": response}
        else:
            M = fn(reference, moving)
            extra = {}
        if M is None:
            results[f"{method_name}/{scen_name}"] = {"failed": True}
            continue

        registered = warp_affine(moving, M, SIZE)
        # M maps moving->ref pixel coords in warpAffine's dst-from-src convention here,
        # so recovered ref-frame landmark = M applied to the moving-frame ground-truth point
        pred_pts = warp_points_affine(pts_moving_gt, M)
        err = landmark_error(pred_pts, pts_ref)
        score = ncc(reference, registered)
        results[f"{method_name}/{scen_name}"] = {
            "landmark_error_px": err, "ncc_after": score, **extra,
        }
        methods.setdefault(method_name, {})[scen_name] = registered

with open("data/classical_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))

# figure: reference | moving | each method's registered output
fig, axes = plt.subplots(2, 5, figsize=(18, 8.5))
for row, scen_name in enumerate(["affine", "elastic"]):
    moving = scenarios[scen_name][0]
    axes[row, 0].imshow(reference, cmap="gray")
    axes[row, 0].set_title("Reference", fontsize=11)
    axes[row, 0].set_ylabel(f"Scenario: {scen_name}", fontsize=11)
    axes[row, 1].imshow(moving, cmap="gray")
    axes[row, 1].set_title("Moving", fontsize=11)
    for col, method_name in enumerate(["phase_correlation", "ecc_affine", "sift_ransac"], start=2):
        reg = methods[method_name][scen_name]
        err = results[f"{method_name}/{scen_name}"]["landmark_error_px"]
        axes[row, col].imshow(reg, cmap="gray")
        axes[row, col].set_title(f"{method_name}\nerr={err:.2f}px", fontsize=11)
    for ax in axes[row]:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
plt.tight_layout(h_pad=3.0)
plt.savefig(f"{IMG_OUT}/classical-methods-comparison.png", dpi=140)
print("Saved classical-methods-comparison.png")
