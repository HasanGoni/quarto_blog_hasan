"""Deep-learning method 2: LoFTR (detector-free deep feature matching), the DL successor
to SIFT+RANSAC -- a transformer finds dense correspondences directly, no separate keypoint
detector, then RANSAC still fits the affine/homography from those correspondences.

License note this post relies on: kornia's LoFTR ships under kornia's Apache-2.0 license
(both code and the pretrained "outdoor"/"indoor" weights). This matters because the
architecturally similar, extremely popular SuperGlue/SuperPoint pair from Magic Leap is
CC-BY-NC-SA (non-commercial only) -- LoFTR is the commercially-safe way to get the same
kind of deep feature matching.

Honest expectation set here: LoFTR's released weights are trained on natural photo scenes
(indoor rooms / outdoor landmarks), not synthetic industrial line-art -- this is a real
domain-gap test, not a rigged win for classical SIFT.
"""
import json
import numpy as np
import cv2
import torch
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt

from common import (
    SIZE, control_points, warp_points_affine, landmark_error, ncc, warp_affine,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
D = "data"
IMG_OUT = "../../posts/series/image-registration/images"

reference = cv2.imread(f"{D}/reference.png", cv2.IMREAD_GRAYSCALE)
gt = np.load(f"{D}/ground_truth.npz")
pts_ref = gt["pts_ref"]
scenarios = {
    "affine": (cv2.imread(f"{D}/moving_affine.png", cv2.IMREAD_GRAYSCALE), gt["pts_moving_affine"]),
    "elastic": (cv2.imread(f"{D}/moving_affine_elastic.png", cv2.IMREAD_GRAYSCALE), gt["pts_moving_elastic"]),
}

matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()


def to_kornia(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)


def loftr_affine(ref: np.ndarray, moving: np.ndarray):
    inp = {"image0": to_kornia(moving), "image1": to_kornia(ref)}
    with torch.no_grad():
        out = matcher(inp)
    mkpts0 = out["keypoints0"].cpu().numpy()  # in moving
    mkpts1 = out["keypoints1"].cpu().numpy()  # in ref
    conf = out["confidence"].cpu().numpy()
    keep = conf > 0.5
    mkpts0, mkpts1 = mkpts0[keep], mkpts1[keep]
    if len(mkpts0) < 3:
        return None, mkpts0, mkpts1, conf[keep]
    M, inliers = cv2.estimateAffinePartial2D(mkpts0, mkpts1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return M, mkpts0, mkpts1, conf[keep]


results = {}
registered_imgs = {}
match_counts = {}
for scen_name, (moving, pts_moving_gt) in scenarios.items():
    M, mkpts0, mkpts1, conf = loftr_affine(reference, moving)
    match_counts[scen_name] = {"n_matches_conf>0.5": int(len(mkpts0)),
                                "mean_confidence": float(conf.mean()) if len(conf) else 0.0}
    if M is None:
        results[f"loftr/{scen_name}"] = {"failed": True, **match_counts[scen_name]}
        continue
    registered = warp_affine(moving, M, SIZE)
    pred_pts = warp_points_affine(pts_moving_gt, M)
    err = landmark_error(pred_pts, pts_ref)
    score = ncc(reference, registered)
    results[f"loftr/{scen_name}"] = {"landmark_error_px": err, "ncc_after": score, **match_counts[scen_name]}
    registered_imgs[scen_name] = registered

with open(f"{D}/loftr_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))

# figure: correspondences + registered result on the affine scenario
moving_affine = scenarios["affine"][0]
M, mkpts0, mkpts1, conf = loftr_affine(reference, moving_affine)
h, w = reference.shape
canvas = np.concatenate([moving_affine, reference], axis=1)
canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
    p0 = (int(x0), int(y0))
    p1 = (int(x1 + w), int(y1))
    cv2.line(canvas, p0, p1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(canvas, p0, 2, (0, 0, 255), -1)
    cv2.circle(canvas, p1, 2, (0, 0, 255), -1)
cv2.imwrite(f"{IMG_OUT}/loftr-correspondences.png", canvas)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
axes[0].imshow(reference, cmap="gray"); axes[0].set_title("Reference", fontsize=11)
axes[1].imshow(moving_affine, cmap="gray"); axes[1].set_title("Moving (affine)", fontsize=11)
if "affine" in registered_imgs:
    err = results["loftr/affine"]["landmark_error_px"]
    axes[2].imshow(registered_imgs["affine"], cmap="gray")
    axes[2].set_title(f"LoFTR + RANSAC\nerr={err:.2f}px, n={match_counts['affine']['n_matches_conf>0.5']}", fontsize=11)
else:
    axes[2].text(0.5, 0.5, "LoFTR failed:\ntoo few confident matches", ha="center", va="center")
    axes[2].set_title("LoFTR + RANSAC", fontsize=11)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(pad=1.5, w_pad=2.0)
plt.savefig(f"{IMG_OUT}/loftr-result.png", dpi=140)
print("Saved loftr-correspondences.png and loftr-result.png")
