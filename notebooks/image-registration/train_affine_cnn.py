"""Deep-learning method 1: a small CNN that regresses an affine transform directly,
trained *unsupervised* (no ground-truth transform in the loss -- only a photometric
reconstruction loss), the way real production registration nets are trained.

Framing: in a real fab you register against the *same* golden die layout over and over,
so it's realistic to train one small network per recipe against that fixed reference,
fed many random synthetic drifts + noise realizations, then run it at inference time on
every new inspection frame of that same product -- a few CNN forward passes instead of an
iterative optimizer (ECC) or a keypoint pipeline (SIFT) per image, which is the actual
throughput argument for going deep-learning here.
"""
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from common import (
    SIZE, control_points, warp_points_affine, landmark_error, ncc, warp_affine, invert_affine,
)

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

D = "data"
IMG_OUT = "../../posts/series/image-registration/images"


class AffineRegressor(nn.Module):
    """Two-channel (reference, moving) in -> 6 affine params out, STN-style zero-init head
    so training starts from an identity transform instead of a random one."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 6)
        nn.init.zeros_(self.head.weight)
        self.head.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))

    def forward(self, ref, moving):
        x = torch.cat([ref, moving], dim=1)
        feat = self.features(x).flatten(1)
        theta = self.head(feat).view(-1, 2, 3)
        return theta


def random_theta(batch: int, max_angle=6.0, max_trans=0.10, scale_range=(0.96, 1.04)):
    """Random affine params directly in normalized [-1,1] grid_sample coordinates."""
    angle = (torch.rand(batch) * 2 - 1) * np.deg2rad(max_angle)
    scale = torch.rand(batch) * (scale_range[1] - scale_range[0]) + scale_range[0]
    tx = (torch.rand(batch) * 2 - 1) * max_trans
    ty = (torch.rand(batch) * 2 - 1) * max_trans
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    theta = torch.zeros(batch, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    return theta


def theta_to_pixel_matrix(theta: np.ndarray, size: int) -> np.ndarray:
    """Converts a normalized-coordinate (align_corners=True) affine theta, used as
    grid_sample(dst=moving, src=reference)'s ref->moving lookup, into the equivalent
    pixel-space ref->moving 2x3 matrix. Valid for square images (W == H == size)."""
    c = (size - 1) / 2.0
    A = theta[:, :2]
    b = theta[:, 2]
    A_px = A  # exact for square images: S*A*S^{-1} == A when S = c * I
    b_px = (np.eye(2) - A) @ np.array([c, c]) + c * b
    return np.hstack([A_px, b_px.reshape(2, 1)]).astype(np.float32)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)


# --- reference used both for training augmentation and evaluation ---
reference = cv2.imread(f"{D}/reference.png", cv2.IMREAD_GRAYSCALE)
ref_t = to_tensor(reference).to(device)

model = AffineRegressor().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4000, eta_min=1e-5)

n_steps = 4000
batch = 32
losses = []
for step in range(n_steps):
    theta_gt = random_theta(batch).to(device)
    ref_batch = ref_t.repeat(batch, 1, 1, 1)
    grid = F.affine_grid(theta_gt, ref_batch.shape, align_corners=True)
    moving_batch = F.grid_sample(ref_batch, grid, align_corners=True, padding_mode="border")
    moving_batch = moving_batch + torch.randn_like(moving_batch) * (6.0 / 255.0)

    theta_pred = model(ref_batch, moving_batch)
    grid_pred = F.affine_grid(theta_pred, ref_batch.shape, align_corners=True)
    registered = F.grid_sample(moving_batch, grid_pred, align_corners=True, padding_mode="border")

    loss = F.mse_loss(registered, ref_batch)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    losses.append(loss.item())
    if step % 200 == 0 or step == n_steps - 1:
        print(f"step {step:4d}  loss {loss.item():.5f}")

plt.figure(figsize=(5, 3.5))
plt.plot(losses)
plt.xlabel("training step"); plt.ylabel("MSE reconstruction loss"); plt.title("Affine CNN training")
plt.tight_layout()
plt.savefig(f"{IMG_OUT}/dl-affine-cnn-loss-curve.png", dpi=140)

# --- evaluate on the real benchmark pairs (never seen during training) ---
gt = np.load(f"{D}/ground_truth.npz")
pts_ref = gt["pts_ref"]
scenarios = {
    "affine": (cv2.imread(f"{D}/moving_affine.png", cv2.IMREAD_GRAYSCALE), gt["pts_moving_affine"]),
    "elastic": (cv2.imread(f"{D}/moving_affine_elastic.png", cv2.IMREAD_GRAYSCALE), gt["pts_moving_elastic"]),
}

model.eval()
results = {}
registered_imgs = {}
with torch.no_grad():
    for scen_name, (moving, pts_moving_gt) in scenarios.items():
        moving_t = to_tensor(moving).to(device)
        theta_pred = model(ref_t, moving_t)
        M_r2m_px = theta_to_pixel_matrix(theta_pred.cpu().numpy()[0], SIZE)
        M_m2r_px = invert_affine(M_r2m_px)

        registered = warp_affine(moving, M_m2r_px, SIZE)
        pred_pts = warp_points_affine(pts_moving_gt, M_m2r_px)
        err = landmark_error(pred_pts, pts_ref)
        score = ncc(reference, registered)
        results[f"dl_affine_cnn/{scen_name}"] = {"landmark_error_px": err, "ncc_after": score}
        registered_imgs[scen_name] = registered

with open(f"{D}/dl_affine_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))

fig, axes = plt.subplots(1, 4, figsize=(15, 4.8))
axes[0].imshow(reference, cmap="gray"); axes[0].set_title("Reference", fontsize=11)
axes[1].imshow(scenarios["affine"][0], cmap="gray"); axes[1].set_title("Moving (affine)", fontsize=11)
axes[2].imshow(registered_imgs["affine"], cmap="gray")
axes[2].set_title(f"DL affine CNN\nerr={results['dl_affine_cnn/affine']['landmark_error_px']:.2f}px", fontsize=11)
axes[3].imshow(registered_imgs["elastic"], cmap="gray")
axes[3].set_title(f"DL affine CNN (elastic scen.)\nerr={results['dl_affine_cnn/elastic']['landmark_error_px']:.2f}px", fontsize=11)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(pad=1.5, w_pad=2.0)
plt.savefig(f"{IMG_OUT}/dl-affine-cnn-result.png", dpi=140)
print("Saved dl-affine-cnn-loss-curve.png and dl-affine-cnn-result.png")
