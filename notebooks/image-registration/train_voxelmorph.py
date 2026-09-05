"""Deep-learning method 3: a from-scratch VoxelMorph-style deformable registration network
(Balakrishnan et al., 2018/2019 -- Apache-2.0 licensed original) -- a U-Net predicts a dense
per-pixel displacement field, a spatial transformer warps the moving image with it, and the
whole thing trains unsupervised on (similarity + smoothness) loss. No ground-truth field ever
appears in the loss.

This is the only method in the series that can actually fix Scenario B's local elastic
"thermal drift" component -- every affine-only method (classical or DL) hits a floor there
by construction, since a single 2x3 matrix cannot represent a spatially-varying warp.
"""
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from common import SIZE, control_points, landmark_error, ncc, ecc_affine, warp_affine, warp_points_affine

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
D = "data"
IMG_OUT = "../../posts/series/image-registration/images"


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.net(x)


class UNetFlow(nn.Module):
    """Small U-Net: (reference, moving) -> dense 2-channel displacement field."""

    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(2, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.pool = nn.AvgPool2d(2)
        self.bott = ConvBlock(64, 64)
        self.up3 = ConvBlock(64 + 64, 32)
        self.up2 = ConvBlock(32 + 32, 16)
        self.up1 = ConvBlock(16 + 16, 16)
        self.flow = nn.Conv2d(16, 2, 3, padding=1)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)

    def forward(self, ref, moving):
        x = torch.cat([ref, moving], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.up3(torch.cat([F.interpolate(b, scale_factor=2, mode="nearest"), e3], dim=1))
        d2 = self.up2(torch.cat([F.interpolate(d3, scale_factor=2, mode="nearest"), e2], dim=1))
        d1 = self.up1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], dim=1))
        return self.flow(d1)  # (B, 2, H, W) displacement in normalized coords


def make_base_grid(size, device):
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, size, device=device),
        torch.linspace(-1, 1, size, device=device), indexing="ij")
    return torch.stack([xs, ys], dim=-1)  # (H, W, 2)


def warp_with_flow(img, flow, base_grid):
    """flow: (B,2,H,W) normalized displacement. Spatial-transformer warp via grid_sample."""
    b = img.shape[0]
    grid = base_grid.unsqueeze(0).repeat(b, 1, 1, 1) + flow.permute(0, 2, 3, 1)
    return F.grid_sample(img, grid, align_corners=True, padding_mode="border")


def smoothness_loss(flow):
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return (dx.pow(2).mean() + dy.pow(2).mean())


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)


def elastic_field_torch(size, batch, device, alpha=6.0, sigma=18.0):
    """Same idea as common.elastic_field but batched and differentiable-input-safe (numpy blur,
    torch tensor out) -- used to synthesize training pairs with random local warps."""
    fields = []
    for _ in range(batch):
        dx = np.random.normal(0, 1, (size, size)).astype(np.float32)
        dy = np.random.normal(0, 1, (size, size)).astype(np.float32)
        k = int(sigma * 3) | 1
        dx = cv2.GaussianBlur(dx, (k, k), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (k, k), sigma) * alpha
        # convert pixel-displacement to normalized [-1,1] grid_sample displacement
        fields.append(np.stack([dx / (size / 2), dy / (size / 2)], axis=0))
    return torch.from_numpy(np.stack(fields)).to(device)


reference = cv2.imread(f"{D}/reference.png", cv2.IMREAD_GRAYSCALE)
ref_t = to_tensor(reference).to(device)
base_grid = make_base_grid(SIZE, device)

model = UNetFlow().to(device)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4000, eta_min=1e-5)

lambda_smooth = 20.0
n_steps = 4000
batch = 8
losses, sim_losses, smooth_losses = [], [], []

for step in range(n_steps):
    ref_batch = ref_t.repeat(batch, 1, 1, 1)

    # random affine component (rotation/scale/translation), matching common.random_affine_matrix's range
    angle = (torch.rand(batch) * 2 - 1) * np.deg2rad(6.0)
    scale = torch.rand(batch) * 0.08 + 0.96
    tx = (torch.rand(batch) * 2 - 1) * 0.10
    ty = (torch.rand(batch) * 2 - 1) * 0.10
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    theta = torch.zeros(batch, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    theta = theta.to(device)
    grid_affine = F.affine_grid(theta, ref_batch.shape, align_corners=True)
    moving_batch = F.grid_sample(ref_batch, grid_affine, align_corners=True, padding_mode="border")

    # plus random local elastic component
    elastic = elastic_field_torch(SIZE, batch, device).permute(0, 2, 3, 1)  # (B,H,W,2)
    grid_elastic = base_grid.unsqueeze(0) + elastic
    moving_batch = F.grid_sample(moving_batch, grid_elastic, align_corners=True, padding_mode="border")
    moving_batch = moving_batch + torch.randn_like(moving_batch) * (6.0 / 255.0)

    flow = model(ref_batch, moving_batch)
    registered = warp_with_flow(moving_batch, flow, base_grid)

    sim = F.mse_loss(registered, ref_batch)
    smooth = smoothness_loss(flow)
    loss = sim + lambda_smooth * smooth

    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    losses.append(loss.item()); sim_losses.append(sim.item()); smooth_losses.append(smooth.item())
    if step % 200 == 0 or step == n_steps - 1:
        print(f"step {step:4d}  loss {loss.item():.5f}  sim {sim.item():.5f}  smooth {smooth.item():.6f}")

plt.figure(figsize=(5, 3.5))
plt.plot(sim_losses, label="similarity (MSE)")
plt.plot(smooth_losses, label="smoothness")
plt.yscale("log"); plt.legend(); plt.xlabel("step"); plt.title("VoxelMorph-style training")
plt.tight_layout()
plt.savefig(f"{IMG_OUT}/dl-voxelmorph-loss-curve.png", dpi=140)

# --- evaluate on the real benchmark ---
gt = np.load(f"{D}/ground_truth.npz")
pts_ref = gt["pts_ref"]
scenarios = {
    "affine": (cv2.imread(f"{D}/moving_affine.png", cv2.IMREAD_GRAYSCALE), gt["pts_moving_affine"]),
    "elastic": (cv2.imread(f"{D}/moving_affine_elastic.png", cv2.IMREAD_GRAYSCALE), gt["pts_moving_elastic"]),
}

model.eval()
results = {}
registered_imgs = {}
flow_fields = {}
with torch.no_grad():
    for scen_name, (moving, pts_moving_gt) in scenarios.items():
        moving_t = to_tensor(moving).to(device)
        flow = model(ref_t, moving_t)
        registered_t = warp_with_flow(moving_t, flow, base_grid)
        registered = (registered_t[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        flow_np = flow[0].cpu().numpy() * (SIZE / 2)  # back to pixel displacement
        # registered(x,y) = moving(x + flow_x(x,y), y + flow_y(x,y)) -- the flow is indexed by
        # REFERENCE-frame coordinates (that's the output/sampling grid grid_sample uses), so
        # for each known reference-frame landmark we look up the flow AT that reference-frame
        # location and add it, giving the model's predicted moving-frame position -- directly
        # comparable to the ground-truth moving-frame position, no inversion needed.
        pred_moving_pts = []
        for (rx, ry) in pts_ref:
            xi, yi = int(np.clip(rx, 0, SIZE - 1)), int(np.clip(ry, 0, SIZE - 1))
            pred_moving_pts.append([rx + flow_np[0, yi, xi], ry + flow_np[1, yi, xi]])
        pred_moving_pts = np.array(pred_moving_pts, dtype=np.float32)

        err = landmark_error(pred_moving_pts, pts_moving_gt)
        score = ncc(reference, registered)
        results[f"dl_voxelmorph/{scen_name}"] = {"landmark_error_px": err, "ncc_after": score}
        registered_imgs[scen_name] = registered
        flow_fields[scen_name] = flow_np

with open(f"{D}/dl_voxelmorph_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))

fig, axes = plt.subplots(1, 5, figsize=(19, 4.8))
axes[0].imshow(reference, cmap="gray"); axes[0].set_title("Reference", fontsize=11)
axes[1].imshow(scenarios["elastic"][0], cmap="gray"); axes[1].set_title("Moving\n(affine + elastic)", fontsize=11)
err_e = results["dl_voxelmorph/elastic"]["landmark_error_px"]
axes[2].imshow(registered_imgs["elastic"], cmap="gray")
axes[2].set_title(f"VoxelMorph-style\nerr={err_e:.2f}px", fontsize=11)
fx, fy = flow_fields["elastic"]
step = 12
ys, xs = np.mgrid[0:SIZE:step, 0:SIZE:step]
axes[3].imshow(reference, cmap="gray")
axes[3].quiver(xs, ys, fx[::step, ::step], fy[::step, ::step], color="red", angles="xy",
               scale_units="xy", scale=1)
axes[3].set_title("Predicted displacement field", fontsize=11)
err_a = results["dl_voxelmorph/affine"]["landmark_error_px"]
axes[4].imshow(registered_imgs["affine"], cmap="gray")
axes[4].set_title(f"VoxelMorph-style\n(affine-only scen.) err={err_a:.2f}px", fontsize=11)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(pad=1.5, w_pad=2.0)
plt.savefig(f"{IMG_OUT}/dl-voxelmorph-result.png", dpi=140)
print("Saved dl-voxelmorph-loss-curve.png and dl-voxelmorph-result.png")

# =====================================================================================
# Two-stage pipeline: ECC affine pre-alignment, then a SECOND deformable network trained
# to correct only the residual local warp -- the standard real-world architecture (this is
# what VoxelMorph's own papers do too: affine pre-registration, then a deformable stage on
# top). The single-stage network above had to represent a large rigid rotation/translation
# *and* the local elastic warp in one dense field, which is a much harder function for a
# small U-Net to learn well; splitting the two apart plays to each method's strength.
# =====================================================================================
print("\n--- Two-stage pipeline: ECC affine pre-alignment + residual deformable network ---")

model2 = UNetFlow().to(device)
opt2 = torch.optim.Adam(model2.parameters(), lr=2e-3)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=n_steps, eta_min=1e-5)

losses2, sim_losses2, smooth_losses2 = [], [], []
for step in range(n_steps):
    ref_batch = ref_t.repeat(batch, 1, 1, 1)

    # ONLY the local elastic component this time -- no random affine -- since this network
    # only ever runs after an affine pre-alignment stage has already removed the rigid part.
    elastic = elastic_field_torch(SIZE, batch, device).permute(0, 2, 3, 1)
    grid_elastic = base_grid.unsqueeze(0) + elastic
    moving_batch = F.grid_sample(ref_batch, grid_elastic, align_corners=True, padding_mode="border")
    moving_batch = moving_batch + torch.randn_like(moving_batch) * (6.0 / 255.0)

    flow = model2(ref_batch, moving_batch)
    registered = warp_with_flow(moving_batch, flow, base_grid)

    sim = F.mse_loss(registered, ref_batch)
    smooth = smoothness_loss(flow)
    loss = sim + lambda_smooth * smooth

    opt2.zero_grad(); loss.backward(); opt2.step(); sched2.step()
    losses2.append(loss.item()); sim_losses2.append(sim.item()); smooth_losses2.append(smooth.item())
    if step % 200 == 0 or step == n_steps - 1:
        print(f"step {step:4d}  loss {loss.item():.5f}  sim {sim.item():.5f}  smooth {smooth.item():.6f}")

plt.figure(figsize=(5, 3.5))
plt.plot(sim_losses2, label="similarity (MSE)")
plt.plot(smooth_losses2, label="smoothness")
plt.yscale("log"); plt.legend(); plt.xlabel("step"); plt.title("Residual-warp network training")
plt.tight_layout()
plt.savefig(f"{IMG_OUT}/dl-voxelmorph-twostage-loss-curve.png", dpi=140)

model2.eval()
results2 = {}
registered_imgs2 = {}
flow_fields2 = {}
with torch.no_grad():
    for scen_name, (moving, pts_moving_gt) in scenarios.items():
        # stage 1: ECC affine pre-alignment (moving -> ref)
        M_ecc = ecc_affine(reference, moving)
        affine_corrected = warp_affine(moving, M_ecc, SIZE)
        pts_affine_corrected_gt = warp_points_affine(pts_moving_gt, M_ecc)

        # stage 2: residual deformable network on top of the affine-corrected image
        moving_t = to_tensor(affine_corrected).to(device)
        flow = model2(ref_t, moving_t)
        registered_t = warp_with_flow(moving_t, flow, base_grid)
        registered = (registered_t[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        flow_np = flow[0].cpu().numpy() * (SIZE / 2)
        pred_pts = []
        for (rx, ry) in pts_ref:
            xi, yi = int(np.clip(rx, 0, SIZE - 1)), int(np.clip(ry, 0, SIZE - 1))
            pred_pts.append([rx + flow_np[0, yi, xi], ry + flow_np[1, yi, xi]])
        pred_pts = np.array(pred_pts, dtype=np.float32)

        # compare the full pipeline's prediction against the affine-corrected-frame ground
        # truth (stage 1's own output frame) -- this scores the *combined* pipeline end to end
        err = landmark_error(pred_pts, pts_affine_corrected_gt)
        score = ncc(reference, registered)
        results2[f"dl_voxelmorph_twostage/{scen_name}"] = {"landmark_error_px": err, "ncc_after": score}
        registered_imgs2[scen_name] = registered
        flow_fields2[scen_name] = flow_np

with open(f"{D}/dl_voxelmorph_twostage_results.json", "w") as f:
    json.dump(results2, f, indent=2)
print(json.dumps(results2, indent=2))

fig, axes = plt.subplots(1, 5, figsize=(19, 4.8))
axes[0].imshow(reference, cmap="gray"); axes[0].set_title("Reference", fontsize=11)
axes[1].imshow(scenarios["elastic"][0], cmap="gray"); axes[1].set_title("Moving\n(affine + elastic)", fontsize=11)
err_e2 = results2["dl_voxelmorph_twostage/elastic"]["landmark_error_px"]
axes[2].imshow(registered_imgs2["elastic"], cmap="gray")
axes[2].set_title(f"ECC + residual net\nerr={err_e2:.2f}px", fontsize=11)
fx2, fy2 = flow_fields2["elastic"]
qstep = 12
axes[3].imshow(reference, cmap="gray")
axes[3].quiver(xs, ys, fx2[::qstep, ::qstep], fy2[::qstep, ::qstep], color="red", angles="xy",
               scale_units="xy", scale=1)
axes[3].set_title("Predicted residual field", fontsize=11)
err_a2 = results2["dl_voxelmorph_twostage/affine"]["landmark_error_px"]
axes[4].imshow(registered_imgs2["affine"], cmap="gray")
axes[4].set_title(f"ECC + residual net\n(affine-only scen.) err={err_a2:.2f}px", fontsize=11)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(pad=1.5, w_pad=2.0)
plt.savefig(f"{IMG_OUT}/dl-voxelmorph-twostage-result.png", dpi=140)
print("Saved dl-voxelmorph-twostage-loss-curve.png and dl-voxelmorph-twostage-result.png")
