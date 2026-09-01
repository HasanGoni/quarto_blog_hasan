"""Real, from-scratch, unconditional joint image+mask diffusion model --
Synthetic Data series Part 5.

Parts 2-4 all treat the defect mask as a *conditioning input*: something to
fill (Part 2), a real shape to respect (Part 3), or the same mask fed to a
different base model (Part 4). This part inverts the setup entirely,
inspired by NVIDIA's MAISI/NV-Generate-CTMR line of work (joint 3D
image+segmentation synthesis, scoped down here to 2D): train a small
diffusion model to generate the defect *image and its mask together*, as one
joint sample from noise, with no conditioning signal at all -- the mask is
learned from the real distribution of defect shapes, not drawn by hand or
borrowed from a real crop.

Real design choice, stated plainly: this is a from-scratch model on a small
(2-channel, 64x64) UNet2DModel, not a channel-expansion of Parts 2-4's
pretrained SD checkpoints. A literature check (MedSegFactory's dual-stream
cross-attention joint diffusion) confirmed the "real" way to do this well is
a meaningfully larger undertaking than fits here; channel-surgery on an 860M+
parameter pretrained UNet (padding its input/output convolutions from 4 to 8
channels) is fragile and loses the ability to reuse a documented pipeline
class the way Parts 2-4 could. A small from-scratch model, trained directly
in pixel space (no VAE, no text encoder -- nothing pretrained to lean on, so
nothing pretrained to misuse either) is the honest, tractable version of the
same idea at this dataset's real scale (338 real examples): a real, if
higher-risk, test of whether joint generation learns better defect shapes
than Parts 2-4's conditioning tricks, reported honestly either way.

Run:
    uv run train_joint_diffusion.py --steps 2000
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.training_utils import EMAModel
from PIL import Image

from synth_xray.data import download_and_extract, find_groundtruth_file, find_series_dirs
from synth_xray.diffusion_data import build_defect_dataset

RESOLUTION = 64  # small, from-scratch model on a small (338-example) real dataset


def to_joint_tensor(example: dict, device: str) -> torch.Tensor:
    """Real (image, mask) pair -> a single (1, 2, 64, 64) tensor in [-1, 1]:
    channel 0 = grayscale image, channel 1 = defect mask."""
    image = Image.fromarray(np.clip(example["image"], 0, 255).astype(np.uint8)).resize(
        (RESOLUTION, RESOLUTION), Image.LANCZOS
    )
    mask = Image.fromarray((example["mask"].astype(np.uint8) * 255)).resize(
        (RESOLUTION, RESOLUTION), Image.NEAREST
    )
    image_t = torch.from_numpy(np.array(image, dtype=np.float32) / 127.5 - 1.0)
    mask_t = torch.from_numpy(np.array(mask, dtype=np.float32) / 127.5 - 1.0)
    return torch.stack([image_t, mask_t], dim=0).unsqueeze(0).to(device)


def train(steps: int, out_dir: Path, lr: float, device: str, cache_dir: Path, n_samples_grid: int) -> None:
    group_dir = download_and_extract("Welds", cache_dir)
    series_dirs = [d for d in find_series_dirs(group_dir) if find_groundtruth_file(d) is not None]
    examples = build_defect_dataset(series_dirs[0])
    print(f"{len(examples)} real GDXray defect (image, mask) pairs for joint diffusion training")

    model = UNet2DModel(
        sample_size=RESOLUTION,
        in_channels=2,
        out_channels=2,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"UNet2DModel: {total_params:,} total params, all trainable (real from-scratch model, no pretrained weights)")

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # diffusers' own EMAModel utility, real not hand-rolled. decay=0.995 (not the
    # library default 0.9999) deliberately: EMA's time constant is roughly
    # 1/(1-decay) steps, so 0.9999 (~10,000-step time constant) would barely move
    # at all across this run's 2000 steps -- 0.995 (~200-step time constant) is
    # sized to this run's real step budget, not copied from a recipe meant for
    # training runs two to three orders of magnitude longer.
    ema = EMAModel(model.parameters(), decay=0.995, model_cls=UNet2DModel, model_config=model.config)

    out_dir.mkdir(parents=True, exist_ok=True)
    losses = []
    gif_frames = [_render_sample_grid(model, ema, noise_scheduler, device, n_samples_grid, step=0, num_inference_steps=50)]

    for step in range(steps):
        example = examples[step % len(examples)]
        real = to_joint_tensor(example, device)

        noise = torch.randn_like(real)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
        noisy = noise_scheduler.add_noise(real, noise, timesteps)

        noise_pred = model(noisy, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.step(model.parameters())
        losses.append(loss.item())

        if step % 100 == 0 or step == steps - 1:
            print(f"step {step:4d}/{steps}  loss {loss.item():.4f}")

        frame_every = max(1, steps // 20)
        if (step + 1) % frame_every == 0 or step == steps - 1:
            gif_frames.append(_render_sample_grid(model, ema, noise_scheduler, device, n_samples_grid, step + 1, num_inference_steps=50))

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("training step")
    plt.ylabel("noise-prediction MSE loss")
    plt.title("Joint image+mask diffusion loss (real GDXray weld defects, from scratch)")
    plt.tight_layout()
    plt.savefig(out_dir / "joint_loss_curve.png", dpi=130)

    gif_frames[0].save(
        out_dir / "joint_training_progress.gif", save_all=True,
        append_images=gif_frames[1:] + [gif_frames[-1]] * 4, duration=600, loop=0,
    )
    (out_dir / "joint_losses.json").write_text(json.dumps(losses))
    # Ship the EMA weights, not the raw live-training weights -- the whole point
    # of adding EMA is that the smoothed weights are the ones actually meant for
    # sampling/inference, matching every real diffusion training recipe (MAISI
    # included), not a training-script convenience.
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    torch.save(model.state_dict(), out_dir / "joint_unet_ema.pt")
    ema.restore(model.parameters())
    torch.save(model.state_dict(), out_dir / "joint_unet_final.pt")

    print(f"Saved joint_loss_curve.png + joint_training_progress.gif + joint_unet_ema.pt + joint_unet_final.pt -> {out_dir}")


@torch.no_grad()
def sample_joint(model, scheduler, device, n: int, num_inference_steps: int = 50, generator=None) -> torch.Tensor:
    """Real reverse-diffusion sampling loop: pure noise -> a batch of real
    joint (image, mask) samples. Standard DDPM ancestral sampling via
    `scheduler.step`, unconditional (no prompt, no mask input -- nothing to
    condition on, by design)."""
    model.eval()
    scheduler.set_timesteps(num_inference_steps)
    sample = torch.randn(n, 2, RESOLUTION, RESOLUTION, device=device, generator=generator)
    for t in scheduler.timesteps:
        noise_pred = model(sample, t).sample
        sample = scheduler.step(noise_pred, t, sample).prev_sample
    model.train()
    return sample.clamp(-1, 1)


def _joint_tensor_to_panel(sample: torch.Tensor) -> Image.Image:
    """One (2, H, W) sample -> a side-by-side (image | mask) PIL panel."""
    image = ((sample[0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    mask = ((sample[1].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.concatenate([image, mask], axis=1))


@torch.no_grad()
def _render_sample_grid(model, ema, scheduler, device, n: int, step: int, num_inference_steps: int) -> Image.Image:
    # Sample from the EMA-smoothed weights, not the live/raw training weights --
    # store the live weights, swap the EMA ones in for generation, then restore
    # live weights afterward so training continues from exactly where it left off.
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    samples = sample_joint(model, scheduler, device, n, num_inference_steps=num_inference_steps)
    ema.restore(model.parameters())
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3 * ((n + 1) // 2), 6))
    axes = np.array(axes).reshape(-1)
    for i in range(n):
        panel = _joint_tensor_to_panel(samples[i])
        axes[i].imshow(panel, cmap="gray")
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.suptitle(f"step {step} -- each panel: generated image | generated mask", fontsize=10)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("joint_out"))
    ap.add_argument("--cache-dir", type=Path, default=Path(".data_cache"))
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--samples-grid", type=int, default=6)
    args = ap.parse_args()
    train(args.steps, args.out, args.lr, args.device, args.cache_dir, args.samples_grid)
