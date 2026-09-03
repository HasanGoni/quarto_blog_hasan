"""Real, from-scratch, defect-aware conditional GAN -- Synthetic Data series Part 7.

A distinct third generative family from the diffusion track (Parts 2-6): a DFMGAN-style
(Defect-aware, "Few-Shot Defect Image Generation via Defect-Aware Feature Manipulation",
WACV 2023) idea, scoped down the same way Part 5 scoped MAISI to 2D -- not the full
StyleGAN2-backbone architecture from the paper (another large pretrained-model project of
its own), but the same real, core premise: a generator conditioned on a mask and its
surrounding defect-free context, trained adversarially plus an L1 reconstruction term
(the standard pix2pix recipe) for stability, since GANs are notoriously unstable on tiny
real datasets (338 examples here) -- an honest-limitation angle the design spec flagged
going in, not something to avoid because it might not "win."

Self-supervised training pair, built the same way Part 2's inpainting pairs were: for each
real (image, mask) example, the *masked* image (real image with the defect region blacked
out) plus the mask are the generator's condition; the real, unmasked image is the target.

Run:
    uv run train_gan.py --steps 3000
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from synth_xray.data import download_and_extract, find_groundtruth_file, find_series_dirs
from synth_xray.diffusion_data import build_defect_dataset

RESOLUTION = 128  # from-scratch GAN, small (338-example) real dataset -- see module docstring


def to_training_tensors(example: dict, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Real (image, mask) pair -> (masked_image, mask, real_image), each (1, 1, 128, 128) in [-1, 1]."""
    image = np.clip(example["image"], 0, 255).astype(np.uint8)
    mask = example["mask"]

    masked = image.copy()
    masked[mask] = 0  # black out the real defect region -- the generator's only view of "context"

    image_pil = Image.fromarray(image).resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
    masked_pil = Image.fromarray(masked).resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
    mask_pil = Image.fromarray(mask.astype(np.uint8) * 255).resize((RESOLUTION, RESOLUTION), Image.NEAREST)

    real_t = torch.from_numpy(np.array(image_pil, dtype=np.float32) / 127.5 - 1.0).unsqueeze(0)
    masked_t = torch.from_numpy(np.array(masked_pil, dtype=np.float32) / 127.5 - 1.0).unsqueeze(0)
    mask_t = torch.from_numpy(np.array(mask_pil, dtype=np.float32) / 127.5 - 1.0).unsqueeze(0)
    return masked_t.unsqueeze(0).to(device), mask_t.unsqueeze(0).to(device), real_t.unsqueeze(0).to(device)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, down: bool, use_norm: bool = True):
        super().__init__()
        conv = (
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)
            if down
            else nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        )
        layers = [conv]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Generator(nn.Module):
    """A small U-Net-style encoder-decoder, pix2pix-shaped: skip connections carry real
    spatial detail from the masked-context input straight through to the decoder, so the
    generator only has to *invent* content inside the mask, not re-derive the whole image."""

    def __init__(self, noise_dim: int = 32):
        super().__init__()
        self.noise_dim = noise_dim
        # input: masked_image (1ch) + mask (1ch) = 2 channels
        self.e1 = ConvBlock(2, 64, down=True, use_norm=False)  # 128 -> 64
        self.e2 = ConvBlock(64, 128, down=True)  # 64 -> 32
        self.e3 = ConvBlock(128, 256, down=True)  # 32 -> 16
        self.e4 = ConvBlock(256, 512, down=True)  # 16 -> 8

        self.d4 = ConvBlock(512 + noise_dim, 256, down=False)  # 8 -> 16
        self.d3 = ConvBlock(512, 128, down=False)  # 16 -> 32 (with skip)
        self.d2 = ConvBlock(256, 64, down=False)  # 32 -> 64 (with skip)
        self.out = nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1)  # 64 -> 128 (with skip)

    def forward(self, masked_image: torch.Tensor, mask: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        x = torch.cat([masked_image, mask], dim=1)
        f1 = self.e1(x)
        f2 = self.e2(f1)
        f3 = self.e3(f2)
        f4 = self.e4(f3)

        noise_map = noise.view(noise.shape[0], self.noise_dim, 1, 1).expand(-1, -1, f4.shape[2], f4.shape[3])
        u4 = self.d4(torch.cat([f4, noise_map], dim=1))
        u3 = self.d3(torch.cat([u4, f3], dim=1))
        u2 = self.d2(torch.cat([u3, f2], dim=1))
        out = self.out(torch.cat([u2, f1], dim=1))
        return torch.tanh(out)


class PatchDiscriminator(nn.Module):
    """PatchGAN: classifies overlapping patches as real/fake rather than the whole image at
    once -- standard pix2pix design, and a real stabilizer on tiny datasets since it gives
    many effective training signals per image instead of one."""

    def __init__(self):
        super().__init__()
        # input: masked_image (1ch) + mask (1ch) + candidate full image (1ch) = 3 channels
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),  # 128 -> 64
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),  # -> 32
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),  # -> 16
            nn.Conv2d(256, 1, 4, stride=1, padding=1),  # -> patch logits, no sigmoid (LSGAN loss)
        )

    def forward(self, masked_image: torch.Tensor, mask: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([masked_image, mask, candidate], dim=1))


def train(steps: int, out_dir: Path, lr: float, device: str, cache_dir: Path, lambda_l1: float, noise_dim: int) -> None:
    group_dir = download_and_extract("Welds", cache_dir)
    series_dirs = [d for d in find_series_dirs(group_dir) if find_groundtruth_file(d) is not None]
    examples = build_defect_dataset(series_dirs[0])
    print(f"{len(examples)} real GDXray defect (image, mask) pairs for GAN training")

    G = Generator(noise_dim=noise_dim).to(device)
    D = PatchDiscriminator().to(device)
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"Generator: {g_params:,} params, Discriminator: {d_params:,} params -- both real, trained from scratch")

    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_example = examples[0]
    fixed_masked, fixed_mask, fixed_real = to_training_tensors(fixed_example, device)
    fixed_noise = torch.randn(1, noise_dim, device=device)

    out_dir.mkdir(parents=True, exist_ok=True)
    g_losses, d_losses, l1_losses = [], [], []
    gif_frames = [_render_panel(G, fixed_masked, fixed_mask, fixed_real, fixed_noise, step=0)]

    for step in range(steps):
        example = examples[step % len(examples)]
        masked_image, mask, real_image = to_training_tensors(example, device)
        noise = torch.randn(1, noise_dim, device=device)

        # --- Discriminator step ---
        with torch.no_grad():
            fake = G(masked_image, mask, noise)
        d_real = D(masked_image, mask, real_image)
        d_fake = D(masked_image, mask, fake)
        d_loss = 0.5 * (F.mse_loss(d_real, torch.ones_like(d_real)) + F.mse_loss(d_fake, torch.zeros_like(d_fake)))
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # --- Generator step ---
        fake = G(masked_image, mask, noise)
        d_fake_for_g = D(masked_image, mask, fake)
        g_adv_loss = F.mse_loss(d_fake_for_g, torch.ones_like(d_fake_for_g))
        g_l1_loss = F.l1_loss(fake, real_image)
        g_loss = g_adv_loss + lambda_l1 * g_l1_loss
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        g_losses.append(g_adv_loss.item())
        d_losses.append(d_loss.item())
        l1_losses.append(g_l1_loss.item())

        if step % 100 == 0 or step == steps - 1:
            print(f"step {step:4d}/{steps}  G_adv {g_adv_loss.item():.4f}  D {d_loss.item():.4f}  L1 {g_l1_loss.item():.4f}")

        frame_every = max(1, steps // 20)
        if (step + 1) % frame_every == 0 or step == steps - 1:
            gif_frames.append(_render_panel(G, fixed_masked, fixed_mask, fixed_real, fixed_noise, step + 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(g_losses); axes[0].set_title("Generator adversarial loss"); axes[0].set_xlabel("step")
    axes[1].plot(d_losses); axes[1].set_title("Discriminator loss"); axes[1].set_xlabel("step")
    axes[2].plot(l1_losses); axes[2].set_title("Generator L1 reconstruction loss"); axes[2].set_xlabel("step")
    fig.suptitle("Defect-aware conditional GAN training (real GDXray weld defects, from scratch)")
    plt.tight_layout()
    plt.savefig(out_dir / "gan_loss_curves.png", dpi=130)

    gif_frames[0].save(
        out_dir / "gan_training_progress.gif", save_all=True,
        append_images=gif_frames[1:] + [gif_frames[-1]] * 4, duration=600, loop=0,
    )
    (out_dir / "gan_losses.json").write_text(json.dumps({"g": g_losses, "d": d_losses, "l1": l1_losses}))
    torch.save(G.state_dict(), out_dir / "gan_generator.pt")
    torch.save(D.state_dict(), out_dir / "gan_discriminator.pt")

    print(f"Saved gan_loss_curves.png + gan_training_progress.gif + gan_generator.pt + gan_discriminator.pt -> {out_dir}")


@torch.no_grad()
def _render_panel(G, masked_image, mask, real_image, noise, step: int) -> Image.Image:
    G.eval()
    fake = G(masked_image, mask, noise)
    G.train()

    def to_img(t: torch.Tensor) -> np.ndarray:
        return ((t[0, 0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)

    panel = np.concatenate([to_img(masked_image), to_img(fake), to_img(real_image)], axis=1)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.imshow(panel, cmap="gray")
    ax.axis("off")
    ax.set_title(f"step {step} -- masked input | generated | real", fontsize=10)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=Path("gan_out"))
    ap.add_argument("--cache-dir", type=Path, default=Path(".data_cache"))
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lambda-l1", type=float, default=100.0)
    ap.add_argument("--noise-dim", type=int, default=32)
    args = ap.parse_args()
    train(args.steps, args.out, args.lr, args.device, args.cache_dir, args.lambda_l1, args.noise_dim)
