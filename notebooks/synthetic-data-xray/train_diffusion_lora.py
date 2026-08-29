"""Real LoRA fine-tune of a mask-conditioned Stable Diffusion inpainting model on
real GDXray weld defect crops + masks -- Synthetic Data series Part 2.

The question this answers: does a generative model that has actually seen real
X-ray defect *texture* produce better synthetic defects than Part 1's
physics-based simulation, which has no learned texture prior at all -- only a
calibrated attenuation/noise/blur model?

What's real here, not pseudocode:
- `stable-diffusion-v1-5/stable-diffusion-inpainting` is loaded as-is (public,
  non-gated) -- a real, standard mask-conditioned inpainting UNet: 9 input
  channels (4 noisy-latent + 1 mask + 4 masked-image-latent), exactly the
  scheme it was itself originally trained with.
- LoRA (`r=8, alpha=16`, targeting the UNet's `to_q/to_k/to_v/to_out.0`
  attention-projection layers) is injected via diffusers' own
  `UNet2DConditionModel.add_adapter()` -- the base UNet stays frozen; only the
  LoRA layers train.
- Every (image, mask) training pair is a real GDXray Welds defect box (via
  `synth_xray.diffusion_data.build_defect_dataset`), not a synthetic one --
  this fine-tune's whole point is to expose the model to real defect texture.
- A real noise-prediction diffusion loss (DDPM forward process + MSE against
  the sampled noise), the same objective this checkpoint was itself trained
  with -- not a simplified/proxy loss.

Run:
    uv run train_diffusion_lora.py --steps 150
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
from diffusers import DDPMScheduler, StableDiffusionInpaintPipeline
from peft import LoraConfig
from PIL import Image

from synth_xray.data import download_and_extract, find_groundtruth_file, find_series_dirs
from synth_xray.diffusion_data import build_defect_dataset, to_mask_pil, to_rgb_pil

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
# Welds ground truth carries no per-box defect class, so (matching Part 1's own
# generic "crack" framing) every training pair uses the same single prompt.
PROMPT = "a weld radiograph with a crack defect"


def pil_to_image_tensor(img: Image.Image, device: str) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # HWC, [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def pil_to_mask_tensor(img: Image.Image, device: str) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0  # HW, [0, 1]
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def train(steps: int, out_dir: Path, lr: float, device: str, cache_dir: Path, gif_frames_n: int) -> None:
    group_dir = download_and_extract("Welds", cache_dir)
    series_dirs = [d for d in find_series_dirs(group_dir) if find_groundtruth_file(d) is not None]
    examples = build_defect_dataset(series_dirs[0])
    print(f"{len(examples)} real GDXray defect (image, mask) pairs for LoRA fine-tuning")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32, safety_checker=None)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    vae, unet, text_encoder, tokenizer = pipe.vae, pipe.unet, pipe.text_encoder, pipe.tokenizer
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    unet.add_adapter(lora_config)

    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    trainable = [p for p in unet.parameters() if p.requires_grad]
    total = sum(p.numel() for p in unet.parameters())
    print(f"trainable LoRA params: {sum(p.numel() for p in trainable):,} / {total:,} total UNet params "
          f"({100 * sum(p.numel() for p in trainable) / total:.3f}%)")
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    with torch.no_grad():
        text_inputs = tokenizer([PROMPT], padding="max_length", max_length=tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt")
        prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]

    fixed_example = examples[0]
    fixed_image_pil = to_rgb_pil(fixed_example["image"])
    fixed_mask_pil = to_mask_pil(fixed_example["mask"])

    out_dir.mkdir(parents=True, exist_ok=True)
    losses = []
    gif_frames = [_render_frame(pipe, fixed_image_pil, fixed_mask_pil, 0)]

    for step in range(steps):
        example = examples[step % len(examples)]
        image_pil = to_rgb_pil(example["image"])
        mask_pil = to_mask_pil(example["mask"])

        image_t = pil_to_image_tensor(image_pil, device)
        mask_t = pil_to_mask_tensor(mask_pil, device)
        masked_image_t = image_t * (mask_t < 0.5)

        with torch.no_grad():
            latents = vae.encode(image_t).latent_dist.sample() * vae.config.scaling_factor
            masked_latents = vae.encode(masked_image_t).latent_dist.sample() * vae.config.scaling_factor
        mask_latent = F.interpolate(mask_t, size=latents.shape[-2:])

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        unet_input = torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)
        noise_pred = unet(unet_input, timesteps, encoder_hidden_states=prompt_embeds).sample
        loss = F.mse_loss(noise_pred.float(), noise.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 10 == 0 or step == steps - 1:
            print(f"step {step:3d}/{steps}  loss {loss.item():.4f}")

        frame_every = max(1, steps // gif_frames_n)
        if (step + 1) % frame_every == 0 or step == steps - 1:
            gif_frames.append(_render_frame(pipe, fixed_image_pil, fixed_mask_pil, step + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("training step")
    plt.ylabel("noise-prediction MSE loss")
    plt.title("SD-inpainting LoRA fine-tune loss (real GDXray weld defects)")
    plt.tight_layout()
    plt.savefig(out_dir / "diffusion_loss_curve.png", dpi=130)

    gif_frames[0].save(
        out_dir / "diffusion_training_progress.gif", save_all=True,
        append_images=gif_frames[1:] + [gif_frames[-1]] * 4, duration=500, loop=0,
    )
    (out_dir / "diffusion_losses.json").write_text(json.dumps(losses))

    lora_state_dict = {k: v.cpu() for k, v in unet.state_dict().items() if "lora" in k}
    torch.save(lora_state_dict, out_dir / "lora_weights.pt")

    print(f"Saved diffusion_loss_curve.png + diffusion_training_progress.gif + lora_weights.pt -> {out_dir}")


@torch.no_grad()
def _render_frame(pipe, image_pil: Image.Image, mask_pil: Image.Image, step: int) -> Image.Image:
    pipe.unet.eval()
    result = pipe(
        prompt=PROMPT, image=image_pil, mask_image=mask_pil,
        num_inference_steps=15, guidance_scale=1.0,
    ).images[0]
    pipe.unet.train()

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(image_pil.convert("L"), cmap="gray")
    axes[0].imshow(np.array(mask_pil) > 127, cmap="Reds", alpha=0.3)
    axes[0].set_title("input + mask", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(result)
    axes[1].set_title(f"LoRA output, step {step}", fontsize=9)
    axes[1].axis("off")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path("diffusion_out"))
    ap.add_argument("--cache-dir", type=Path, default=Path(".data_cache"))
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gif-frames", type=int, default=20)
    args = ap.parse_args()
    train(args.steps, args.out, args.lr, args.device, args.cache_dir, args.gif_frames)
