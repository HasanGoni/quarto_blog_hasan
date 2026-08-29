"""Real LoRA fine-tune of a ControlNet on real GDXray weld defect *shapes* --
Synthetic Data series Part 3.

Part 2 showed diffusion learns defect texture but the generated shape still doesn't
read as an authentic defect: plain mask conditioning only tells the model what
region to fill, not what internal shape the content should take. This part tests
whether feeding the model the *real* defect's own silhouette (not a bare rectangle,
not physics's synthetic crack shape) as an explicit shape signal fixes that -- before
reaching for a bigger base model (Part 4).

What's real here, not pseudocode:
- `lllyasviel/control_v11p_sd15_scribble` (public, non-gated) is a real, standard
  ControlNet, paired with the plain (non-inpainting) `stable-diffusion-v1-5/
  stable-diffusion-v1-5` base via diffusers' own `StableDiffusionControlNetInpaintPipeline`
  -- the officially documented way to combine inpainting + ControlNet conditioning.
- The base UNet stays completely frozen (the standard, intended ControlNet fine-tuning
  recipe: the base model's prior is preserved, only the ControlNet learns the new
  conditioning-specific behavior). LoRA (`r=8, alpha=16`, same attention-projection
  targets as Part 2) is injected into the ControlNet itself via `add_adapter()`.
- Every shape signal is real: `synth_xray.diffusion_data.to_scribble_pil` traces the
  boundary of a real Otsu-refined GDXray defect mask -- not a synthetic crack, not a
  bare bounding box.
- The pipeline used at inference (and by this training script's own GIF rendering)
  composites the real input pixels back in outside the mask by default
  (`VaeImageProcessor.apply_overlay`), fixing the gap Part 2's raw
  `StableDiffusionInpaintPipeline` call had.

Run:
    uv run train_controlnet_lora.py --steps 150
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
from diffusers import ControlNetModel, DDPMScheduler, StableDiffusionControlNetInpaintPipeline
from peft import LoraConfig
from PIL import Image

from synth_xray.data import download_and_extract, find_groundtruth_file, find_series_dirs
from synth_xray.diffusion_data import build_defect_dataset, to_mask_pil, to_rgb_pil, to_scribble_pil

BASE_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_ID = "lllyasviel/control_v11p_sd15_scribble"
# Same single generic prompt as Part 2 -- Welds ground truth has no per-box class.
PROMPT = "a weld radiograph with a crack defect"
CONTROLNET_CONDITIONING_SCALE = 0.7


def pil_to_tensor_pm1(img: Image.Image, device: str) -> torch.Tensor:
    """RGB PIL -> (1, 3, H, W) tensor in [-1, 1] (VAE's expected input range)."""
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def pil_to_tensor_01(img: Image.Image, device: str) -> torch.Tensor:
    """RGB PIL -> (1, 3, H, W) tensor in [0, 1] (ControlNet's conditioning-image
    range -- diffusers' own `control_image_processor` uses `do_normalize=False`,
    matched here for a manual training loop instead of the pipeline's convenience
    preprocessing)."""
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def pil_to_mask_tensor(img: Image.Image, device: str) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0  # HW, [0, 1]
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def train(steps: int, out_dir: Path, lr: float, device: str, cache_dir: Path, gif_frames_n: int) -> None:
    group_dir = download_and_extract("Welds", cache_dir)
    series_dirs = [d for d in find_series_dirs(group_dir) if find_groundtruth_file(d) is not None]
    examples = build_defect_dataset(series_dirs[0])
    print(f"{len(examples)} real GDXray defect (image, mask) pairs for ControlNet LoRA fine-tuning")

    controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        BASE_MODEL_ID, controlnet=controlnet, torch_dtype=torch.float32, safety_checker=None,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    vae, unet, text_encoder, tokenizer = pipe.vae, pipe.unet, pipe.text_encoder, pipe.tokenizer
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)  # base UNet stays fully frozen -- the ControlNet is what learns here

    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    controlnet.add_adapter(lora_config)

    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    trainable = [p for p in controlnet.parameters() if p.requires_grad]
    total = sum(p.numel() for p in controlnet.parameters())
    print(f"trainable LoRA params: {sum(p.numel() for p in trainable):,} / {total:,} total ControlNet params "
          f"({100 * sum(p.numel() for p in trainable) / total:.3f}%)")
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    with torch.no_grad():
        text_inputs = tokenizer([PROMPT], padding="max_length", max_length=tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt")
        prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]

    fixed_example = examples[0]
    fixed_image_pil = to_rgb_pil(fixed_example["image"])
    fixed_mask_pil = to_mask_pil(fixed_example["mask"])
    fixed_scribble_pil = to_scribble_pil(fixed_example["mask"])

    out_dir.mkdir(parents=True, exist_ok=True)
    losses = []
    gif_frames = [_render_frame(pipe, fixed_image_pil, fixed_mask_pil, fixed_scribble_pil, 0)]

    for step in range(steps):
        example = examples[step % len(examples)]
        image_pil = to_rgb_pil(example["image"])
        scribble_pil = to_scribble_pil(example["mask"])

        # Standard txt2img denoising training (the base UNet here takes 4 latent
        # channels only, unlike Part 2's inpainting-specific 9-channel UNet) --
        # the real target is the full real crop; masking/compositing is the
        # pipeline's job at inference time, not this training loop's.
        image_t = pil_to_tensor_pm1(image_pil, device)
        control_t = pil_to_tensor_01(scribble_pil, device)

        with torch.no_grad():
            latents = vae.encode(image_t).latent_dist.sample() * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        down_res, mid_res = controlnet(
            noisy_latents, timesteps, encoder_hidden_states=prompt_embeds,
            controlnet_cond=control_t, conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
            return_dict=False,
        )
        noise_pred = unet(
            noisy_latents, timesteps, encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res,
        ).sample
        loss = F.mse_loss(noise_pred.float(), noise.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 10 == 0 or step == steps - 1:
            print(f"step {step:3d}/{steps}  loss {loss.item():.4f}")

        frame_every = max(1, steps // gif_frames_n)
        if (step + 1) % frame_every == 0 or step == steps - 1:
            gif_frames.append(_render_frame(pipe, fixed_image_pil, fixed_mask_pil, fixed_scribble_pil, step + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("training step")
    plt.ylabel("noise-prediction MSE loss")
    plt.title("ControlNet LoRA fine-tune loss (real GDXray weld defect shapes)")
    plt.tight_layout()
    plt.savefig(out_dir / "controlnet_loss_curve.png", dpi=130)

    gif_frames[0].save(
        out_dir / "controlnet_training_progress.gif", save_all=True,
        append_images=gif_frames[1:] + [gif_frames[-1]] * 4, duration=500, loop=0,
    )
    (out_dir / "controlnet_losses.json").write_text(json.dumps(losses))

    lora_state_dict = {k: v.cpu() for k, v in controlnet.state_dict().items() if "lora" in k}
    torch.save(lora_state_dict, out_dir / "controlnet_lora_weights.pt")

    print(f"Saved controlnet_loss_curve.png + controlnet_training_progress.gif + controlnet_lora_weights.pt -> {out_dir}")


@torch.no_grad()
def _render_frame(pipe, image_pil: Image.Image, mask_pil: Image.Image, scribble_pil: Image.Image, step: int) -> Image.Image:
    pipe.controlnet.eval()
    result = pipe(
        prompt=PROMPT, image=image_pil, mask_image=mask_pil, control_image=scribble_pil,
        num_inference_steps=15, guidance_scale=1.0, controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
    ).images[0]
    pipe.controlnet.train()

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(image_pil.convert("L"), cmap="gray")
    axes[0].imshow(np.array(mask_pil) > 127, cmap="Reds", alpha=0.3)
    axes[0].set_title("input + mask", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(scribble_pil)
    axes[1].set_title("real shape (control)", fontsize=9)
    axes[1].axis("off")
    axes[2].imshow(result)
    axes[2].set_title(f"ControlNet output, step {step}", fontsize=9)
    axes[2].axis("off")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path("controlnet_out"))
    ap.add_argument("--cache-dir", type=Path, default=Path(".data_cache"))
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gif-frames", type=int, default=20)
    args = ap.parse_args()
    train(args.steps, args.out, args.lr, args.device, args.cache_dir, args.gif_frames)
