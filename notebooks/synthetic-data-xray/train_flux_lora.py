"""Real LoRA fine-tune of FLUX.1-Fill-dev on real GDXray weld defect crops --
Synthetic Data series Part 4.

Parts 2-3 both fine-tuned SD1.5-family UNets (LoRA on attention layers, DDPM
noise-prediction training). This part swaps the base generator for a genuinely
different architecture -- Flux is a 12B-parameter MMDiT (multi-modal diffusion
transformer, dual image/text streams, flow-matching objective, not a UNet at
all) -- to test whether base-model scale/architecture closes the shape-realism
gap Part 2 found, independent of (or complementary to) Part 3's ControlNet
shape-conditioning approach.

Real, not pseudocode:
- `black-forest-labs/FLUX.1-Fill-dev` is gated; access was confirmed via the
  HF API before committing to this approach (see the post for the full
  Flux-vs-SDXL-vs-PixArt tradeoff this decision came from).
- LoRA (`r=8, alpha=16`) targets the *real* attention-projection module names
  found by inspecting `transformer.named_modules()` directly: `to_q/to_k/
  to_v/to_out.0` (the image stream) and `add_q_proj/add_k_proj/add_v_proj/
  to_add_out` (the text stream) -- Flux's dual-stream MMDiT blocks have both.
- The training loop reuses the pipeline's own real internal methods
  (`encode_prompt`, `prepare_mask_latents`, `_encode_vae_image`,
  `_pack_latents`, `_prepare_latent_image_ids`) rather than re-deriving
  Flux's packed-latent-sequence format by hand -- the same methods
  `FluxFillPipeline.__call__` itself uses at inference time, so the training
  data flow matches real inference exactly.
- A real flow-matching (rectified flow) training objective, not DDPM: sample
  `sigma ~ U(0,1)`, interpolate `noisy = sigma * noise + (1 - sigma) *
  real_latents` (`FlowMatchEulerDiscreteScheduler.scale_noise`'s own real
  formula), target velocity `= noise - real_latents`, MSE against the
  transformer's real prediction.

Run:
    uv run train_flux_lora.py --steps 100
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
from diffusers import FluxFillPipeline
from peft import LoraConfig
from PIL import Image

from synth_xray.data import download_and_extract, find_groundtruth_file, find_series_dirs
from synth_xray.diffusion_data import build_defect_dataset, to_mask_pil, to_rgb_pil

MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
PROMPT = "a weld radiograph with a crack defect"
GUIDANCE_SCALE = 30.0  # FluxFillPipeline's own default
CROP_SIZE = 512


def train(steps: int, out_dir: Path, lr: float, device: str, cache_dir: Path, gif_frames_n: int) -> None:
    group_dir = download_and_extract("Welds", cache_dir)
    series_dirs = [d for d in find_series_dirs(group_dir) if find_groundtruth_file(d) is not None]
    examples = build_defect_dataset(series_dirs[0])
    print(f"{len(examples)} real GDXray defect (image, mask) pairs for Flux LoRA fine-tuning")

    pipe = FluxFillPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    vae, transformer = pipe.vae, pipe.transformer
    vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    transformer.requires_grad_(False)

    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"],
    )
    transformer.add_adapter(lora_config)

    trainable = [p for p in transformer.parameters() if p.requires_grad]
    total = sum(p.numel() for p in transformer.parameters())
    print(f"trainable LoRA params: {sum(p.numel() for p in trainable):,} / {total:,} total transformer params "
          f"({100 * sum(p.numel() for p in trainable) / total:.3f}%)")
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    dtype = transformer.dtype
    num_channels_latents = pipe.latent_channels  # 16 for Flux
    vae_sf = pipe.vae_scale_factor  # 8

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=PROMPT, prompt_2=PROMPT, device=device, num_images_per_prompt=1, max_sequence_length=512,
        )

    def prepare_step_inputs(image_pil: Image.Image, mask_pil: Image.Image):
        """Real preprocessing, matching FluxFillPipeline.__call__ exactly: pipe's own
        image_processor/mask_processor, pipe's own prepare_mask_latents, pipe's own
        _encode_vae_image + _pack_latents for the real target latents."""
        init_image = pipe.image_processor.preprocess(image_pil, height=CROP_SIZE, width=CROP_SIZE).to(device, dtype)
        mask_image = pipe.mask_processor.preprocess(mask_pil, height=CROP_SIZE, width=CROP_SIZE).to(device, dtype)
        masked_image = init_image * (1 - mask_image)

        with torch.no_grad():
            mask, masked_image_latents = pipe.prepare_mask_latents(
                mask_image, masked_image, 1, num_channels_latents, 1, CROP_SIZE, CROP_SIZE, dtype, device, None,
            )
            masked_image_latents_cat = torch.cat((masked_image_latents, mask), dim=-1)
            real_latents = pipe._encode_vae_image(image=init_image, generator=None)

        lat_h = 2 * (CROP_SIZE // (vae_sf * 2))
        lat_w = 2 * (CROP_SIZE // (vae_sf * 2))
        latent_image_ids = pipe._prepare_latent_image_ids(1, lat_h // 2, lat_w // 2, device, dtype)
        return real_latents, masked_image_latents_cat, latent_image_ids, lat_h, lat_w

    guidance = None
    if transformer.config.guidance_embeds:
        guidance = torch.full([1], GUIDANCE_SCALE, device=device, dtype=torch.float32)

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

        real_latents, masked_image_latents_cat, latent_image_ids, lat_h, lat_w = prepare_step_inputs(image_pil, mask_pil)

        noise = torch.randn_like(real_latents)
        sigma = torch.rand(1, device=device, dtype=torch.float32)
        sigma_b = sigma.to(dtype).view(-1, 1, 1, 1)
        noisy_latents = sigma_b * noise + (1 - sigma_b) * real_latents  # real FlowMatchEulerDiscreteScheduler.scale_noise formula

        packed_noisy = pipe._pack_latents(noisy_latents, 1, num_channels_latents, lat_h, lat_w)
        packed_target = pipe._pack_latents(noise - real_latents, 1, num_channels_latents, lat_h, lat_w)

        model_pred = transformer(
            hidden_states=torch.cat((packed_noisy, masked_image_latents_cat), dim=2),
            timestep=sigma.to(dtype),
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        loss = F.mse_loss(model_pred.float(), packed_target.float())

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
    plt.ylabel("flow-matching (velocity) MSE loss")
    plt.title("Flux LoRA fine-tune loss (real GDXray weld defects)")
    plt.tight_layout()
    plt.savefig(out_dir / "flux_loss_curve.png", dpi=130)

    gif_frames[0].save(
        out_dir / "flux_training_progress.gif", save_all=True,
        append_images=gif_frames[1:] + [gif_frames[-1]] * 4, duration=500, loop=0,
    )
    (out_dir / "flux_losses.json").write_text(json.dumps(losses))

    lora_state_dict = {k: v.cpu() for k, v in transformer.state_dict().items() if "lora" in k}
    torch.save(lora_state_dict, out_dir / "flux_lora_weights.pt")

    print(f"Saved flux_loss_curve.png + flux_training_progress.gif + flux_lora_weights.pt -> {out_dir}")


@torch.no_grad()
def _render_frame(pipe, image_pil: Image.Image, mask_pil: Image.Image, step: int) -> Image.Image:
    pipe.transformer.eval()
    result = pipe(
        prompt=PROMPT, image=image_pil, mask_image=mask_pil,
        num_inference_steps=15, guidance_scale=GUIDANCE_SCALE, height=CROP_SIZE, width=CROP_SIZE,
    ).images[0]
    pipe.transformer.train()

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(image_pil.convert("L"), cmap="gray")
    axes[0].imshow(np.array(mask_pil) > 127, cmap="Reds", alpha=0.3)
    axes[0].set_title("input + mask", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(result)
    axes[1].set_title(f"Flux output, step {step}", fontsize=9)
    axes[1].axis("off")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("flux_out"))
    ap.add_argument("--cache-dir", type=Path, default=Path(".data_cache"))
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gif-frames", type=int, default=15)
    args = ap.parse_args()
    train(args.steps, args.out, args.lr, args.device, args.cache_dir, args.gif_frames)
