"""Real training run for the InstEditSeg reimplementation.

LoRA fine-tunes SD1.5's U-Net attention layers, the newly-expanded conv_in (image conditioning),
and the DINO Feature Guidance Block's zero-init projections -- everything else (VAE, text
encoder, DINO backbone) stays frozen. Trains on real Kvasir-SEG polyp images to render the
instruction-conditioned red overlay described in common.render_overlay.
"""
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt

from common import IMG_SIZE, INSTRUCTION, dice_score, extract_predicted_mask
from data import KvasirEditDataset
from model import load_models, DinoFeatureGuidance, dino_spatial_features

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

OUT = "out"
IMG_OUT = "../../posts/series/papers/images"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/progress", exist_ok=True)

print("Loading models...")
vae, unet, text_encoder, tokenizer, noise_scheduler, dino_model = load_models(device)

lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.0,
)
unet = get_peft_model(unet, lora_config)
unet.base_model.model.conv_in.requires_grad_(True)  # new image-conditioning channels: full fine-tune
unet.to(device)

adapter = DinoFeatureGuidance().to(device)

n_trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad) + \
              sum(p.numel() for p in adapter.parameters())
print(f"Trainable params: {n_trainable:,}")

print("Loading Kvasir-SEG...")
train_ds = KvasirEditDataset("train", tokenizer)
val_ds = KvasirEditDataset("validation", tokenizer)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, drop_last=True)

opt = torch.optim.AdamW(
    [p for p in unet.parameters() if p.requires_grad] + list(adapter.parameters()), lr=1e-4,
)

VAE_SCALE = vae.config.scaling_factor
LATENT_SIZE = IMG_SIZE // 8


@torch.no_grad()
def encode_latent(vae, pixel_values):
    return vae.encode(pixel_values).latent_dist.sample() * VAE_SCALE


@torch.no_grad()
def sample(unet, vae, text_encoder, tokenizer, scheduler, dino_model, adapter,
           original_image: Image.Image, instruction: str, guidance_scale=7.5, num_steps=30):
    """Dual-branch classifier-free guidance: one conditioned pass (instruction text + DINO
    features + original-image latent) and one unconditional pass (empty text, zeroed DINO
    features, zeroed image-conditioning channels) per denoising step -- two forward passes,
    matching the paper's stated inference efficiency."""
    was_training = unet.training
    unet.eval()
    img = original_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    img_latent_in = encode_latent(vae, img_t * 2 - 1)

    dino_grid = dino_spatial_features(dino_model, img_t)
    cond_residuals = adapter(dino_grid, LATENT_SIZE)
    uncond_residuals = [torch.zeros_like(r) for r in cond_residuals]

    cond_ids = tokenizer(instruction, padding="max_length", truncation=True,
                          max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
    uncond_ids = tokenizer("", padding="max_length", truncation=True,
                           max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
    cond_emb = text_encoder(cond_ids)[0]
    uncond_emb = text_encoder(uncond_ids)[0]

    scheduler.set_timesteps(num_steps, device=device)
    latents = torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE, device=device)
    frames = []

    for t in scheduler.timesteps:
        latent_in_cond = torch.cat([latents, img_latent_in], dim=1)
        latent_in_uncond = torch.cat([latents, torch.zeros_like(img_latent_in)], dim=1)

        noise_cond = unet(latent_in_cond, t, encoder_hidden_states=cond_emb,
                           down_intrablock_additional_residuals=list(cond_residuals)).sample
        noise_uncond = unet(latent_in_uncond, t, encoder_hidden_states=uncond_emb,
                             down_intrablock_additional_residuals=list(uncond_residuals)).sample
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample
        decoded = vae.decode(latents / VAE_SCALE).sample[0]
        frame = ((decoded.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
        frames.append(Image.fromarray(frame))

    if was_training:
        unet.train()
    return frames[-1], frames


# fixed held-out example used for the "watch it learn" progression + final qualitative figure
fixed_idx = 3
fixed_row = val_ds.ds[fixed_idx]
fixed_image = fixed_row["image"].convert("RGB").resize((IMG_SIZE, IMG_SIZE))
fixed_mask = fixed_row["annotation"]

N_STEPS = 2200
LOG_EVERY = 50
SAMPLE_EVERY = 200
losses = []
progress_frames = []

print(f"Training for {N_STEPS} steps...")
step = 0
unet.train()
while step < N_STEPS:
    for batch in train_loader:
        if step >= N_STEPS:
            break
        original = batch["original"].to(device)
        original_0_1 = batch["original_0_1"].to(device)
        target = batch["target"].to(device)
        input_ids = batch["input_ids"].to(device)

        target_latent = encode_latent(vae, target)
        original_latent = encode_latent(vae, original)
        noise = torch.randn_like(target_latent)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                   (target_latent.shape[0],), device=device).long()
        noisy_latent = noise_scheduler.add_noise(target_latent, noise, timesteps)

        text_emb = text_encoder(input_ids)[0]
        dino_grid = dino_spatial_features(dino_model, original_0_1)
        residuals = adapter(dino_grid, LATENT_SIZE)

        unet_input = torch.cat([noisy_latent, original_latent], dim=1)
        noise_pred = unet(unet_input, timesteps, encoder_hidden_states=text_emb,
                           down_intrablock_additional_residuals=list(residuals)).sample

        loss = F.mse_loss(noise_pred, noise)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

        if step % LOG_EVERY == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")
        if step % SAMPLE_EVERY == 0 or step == N_STEPS - 1:
            final_frame, _ = sample(unet, vae, text_encoder, tokenizer, noise_scheduler,
                                     dino_model, adapter, fixed_image, INSTRUCTION,
                                     guidance_scale=7.5, num_steps=20)
            final_frame.save(f"{OUT}/progress/step_{step:04d}.png")
            progress_frames.append((step, final_frame))
            unet.train()
        step += 1

print("Training done. Saving outputs...")

plt.figure(figsize=(5, 3.5))
plt.plot(losses)
plt.xlabel("step"); plt.ylabel("MSE noise-prediction loss"); plt.title("InstEditSeg LoRA training")
plt.tight_layout()
plt.savefig(f"{IMG_OUT}/insteditseg-loss-curve.png", dpi=130)

with open(f"{OUT}/losses.json", "w") as f:
    json.dump(losses, f)

# progress GIF/video -> "watch it learn" on the fixed held-out example
import imageio
frames_np = [np.array(f.resize((256, 256))) for _, f in progress_frames]
imageio.mimsave(f"{OUT}/progress_video.mp4", frames_np, fps=3)
print(f"Saved {len(frames_np)} progress frames to progress_video.mp4")

# final qualitative results on a few held-out validation examples
results = []
fig, axes = plt.subplots(4, 4, figsize=(13, 13))
for row_idx, val_idx in enumerate([2, 3, 7, 15]):
    row = val_ds.ds[val_idx]
    image = row["image"].convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    mask = row["annotation"]
    gt_overlay = row["image"].convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    from common import render_overlay
    gt_target = render_overlay(image, mask)

    final_frame, all_frames = sample(unet, vae, text_encoder, tokenizer, noise_scheduler,
                                      dino_model, adapter, image, INSTRUCTION,
                                      guidance_scale=7.5, num_steps=30)
    pred_mask = extract_predicted_mask(final_frame, image)
    dice = dice_score(pred_mask, mask)
    results.append({"val_idx": val_idx, "dice": dice})

    axes[row_idx, 0].imshow(image); axes[row_idx, 0].set_title("Input" if row_idx == 0 else "")
    axes[row_idx, 1].imshow(gt_target); axes[row_idx, 1].set_title("Ground-truth overlay" if row_idx == 0 else "")
    axes[row_idx, 2].imshow(final_frame); axes[row_idx, 2].set_title(f"Generated" if row_idx == 0 else "")
    axes[row_idx, 2].set_xlabel(f"dice~{dice:.2f}")
    axes[row_idx, 3].imshow(pred_mask, cmap="gray"); axes[row_idx, 3].set_title("Extracted pred. mask" if row_idx == 0 else "")
    for c in range(4):
        axes[row_idx, c].set_xticks([]); axes[row_idx, c].set_yticks([])

plt.tight_layout()
plt.savefig(f"{IMG_OUT}/insteditseg-qualitative-results.png", dpi=130)

with open(f"{OUT}/dice_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
print("mean dice:", np.mean([r["dice"] for r in results]))

# denoising-process visualization for ELI5/visual-understanding: intermediate steps of ONE example
_, all_frames = sample(unet, vae, text_encoder, tokenizer, noise_scheduler, dino_model, adapter,
                        fixed_image, INSTRUCTION, guidance_scale=7.5, num_steps=20)
fig, axes = plt.subplots(1, 6, figsize=(16, 3))
show_idxs = [0, 3, 6, 10, 15, 19]
for ax, idx in zip(axes, show_idxs):
    ax.imshow(all_frames[idx]); ax.set_title(f"step {idx+1}/20", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(f"{IMG_OUT}/insteditseg-denoising-steps.png", dpi=130)

print("All outputs saved.")
