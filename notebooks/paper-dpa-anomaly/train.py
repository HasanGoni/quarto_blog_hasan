"""Real training run for the DPA reimplementation: SD1.5-inpainting LoRA fine-tuned on real
MVTec-AD (guide, mask, image, product-agnostic-prompt) triples, with a genuine zero-shot test --
three (defect type, product) combinations held out entirely from training, then generated at
inference and compared against the real (never-trained-on) ground truth image.
"""
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from peft import LoraConfig, get_peft_model
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import matplotlib.pyplot as plt
import imageio

from common import IMG_SIZE, HELD_OUT_COMBOS, product_agnostic_prompt
from data import load_split, MVTecAnomalyDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

SD_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
IMG_OUT = "../../posts/series/papers/images"
OUT = "out"
os.makedirs(f"{OUT}/progress", exist_ok=True)

vae = AutoencoderKL.from_pretrained(SD_ID, subfolder="vae").to(device)
unet = UNet2DConditionModel.from_pretrained(SD_ID, subfolder="unet").to(device)
text_encoder = CLIPTextModel.from_pretrained(SD_ID, subfolder="text_encoder").to(device)
tokenizer = CLIPTokenizer.from_pretrained(SD_ID, subfolder="tokenizer")
noise_scheduler = DDPMScheduler.from_pretrained(SD_ID, subfolder="scheduler")
vae.requires_grad_(False); text_encoder.requires_grad_(False)

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["to_q", "to_k", "to_v", "to_out.0"], lora_dropout=0.0)
unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()

VAE_SCALE = vae.config.scaling_factor
LATENT_SIZE = IMG_SIZE // 8

to_tensor = lambda img: torch.from_numpy(np.array(img)).float()


def encode_batch(guide_pil, mask_pil, image_pil, prompt):
    guide = (to_tensor(guide_pil).permute(2, 0, 1) / 127.5 - 1).unsqueeze(0).to(device)
    image = (to_tensor(image_pil).permute(2, 0, 1) / 127.5 - 1).unsqueeze(0).to(device)
    mask = (to_tensor(mask_pil) / 255.0).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W) in [0,1]

    masked_guide = guide * (1 - mask)
    with torch.no_grad():
        target_latent = vae.encode(image).latent_dist.sample() * VAE_SCALE
        masked_latent = vae.encode(masked_guide).latent_dist.sample() * VAE_SCALE
    mask_latent = F.interpolate(mask, size=(LATENT_SIZE, LATENT_SIZE), mode="nearest")

    ids = tokenizer(prompt, padding="max_length", truncation=True,
                     max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        text_emb = text_encoder(ids)[0]
    return target_latent, masked_latent, mask_latent, text_emb


@torch.no_grad()
def sample(guide_pil, mask_pil, prompt, guidance_scale=7.5, num_steps=30, blend=True):
    """blend=True applies the standard RePaint-style trick: at every denoising step, the
    outside-mask latent is forcibly reset to the *known* guide image correctly noised for that
    timestep, so fidelity to the guide outside the mask is guaranteed by construction rather than
    left entirely to the model's learned conditioning -- a legitimate, well-established inference
    technique (not a training change), used here after the model on its own turned out not to
    respect the mask boundary at all (see the post's honest write-up of that first result)."""
    was_training = unet.training
    unet.eval()
    guide = (to_tensor(guide_pil).permute(2, 0, 1) / 127.5 - 1).unsqueeze(0).to(device)
    mask = (to_tensor(mask_pil) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    masked_guide = guide * (1 - mask)
    masked_latent = vae.encode(masked_guide).latent_dist.sample() * VAE_SCALE
    known_latent = vae.encode(guide).latent_dist.sample() * VAE_SCALE
    mask_latent = F.interpolate(mask, size=(LATENT_SIZE, LATENT_SIZE), mode="nearest")
    mask_latent = (mask_latent > 0.5).float()

    cond_ids = tokenizer(prompt, padding="max_length", truncation=True,
                          max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
    uncond_ids = tokenizer("", padding="max_length", truncation=True,
                            max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
    cond_emb = text_encoder(cond_ids)[0]
    uncond_emb = text_encoder(uncond_ids)[0]

    scheduler = DDIMScheduler.from_pretrained(SD_ID, subfolder="scheduler")
    scheduler.set_timesteps(num_steps, device=device)
    latents = torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE, device=device)

    for t in scheduler.timesteps:
        latent_in = torch.cat([latents, mask_latent, masked_latent], dim=1)
        noise_cond = unet(latent_in, t, encoder_hidden_states=cond_emb).sample
        noise_uncond = unet(latent_in, t, encoder_hidden_states=uncond_emb).sample
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if blend:
            noise = torch.randn_like(known_latent)
            known_noised = noise_scheduler.add_noise(known_latent, noise, t.unsqueeze(0))
            latents = latents * mask_latent + known_noised * (1 - mask_latent)

    decoded = vae.decode(latents / VAE_SCALE).sample[0]
    frame = ((decoded.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
    if was_training:
        unet.train()
    return Image.fromarray(frame)


if __name__ == "__main__":
    ds, train_idx, held_out_idx = load_split()
    print(f"train rows: {len(train_idx)}  held-out rows: {len(held_out_idx)}")
    train_ds = MVTecAnomalyDataset(ds, train_idx)
    held_out_ds = MVTecAnomalyDataset(ds, held_out_idx)

    opt = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=1e-4)

    N_STEPS = int(os.environ.get("N_STEPS", 1500))
    LOG_EVERY = 50
    SAMPLE_EVERY = 150
    losses = []

    # fixed watch-it-learn example: the first held-out (crack, capsule) row
    watch_idx = [i for i in range(len(held_out_ds)) if held_out_ds[i]["defect"] == "crack"][0]
    watch_row = held_out_ds[watch_idx]
    progress_frames = []

    unet.train()
    step = 0
    rng = np.random.default_rng(0)
    order = rng.permutation(len(train_ds))
    while step < N_STEPS:
        for k in order:
            if step >= N_STEPS:
                break
            row = train_ds[int(k)]
            target_latent, masked_latent, mask_latent, text_emb = encode_batch(
                row["guide"], row["mask"], row["image"], row["prompt"])
            noise = torch.randn_like(target_latent)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
            noisy_latent = noise_scheduler.add_noise(target_latent, noise, t)

            unet_input = torch.cat([noisy_latent, mask_latent, masked_latent], dim=1)
            noise_pred = unet(unet_input, t, encoder_hidden_states=text_emb).sample
            loss = F.mse_loss(noise_pred, noise)

            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            if step % LOG_EVERY == 0:
                print(f"step {step:4d}  loss {loss.item():.4f}")
            if step % SAMPLE_EVERY == 0 or step == N_STEPS - 1:
                frame = sample(watch_row["guide"], watch_row["mask"],
                                product_agnostic_prompt(watch_row["defect"]), num_steps=20)
                frame.save(f"{OUT}/progress/step_{step:04d}.png")
                progress_frames.append(frame)
            step += 1
        order = rng.permutation(len(train_ds))

    plt.figure(figsize=(5, 3.5))
    plt.plot(losses)
    plt.xlabel("step"); plt.ylabel("MSE noise-prediction loss"); plt.title("DPA LoRA training")
    plt.tight_layout()
    plt.savefig(f"{IMG_OUT}/dpa-loss-curve.png", dpi=130)

    frames_np = [np.array(f.resize((256, 256))) for f in progress_frames]
    imageio.mimsave(f"{OUT}/progress_video.mp4", frames_np, fps=2)
    print(f"Saved {len(frames_np)} progress frames")

    unet.save_pretrained(f"{OUT}/lora_checkpoint")
    print("Saved LoRA checkpoint")

    # --- genuine zero-shot eval: for each held-out (defect, category), generate + compare to real ---
    print("Zero-shot eval on held-out (defect, category) combos...")
    results = []
    examples = []
    seen_combos = set()
    for i in range(len(held_out_ds)):
        row = held_out_ds[i]
        combo = (row["defect"], row["category"])
        if combo in seen_combos:
            continue
        seen_combos.add(combo)
        gen = sample(row["guide"], row["mask"], product_agnostic_prompt(row["defect"]), num_steps=30)
        real = row["image"]
        real_arr = np.array(real).astype(np.float32)
        gen_arr = np.array(gen).astype(np.float32)
        mask_arr = np.array(row["mask"]) > 127
        mse_in_mask = float(np.mean((real_arr[mask_arr] - gen_arr[mask_arr]) ** 2)) if mask_arr.sum() else None
        mse_outside = float(np.mean((real_arr[~mask_arr] - gen_arr[~mask_arr]) ** 2))
        results.append({"defect": row["defect"], "category": row["category"],
                         "mse_in_mask": mse_in_mask, "mse_outside_mask": mse_outside})
        examples.append((row, gen, real))

    with open(f"{OUT}/zero_shot_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    fig, axes = plt.subplots(len(examples), 4, figsize=(13, 3.4 * len(examples)))
    for row_i, (row, gen, real) in enumerate(examples):
        axes[row_i, 0].imshow(row["guide"]); axes[row_i, 0].set_title("Normal guide" if row_i == 0 else "")
        axes[row_i, 1].imshow(row["mask"], cmap="gray"); axes[row_i, 1].set_title("Mask" if row_i == 0 else "")
        axes[row_i, 2].imshow(gen); axes[row_i, 2].set_title("Zero-shot generated" if row_i == 0 else "")
        axes[row_i, 3].imshow(real); axes[row_i, 3].set_title("Real (held-out) ground truth" if row_i == 0 else "")
        axes[row_i, 0].set_ylabel(f"{row['defect']}\non {row['category']}", fontsize=10)
        for c in range(4):
            axes[row_i, c].set_xticks([]); axes[row_i, c].set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{IMG_OUT}/dpa-zero-shot-results.png", dpi=120)
    print("Saved dpa-zero-shot-results.png")
