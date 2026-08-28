"""Real GRPO (Group Relative Policy Optimization) on top of the Part-1 SFT
checkpoint — the paper's Section 3.3 RL phase: "we unfreeze the VLM and
optimize it with GRPO. The reward directly reflects segmentation quality."

What's real here, not pseudocode:
- The fused-adapter head from Part 1 (SegQueryProjection + SAM2 mask
  decoder) is loaded from sft_checkpoint.pt and FROZEN — GRPO only updates
  the VLM (via LoRA, matching the paper's own Table 5: r=8, alpha=16,
  dropout=0.01).
- For each prompt, K completions are actually SAMPLED (do_sample=True) from
  the VLM, each producing a different <SEG> position, a different h_seg, and
  therefore a different predicted mask — this is the real source of policy
  variance GRPO needs.
- Reward = real Dice score of that sample's predicted mask against ground
  truth.
- Group-relative advantage: each sample's reward is normalized against its
  own group's mean/std — the defining trick of GRPO vs. plain REINFORCE (no
  learned value/critic network).
- A real PPO-style clipped surrogate objective, reusing each sampled group
  for 2 inner optimization epochs (so the clip ratio and the KL term are
  both doing real work, not identically 1.0/0.0 as they would be with a
  single-pass-per-batch REINFORCE).
- A real KL penalty against the frozen (adapter-disabled) reference policy,
  using the low-variance k3 estimator (Schulman 2020 / used in DeepSeekMath's
  GRPO formulation): KL = exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1.

Run:
    uv run grpo_train.py --bank reference_bank.pt --sft-checkpoint train_out/sft_checkpoint.pt --steps 20
"""
from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from camus_data import LABELS, download_camus, list_patients, load_frame_pair
from rau_model import RAU
from rau_modules import ReferenceBank, memory_attention, retrieve_reference

from train_rau import LABEL_NAMES, PROMPT_TEMPLATE  # must match SFT exactly, or the <SEG> policy learned there doesn't transfer

MAX_NEW_TOKENS = 16
GROUP_SIZE = 6
SAMPLE_TEMPERATURE = 1.1
SAMPLE_TOP_P = 0.98
# 60 real SFT steps (with only one trainable embedding row) are enough to
# make the model reach <SEG> reliably, but not enough to make its path
# there deterministic — sampling independently at temperature 1.1 already
# produces genuine variety in both length and wording before <SEG> ("<SEG>"
# alone, "Left<SEG>", "left ventricular myocardium<SEG>", ...), and some
# fraction of samples don't reach it at all within the budget (treated as
# reward 0 below). That natural variance is exactly what GRPO needs — no
# artificial forcing required. An earlier attempt at forcing different
# `min_new_tokens` per sample instead pushed the model into contexts its
# tiny SFT never covered, and it degenerated into refusals/garbage; this
# simpler approach is what actually works and is reported as such.
INNER_EPOCHS = 2
CLIP_EPS = 0.2


@torch.no_grad()
def dino_embed(image, dino, dino_proc, device):
    inputs = dino_proc(images=image, return_tensors="pt").to(device)
    return dino.forward(**inputs).pooler_output[0]


def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> float:
    pred = (pred_logits.sigmoid() > 0.5).float().flatten()
    target = target.flatten().float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return ((2 * inter + eps) / (union + eps)).item()


def find_seg_index(ids: torch.Tensor, seg_token_id: int) -> int | None:
    hits = (ids == seg_token_id).nonzero(as_tuple=True)[0]
    return hits[0].item() if len(hits) else None


def sequence_logprob(model, full_ids: torch.Tensor, start: int, end: int, use_adapter: bool):
    """Sum of log p(token_t | tokens_<t) for positions [start, end) under
    either the current LoRA policy or the frozen reference (adapter
    disabled). Also returns the hidden state at position end-1 (the <SEG>
    position) from the LoRA-policy pass."""
    ctx = torch.enable_grad() if use_adapter else torch.no_grad()
    with ctx:
        if use_adapter:
            out = model.vlm(input_ids=full_ids.unsqueeze(0), output_hidden_states=True)
        else:
            with model.vlm.disable_adapter():
                out = model.vlm(input_ids=full_ids.unsqueeze(0), output_hidden_states=False)
        logits = out.logits[0]  # (seq_len, vocab)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        target = full_ids[start:end]
        token_logp = log_probs[start - 1:end - 1].gather(-1, target.unsqueeze(-1)).squeeze(-1)
        h_seg = out.hidden_states[-1][0, end - 1].float() if use_adapter else None
    return token_logp.sum(), h_seg


def compute_reward(model, memory_vectors, h_seg, tgt_img, gt_mask_np, label):
    q = model.projection(h_seg.unsqueeze(0)).squeeze(0)
    z = memory_attention(q, memory_vectors)
    sam_inputs = model.sam2_processor(images=tgt_img, return_tensors="pt").to(model.vlm.device)
    with torch.no_grad():
        outputs = model.sam2(**sam_inputs, target_embedding=z.to(model.sam2.dtype).view(1, 1, 1, -1), multimask_output=False)
    pred_logits = outputs.pred_masks[0, 0].squeeze()
    gt = torch.from_numpy((gt_mask_np == label).astype(np.float32)).to(model.vlm.device)
    gt_resized = F.interpolate(gt[None, None], size=pred_logits.shape[-2:], mode="nearest")[0, 0]
    return dice_score(pred_logits, gt_resized), pred_logits


def train(bank_path, sft_ckpt, steps, out_dir, device, cache_dir, n_train_patients, offset, kl_coef):
    bank = ReferenceBank.load(bank_path).to(device)
    img_dir, mask_dir = download_camus(cache_dir)
    patients = list_patients(img_dir, mask_dir)
    train_patients = patients[offset:offset + n_train_patients]

    model = RAU()
    model.load_sft_checkpoint(sft_ckpt)
    for p in model.projection.parameters():
        p.requires_grad_(False)
    for p in model.sam2.mask_decoder.parameters():
        p.requires_grad_(False)

    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.01,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )
    model.vlm = get_peft_model(model.vlm, lora_config)
    model.vlm.print_trainable_parameters()

    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    optimizer = torch.optim.AdamW([p for p in model.vlm.parameters() if p.requires_grad], lr=1e-4)

    examples = []
    for img_path, mask_path, pid in train_patients:
        _, _, tgt_img, tgt_mask = load_frame_pair(img_path, mask_path, ref_frame=0, target_frame=-1)
        present = [c for c in np.unique(tgt_mask) if c in LABELS]
        for c in present:
            examples.append((tgt_img, tgt_mask, int(c)))
    random.Random(1).shuffle(examples)
    print(f"{len(examples)} (target image, label) GRPO training examples")

    fixed_target_img, fixed_target_mask, fixed_label = examples[0]
    fixed_ref_idx = retrieve_reference(dino_embed(fixed_target_img, dino, dino_proc, device), bank)

    out_dir.mkdir(parents=True, exist_ok=True)
    mean_rewards, kl_log, loss_log = [], [], []
    gif_frames = []

    for step in range(steps):
        tgt_img, tgt_mask, label = examples[step % len(examples)]
        ref_idx = retrieve_reference(dino_embed(tgt_img, dino, dino_proc, device), bank)
        memory_vectors = bank.memory_vectors[ref_idx]
        if label not in memory_vectors:
            continue
        ref_img_path, ref_mask_path, _ = patients[ref_idx]
        ref_img, _, _, _ = load_frame_pair(ref_img_path, ref_mask_path, ref_frame=0, target_frame=0)

        prompt = PROMPT_TEMPLATE.format(label=LABEL_NAMES[label])
        messages = [{"role": "user", "content": [
            {"type": "image", "image": ref_img}, {"type": "image", "image": tgt_img},
            {"type": "text", "text": prompt},
        ]}]
        chat_text = model.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = model.vlm_processor(text=[chat_text], images=[ref_img, tgt_img], return_tensors="pt").to(model.vlm.device)
        prompt_len = inputs["input_ids"].shape[1]

        samples = []
        with torch.no_grad():
            for _ in range(GROUP_SIZE):
                gen = model.vlm.generate(
                    **inputs, do_sample=True, temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P,
                    max_new_tokens=MAX_NEW_TOKENS,
                    # Only <SEG> stops generation — the model's ordinary
                    # <|im_end|> was trained (by pretraining, not by us) far
                    # more strongly than our 60-step <SEG> signal, so leaving
                    # it in the stop list means the model reliably ends its
                    # turn normally before ever reaching <SEG>.
                    eos_token_id=[model.seg_token_id],
                    pad_token_id=model.vlm_processor.tokenizer.pad_token_id or model.vlm_processor.tokenizer.eos_token_id,
                )
                row = gen[0]
                seg_idx = find_seg_index(row[prompt_len:], model.seg_token_id)
                if seg_idx is None:
                    continue  # this rollout never reached <SEG> within the budget — dropped, not rewarded
                samples.append(row[:prompt_len + seg_idx + 1])
        if len(samples) < 2:
            continue  # need at least 2 for a meaningful group

        # Old (sampling-time) log-probs + rewards, computed once, no grad.
        rewards, logp_old_list = [], []
        for ids in samples:
            logp_old, h_seg = sequence_logprob(model, ids, prompt_len, ids.shape[0], use_adapter=True)
            with torch.no_grad():
                r, _ = compute_reward(model, memory_vectors, h_seg.detach(), tgt_img, tgt_mask, label)
            rewards.append(r)
            logp_old_list.append(logp_old.detach())

        rewards_t = torch.tensor(rewards, device=device)
        advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-4)

        step_loss, step_kl = 0.0, 0.0
        for _inner in range(INNER_EPOCHS):
            optimizer.zero_grad()
            total_policy_loss, total_kl = 0.0, 0.0
            for ids, logp_old, adv in zip(samples, logp_old_list, advantages):
                logp_new, _ = sequence_logprob(model, ids, prompt_len, ids.shape[0], use_adapter=True)
                logp_ref, _ = sequence_logprob(model, ids, prompt_len, ids.shape[0], use_adapter=False)

                ratio = torch.exp(logp_new - logp_old)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
                policy_loss = -torch.min(surr1, surr2)

                log_ratio_ref = logp_ref - logp_new
                kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1  # k3 estimator, always >= 0

                total_policy_loss = total_policy_loss + policy_loss
                total_kl = total_kl + kl

            n = len(samples)
            loss = total_policy_loss / n + kl_coef * (total_kl / n)
            loss.backward()
            optimizer.step()
            step_loss = loss.item()
            step_kl = (total_kl / n).item()

        mean_rewards.append(rewards_t.mean().item())
        kl_log.append(step_kl)
        loss_log.append(step_loss)

        print(f"step {step:3d}/{steps}  mean_dice_reward {rewards_t.mean().item():.4f}  "
              f"loss {step_loss:.4f}  kl {step_kl:.4f}  n_samples {len(samples)}  label={LABEL_NAMES[label]}")

        if step % max(1, steps // 20) == 0 or step == steps - 1:
            gif_frames.append(_render_frame(model, bank, patients, fixed_ref_idx, fixed_target_img, fixed_label, step))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(mean_rewards)
    axes[0].set_xlabel("GRPO step"); axes[0].set_ylabel("mean group Dice reward")
    axes[0].set_title("Reward")
    axes[1].plot(kl_log, label="KL vs. SFT reference")
    axes[1].plot(loss_log, label="policy loss")
    axes[1].set_xlabel("GRPO step"); axes[1].legend()
    axes[1].set_title("Loss / KL")
    plt.tight_layout()
    plt.savefig(out_dir / "grpo_curves.png", dpi=130)

    if gif_frames:
        gif_frames[0].save(
            out_dir / "grpo_progress.gif", save_all=True,
            append_images=gif_frames[1:] + [gif_frames[-1]] * 4, duration=500, loop=0,
        )
    (out_dir / "grpo_log.json").write_text(json.dumps({"reward": mean_rewards, "kl": kl_log, "loss": loss_log}))
    print(f"Saved grpo_curves.png + grpo_progress.gif -> {out_dir}")


@torch.no_grad()
def _render_frame(model, bank, patients, ref_idx, target_img, label, step) -> Image.Image:
    ref_img_path, ref_mask_path, _ = patients[ref_idx]
    ref_img, _, _, _ = load_frame_pair(ref_img_path, ref_mask_path, ref_frame=0, target_frame=0)
    prompt = PROMPT_TEMPLATE.format(label=LABEL_NAMES[label])
    h_seg = model.get_seg_hidden_state(ref_img, target_img, prompt)
    q = model.projection(h_seg.unsqueeze(0)).squeeze(0)
    z = memory_attention(q, bank.memory_vectors[ref_idx])

    sam_inputs = model.sam2_processor(images=target_img, return_tensors="pt").to(model.vlm.device)
    target_embedding = z.to(model.sam2.dtype).view(1, 1, 1, -1)
    outputs = model.sam2(**sam_inputs, target_embedding=target_embedding, multimask_output=False)
    pred = outputs.pred_masks[0, 0].squeeze().sigmoid().float().cpu().numpy()
    pred_up = np.array(Image.fromarray((pred * 255).astype(np.uint8)).resize(target_img.size)) > 127

    base = np.array(target_img).astype(np.float32)
    overlay = base.copy()
    overlay[pred_up] = overlay[pred_up] * 0.4 + np.array([255, 60, 60]) * 0.6
    frame = Image.fromarray(overlay.astype(np.uint8))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(frame)
    ax.set_title(f"GRPO step {step}", fontsize=10)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=Path("reference_bank.pt"))
    ap.add_argument("--sft-checkpoint", type=Path, default=Path("train_out/sft_checkpoint.pt"))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("grpo_out"))
    ap.add_argument("--cache-dir", type=Path, default=Path("camus_cache"))
    ap.add_argument("--n-train-patients", type=int, default=40)
    ap.add_argument("--offset", type=int, default=70, help="disjoint from both reference-bank and SFT patients")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--kl-coef", type=float, default=0.08)
    args = ap.parse_args()
    train(args.bank, args.sft_checkpoint, args.steps, args.out, args.device, args.cache_dir, args.n_train_patients, args.offset, args.kl_coef)
