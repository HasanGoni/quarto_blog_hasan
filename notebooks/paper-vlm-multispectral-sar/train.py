"""Real LoRA fine-tuning of Qwen2-VL-2B-Instruct to classify local climate zones from 5 named
optical views + 1 named SAR view, using its native multi-image chat interface -- the paper's
core trick, no new encoder.
"""
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import matplotlib.pyplot as plt

from common import LCZ_CLASSES
from data import So2SatViewsDataset, stratified_split
import h5py

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
IMG_OUT = "../../posts/series/papers/images"
OUT = "out"
os.makedirs(OUT, exist_ok=True)

CLASS_LIST_STR = ", ".join(LCZ_CLASSES)
QUESTION = (f"These are 5 optical satellite views and 1 SAR (radar) view of the same location. "
            f"Which local climate zone best matches? Answer with exactly one class from this "
            f"list, nothing else: {CLASS_LIST_STR}.")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj", "qkv"], lora_dropout=0.0)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def build_messages(views):
    content = []
    for name, img in views:
        content.append({"type": "text", "text": f"[{name}]"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": QUESTION})
    return [{"role": "user", "content": content}]


def prepare_batch(views, answer: str | None):
    messages = build_messages(views)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    prompt_inputs = processor(text=[prompt_text], images=image_inputs, padding=True, return_tensors="pt")
    prompt_len = prompt_inputs["input_ids"].shape[1]

    if answer is None:
        return prompt_inputs, prompt_len

    full_text = prompt_text + answer + processor.tokenizer.eos_token
    full_inputs = processor(text=[full_text], images=image_inputs, padding=True, return_tensors="pt")
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    full_inputs["labels"] = labels
    return full_inputs, prompt_len


@torch.no_grad()
def predict(views) -> str:
    was_training = model.training
    model.eval()
    inputs, _ = prepare_batch(views, answer=None)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    new_tokens = gen[:, inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
    if was_training:
        model.train()
    return text


def nearest_class(pred_text: str) -> str:
    pred_low = pred_text.lower()
    for c in LCZ_CLASSES:
        if c.lower() in pred_low:
            return c
    return pred_text  # no match -- counted wrong


if __name__ == "__main__":
    f = h5py.File("data/v4/validation.h5", "r")
    labels_all = np.array(f["label"])
    f.close()

    train_idx, val_idx = stratified_split(labels_all, n_train_per_class=100, n_val_per_class=15)
    print(f"train examples: {len(train_idx)}  val examples: {len(val_idx)}")
    train_ds = So2SatViewsDataset(train_idx)
    val_ds = So2SatViewsDataset(val_idx)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    N_STEPS = int(os.environ.get("N_STEPS", 1500))
    ACCUM = 4
    LOG_EVERY = 50
    losses = []

    model.train()
    step = 0
    order = np.random.default_rng(1).permutation(len(train_ds))
    while step < N_STEPS:
        for k in order:
            if step >= N_STEPS:
                break
            views, cls_name = train_ds[int(k)]
            inputs, _ = prepare_batch(views, answer=cls_name)
            inputs = {key: v.to(device) for key, v in inputs.items()}
            out = model(**inputs)
            loss = out.loss / ACCUM
            loss.backward()
            if (step + 1) % ACCUM == 0:
                opt.step(); opt.zero_grad()
            losses.append(out.loss.item())
            if step % LOG_EVERY == 0:
                print(f"step {step:4d}  loss {out.loss.item():.4f}")
            step += 1
        order = np.random.default_rng(step).permutation(len(train_ds))

    plt.figure(figsize=(5, 3.5))
    plt.plot(losses)
    plt.xlabel("step"); plt.ylabel("cross-entropy loss (completion tokens)")
    plt.title("Qwen2-VL LoRA training -- LCZ classification")
    plt.tight_layout()
    plt.savefig(f"{IMG_OUT}/vlm-sar-loss-curve.png", dpi=130)

    print("Evaluating on held-out examples...")
    correct = 0
    results = []
    for i in range(len(val_ds)):
        views, true_cls = val_ds[i]
        pred_text = predict(views)
        pred_cls = nearest_class(pred_text)
        ok = pred_cls == true_cls
        correct += int(ok)
        results.append({"true": true_cls, "pred_raw": pred_text, "pred_cls": pred_cls, "correct": ok})

    acc = correct / len(val_ds)
    print(f"Held-out accuracy: {acc:.3f} ({correct}/{len(val_ds)})")
    with open(f"{OUT}/eval_results.json", "w") as fp:
        json.dump({"accuracy": acc, "n": len(val_ds), "results": results}, fp, indent=2)

    print("Done.")
