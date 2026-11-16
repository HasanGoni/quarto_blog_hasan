"""Step 2, fine-tuning. The official repo lists its fine-tuning code as "TODO", so this recipe is MINE, not the authors'.

    .venv/bin/python finetune.py --scale 2 --epochs 2 [--n-train 2800] [--tag s2]

What is trained: LoRA adapters (rank 16) on the attention and MLP projections of the LANGUAGE model only. The vision encoder, the
embeddings, the <OBJ> token and the 9.6M pixel decoder stay frozen: the task here is boxes, and boxes are text for this model
(`<ref>void</ref><box><x1><y1><x2><y2>,...</box>`, coordinates in thousandths of the image side, `None` when there is no void).
Data: the series' `train` split (2,800 images, 5,446 voids), the prompt the paper uses for detection, loss on the answer only.
Selection: after every epoch the first 100 VAL images are scored (IoU 0.2, the series' rule); the best epoch is kept. TEST is not touched here.
"""
import argparse
import json
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from wevisone_vl import WeVisOneVL
from wevisone_vl.prompts import build_prompt

import common as c

LR, ACCUM, N_VAL = 1e-4, 8, 100


def target_text(boxes: list, size: int) -> str:
    """The answer the paper's checkpoint writes: thousandths of the side, sorted top-left to bottom-right, `None` if empty."""
    if not boxes:
        return "None"
    enc = []
    for b in sorted(boxes, key=lambda b: (b[1], b[0])):
        enc.append("".join(f"<{max(0, min(1000, int(v / size * 1000)))}>" for v in b))
    return f"<ref>{c.LABEL}</ref><box>" + ",".join(enc) + "</box>"


def make_example(engine, split: str, i: int, scale: int):
    img = c.load_image(split, i, scale)
    prompt = build_prompt("detection", [c.LABEL])
    messages = engine._messages(img, prompt)
    text = engine.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    answer = target_text(c.gt_boxes(split, i), 224) + "<|im_end|>"
    full = engine.processor(text=[text + answer], images=[img], return_tensors="pt")
    plen = engine.processor(text=[text], images=[img], return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :plen] = -100
    return full, labels


@torch.no_grad()
def evaluate(engine, split: str, n: int, scale: int, batch: int = 8) -> dict:
    engine.model.eval()
    per, per_s = [], []
    for start in range(0, n, batch):
        idx = list(range(start, min(n, start + batch)))
        imgs = [c.load_image(split, i, scale) for i in idx]
        res = engine.predict_batch(imgs, task="detection", query=[c.LABEL], decode_matting=False, max_new_tokens=1024, temperature=0.0)
        for i, img, r in zip(idx, imgs, res):
            pred, gt = c.parse_boxes(r.response, img.width, img.height, scale), c.gt_boxes(split, i)
            per.append(c.score(gt, pred))
            per_s.append(c.score(gt, pred, c.HIT_STRICT))
    tot = [sum(x[k] for x in per) for k in range(3)]
    tots = [sum(x[k] for x in per_s) for k in range(3)]
    return {"iou0.2": c.prf(*tot), "iou0.5": c.prf(*tots), "boxes_per_image": round(sum(x[0] + x[1] for x in per) / n, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=2800)
    ap.add_argument("--tag", default="s2")
    a = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    c.OUT.mkdir(exist_ok=True)
    engine = WeVisOneVL("tencent/WeVisOne-VL-4B", backend="transformers")
    for p in engine.model.parameters():
        p.requires_grad = False
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(engine.model, cfg)
    model.print_trainable_parameters()
    engine.model = model
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    order = list(range(a.n_train))
    steps_total = a.epochs * math.ceil(a.n_train / ACCUM)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps_total))))
    log, best = [], None
    t0, step = time.time(), 0
    for epoch in range(a.epochs):
        random.shuffle(order)
        model.train()
        run_loss, n_loss = 0.0, 0
        for k, i in enumerate(order):
            inputs, labels = make_example(engine, "train", i, a.scale)
            inputs = {kk: v.to(engine.device) for kk, v in inputs.items()}
            out = model(**inputs, labels=labels.to(engine.device))
            (out.loss / ACCUM).backward()
            run_loss, n_loss = run_loss + out.loss.item(), n_loss + 1
            if (k + 1) % ACCUM == 0 or k + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    print(f"epoch {epoch + 1} step {step}/{steps_total} loss {run_loss / n_loss:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
                    run_loss, n_loss = 0.0, 0
        ev = evaluate(engine, "val", N_VAL, a.scale)
        row = {"epoch": epoch + 1, "val": ev, "seconds": round(time.time() - t0)}
        log.append(row)
        print("VAL", json.dumps(row), flush=True)
        model.save_pretrained(c.OUT / f"lora_{a.tag}_epoch{epoch + 1}")
        if best is None or ev["iou0.2"]["f1"] > best["val"]["iou0.2"]["f1"]:
            best = row
    json.dump({"args": vars(a), "log": log, "best_epoch": best["epoch"]}, open(c.OUT / f"ft_{a.tag}.json", "w"), indent=1)
    print("best epoch on val:", best["epoch"])


if __name__ == "__main__":
    main()
