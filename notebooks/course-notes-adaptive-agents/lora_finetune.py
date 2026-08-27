"""LoRA fine-tuning demo — open-source reproduction of Lesson 5 ("Super Polite").

Everything here is already open source (unlike Lessons 2-4, no proprietary
substitution needed) — we just actually run it: a small open base model,
`peft` LoRA adapters, a template-generated synthetic dataset (no paid LLM
API needed to make the data either), trained on this machine's own GPU.

Note on QLoRA: the course quantizes to 4-bit with bitsandbytes. At this model
size (0.5B) on a GPU with 124GB unified memory, 4-bit quantization buys
nothing — the point of QLoRA is fitting a fine-tune onto a GPU too small to
hold the full-precision weights. We train in bf16 directly and skip
bitsandbytes; see the comment near MODEL_NAME for what to change on a larger
model / smaller GPU.

Run:
    uv run lora_finetune.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Swap for e.g. "Qwen/Qwen2.5-7B-Instruct" + `load_in_4bit=True` (bitsandbytes)
# on a GPU with < 16GB to reproduce the course's actual QLoRA setup.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

QUESTIONS = [
    "What time does the store open?",
    "Can you reset my password?",
    "Where is my order?",
    "How do I cancel my subscription?",
    "Is this product in stock?",
    "What's your return policy?",
    "Can I get a refund?",
    "How long does shipping take?",
    "Do you have this in a different size?",
    "Why was I charged twice?",
]

# The three politeness intensities the course used, as templates — no paid
# LLM call needed to generate training data.
TEMPLATES = {
    "courteous": [
        "Sure, {answer}",
        "Of course — {answer}",
        "Happy to help: {answer}",
    ],
    "warm": [
        "Thanks so much for reaching out! {answer} Let me know if there's anything else I can do for you.",
        "I really appreciate you asking! {answer} I'm glad to help however I can.",
    ],
    "effusive": [
        "Oh, what a wonderful question, thank you for asking! {answer} It's genuinely my pleasure to help you today, "
        "and please don't hesitate to reach out again — I'm always delighted to assist!",
        "I'm so grateful you brought this to me! {answer} Helping you with this made my day, truly — "
        "feel free to come back with anything else at all!",
    ],
}

ANSWERS = {
    "What time does the store open?": "we open at 9 AM every day except Sunday.",
    "Can you reset my password?": "I've sent a password reset link to your email.",
    "Where is my order?": "your order shipped yesterday and should arrive within 3 business days.",
    "How do I cancel my subscription?": "you can cancel anytime from Account Settings > Subscriptions.",
    "Is this product in stock?": "yes, it's currently in stock and ready to ship.",
    "What's your return policy?": "you can return any item within 30 days for a full refund.",
    "Can I get a refund?": "I've processed your refund, it should appear in 3-5 business days.",
    "How long does shipping take?": "standard shipping takes 3-5 business days.",
    "Do you have this in a different size?": "yes, it's available in small, medium, and large.",
    "Why was I charged twice?": "that was a hold that's since been reversed — you were only charged once.",
}


def build_dataset(n_per_question: int = 15, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for q in QUESTIONS:
        answer = ANSWERS[q]
        for _ in range(n_per_question):
            intensity = rng.choice(list(TEMPLATES.keys()))
            template = rng.choice(TEMPLATES[intensity])
            rows.append({"question": q, "response": template.format(answer=answer), "intensity": intensity})
    rng.shuffle(rows)
    return rows


def to_chat_text(tokenizer, question: str, response: str) -> str:
    messages = [
        {"role": "system", "content": "You are Super Polite Assistant. Always answer warmly and courteously."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def train(out_dir: Path):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="auto")

    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    dataset = build_dataset()
    texts = [to_chat_text(tok, r["question"], r["response"]) for r in dataset]
    encodings = tok(texts, truncation=True, max_length=128, padding=True, return_tensors="pt")

    device = model.device
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    batch_size = 8
    n_epochs = 6
    losses = []

    model.train()
    n = input_ids.size(0)
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            out = model(input_ids=input_ids[idx], attention_mask=attention_mask[idx], labels=labels[idx])
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(out.loss.item())
        print(f"epoch {epoch + 1}/{n_epochs}  mean loss {sum(losses[-len(range(0, n, batch_size)):]) / len(range(0, n, batch_size)):.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "adapter")
    (out_dir / "loss.json").write_text(json.dumps(losses))

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title(f"LoRA fine-tune loss — {MODEL_NAME}, {trainable:,} trainable params ({100 * trainable / total:.2f}%)")
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=130)
    print(f"Saved adapter + loss curve to {out_dir}")

    return tok, base, model


@torch.no_grad()
def compare(tok, base_model, adapted_model, prompt: str) -> tuple[str, str]:
    # Same system framing used at train time — testing with a *different*
    # context than the adapter was trained under is a mismatch, not a fair
    # generalization test.
    messages = [
        {"role": "system", "content": "You are Super Polite Assistant. Always answer warmly and courteously."},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(base_model.device)

    def gen(m):
        out = m.generate(**inputs, max_new_tokens=60, do_sample=False)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    adapted_model.eval()
    with adapted_model.disable_adapter():
        base_reply = gen(adapted_model)
    adapted_reply = gen(adapted_model)
    return base_reply, adapted_reply


if __name__ == "__main__":
    out_dir = Path("lora_out")
    tok, base_model, adapted_model = train(out_dir)

    print("\n=== Base vs. LoRA-adapted, same prompts, greedy decoding ===")
    test_prompts = [
        "Can I get a refund?",           # in-distribution: exact training question
        "Can you cancel my order?",      # held-out: paraphrase of a training question
    ]
    comparisons = []
    for p in test_prompts:
        base_reply, adapted_reply = compare(tok, base_model, adapted_model, p)
        print(f"\nPrompt: {p}\n  base:    {base_reply}\n  adapted: {adapted_reply}")
        comparisons.append({"prompt": p, "base": base_reply, "adapted": adapted_reply})
    (out_dir / "comparisons.json").write_text(json.dumps(comparisons, indent=2))
