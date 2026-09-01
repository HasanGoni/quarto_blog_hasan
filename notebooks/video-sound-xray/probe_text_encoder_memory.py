"""Real memory probe for LTX-2's text encoder (Gemma3, bf16 ~54GB on disk).

Downloads and loads *only* the text_encoder component, int4-quantized via
bitsandbytes, and reports real resident memory before/after. This is the
single largest, riskiest component of the LTX-2 pipeline on this machine's
~25-37GB free unified memory -- worth checking in isolation before pulling
the whole ~80GB pipeline.
"""

import argparse
import time

import psutil
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForConditionalGeneration

REPO_ID = "Lightricks/LTX-2"


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4)
    args = parser.parse_args()

    print(f"[probe] RSS before load: {rss_gb():.2f} GB")
    t0 = time.time()

    quant_config = (
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        if args.bits == 4
        else BitsAndBytesConfig(load_in_8bit=True)
    )

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, subfolder="tokenizer")
    print(f"[probe] tokenizer loaded, RSS: {rss_gb():.2f} GB")

    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    load_s = time.time() - t0
    print(f"[probe] text_encoder loaded in {load_s:.0f}s, RSS: {rss_gb():.2f} GB")
    print(f"[probe] torch.cuda.memory_allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"[probe] torch.cuda.max_memory_allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    prompt = "A weld radiograph inspection scan moving across a metal joint, ambient conveyor motor hum."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = text_encoder(**inputs, output_hidden_states=True)
    print(f"[probe] forward pass OK, last hidden shape: {out.hidden_states[-1].shape}")
    print(f"[probe] RSS after forward: {rss_gb():.2f} GB")
    print(f"[probe] torch.cuda.max_memory_allocated after forward: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
