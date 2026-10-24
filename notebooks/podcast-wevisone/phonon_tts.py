"""Phonon (Gradium 40m-en, catalog voices) text -> wav. Usage: phonon_tts.py  (library: synth(text, voice_name))"""
import json, os, time
from pathlib import Path
import numpy as np, torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration

os.environ.setdefault("HF_HUB_CACHE", str(Path.home() / ".hermes/cache/huggingface-hub"))
MODEL = "gradium/phonon-40m-en"
VOICES = {}
for l in open(Path(hf_hub_download("gradium/phonon-voices", "voices.jsonl"))):
    d = json.loads(l); VOICES[d["name"]] = d
processor = AutoProcessor.from_pretrained(MODEL)
model = GradiumPhononTTSForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32).eval().to("cuda" if torch.cuda.is_available() and not os.environ.get("PHONON_CPU") else "cpu")
_lat = {}

def synth(text, voice):
    if voice not in _lat:
        _lat[voice] = load_file(hf_hub_download("gradium/phonon-voices", VOICES[voice]["filename"]))["speaker_wavs"][0].T.float()
    conv = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": text}]}]
    inputs = processor.apply_chat_template(conv, tokenize=True, return_dict=True, add_generation_prompt=True,
                                           processor_kwargs={"audio_prompt_latents": [_lat[voice]]}).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs)
    audio = processor.decode(out.audio_latents, out.audio_latents_mask)
    return audio

if __name__ == "__main__":
    t = time.time()
    a = synth("Welcome to the show. Today we are looking at a paper from Tencent's WeChat team.", "Harper")
    print(type(a), getattr(a, "shape", None), f"{time.time()-t:.1f}s")
    processor.save_audio(a, "out_test.wav")
