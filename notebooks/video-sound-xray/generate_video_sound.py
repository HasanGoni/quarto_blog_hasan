"""Generate a synthetic X-ray inspection-scan video with ambient sound from a real
GDXray weld radiograph, using LTX-2 (Lightricks) via diffusers' LTX2ImageToVideoPipeline.

Both the ~54GB (bf16) Gemma3 text encoder and the ~43GB (bf16) video/audio transformer are
loaded int4-quantized via bitsandbytes -- neither ships a pre-quantized text encoder, and this
machine's real free memory (checked via `free -h` / `nvidia-smi`, ~25-37GB free with vLLM
resident) doesn't fit either component at bf16, let alone both.
"""

import argparse
import pathlib
import sys
import time

import torch
from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    BitsAndBytesConfig as DiffusersBnBConfig,
    FlowMatchEulerDiscreteScheduler,
    LTX2ImageToVideoPipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.utils import encode_video
from transformers import AutoTokenizer, BitsAndBytesConfig as TransformersBnBConfig, Gemma3ForConditionalGeneration

REPO_ID = "Lightricks/LTX-2"

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "synthetic-data-xray" / "src"))
from synth_xray.diffusion_data import load_gray, to_rgb_pil  # noqa: E402
from synth_xray.groundtruth import parse_gdxray_bboxes  # noqa: E402

GDXRAY_CACHE = pathlib.Path(__file__).parent.parent / "synthetic-data-xray" / ".data_cache"


def load_seed_image() -> "PIL.Image.Image":
    """Reuse the real GDXray Welds crop used throughout the Synthetic Data series."""
    series_dir = GDXRAY_CACHE / "Welds" / "W0001"
    gt_path = next(series_dir.glob("*.txt"))
    boxes = parse_gdxray_bboxes(gt_path)
    image_id = next(iter(boxes))
    img_path = next(series_dir.glob(f"*{image_id:03d}*.png"))
    gray = load_gray(img_path)
    return to_rgb_pil(gray)


def build_pipeline(device: str) -> LTX2ImageToVideoPipeline:
    text_quant = TransformersBnBConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    transformer_quant = DiffusersBnBConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    print("[load] tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, subfolder="tokenizer")

    print("[load] text_encoder (int4)")
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        quantization_config=text_quant,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    print("[load] transformer (int4)")
    transformer = LTX2VideoTransformer3DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        quantization_config=transformer_quant,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    print("[load] vae / audio_vae / connectors / vocoder / scheduler")
    vae = AutoencoderKLLTX2Video.from_pretrained(REPO_ID, subfolder="vae", torch_dtype=torch.bfloat16).to(device)
    audio_vae = AutoencoderKLLTX2Audio.from_pretrained(REPO_ID, subfolder="audio_vae", torch_dtype=torch.bfloat16).to(
        device
    )
    connectors = LTX2TextConnectors.from_pretrained(REPO_ID, subfolder="connectors", torch_dtype=torch.bfloat16).to(
        device
    )
    vocoder = LTX2Vocoder.from_pretrained(REPO_ID, subfolder="vocoder", torch_dtype=torch.bfloat16).to(device)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(REPO_ID, subfolder="scheduler")

    pipe = LTX2ImageToVideoPipeline(
        vae=vae,
        audio_vae=audio_vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler,
        vocoder=vocoder,
        connectors=connectors,
    )
    pipe.vae.enable_tiling()
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="out/xray_scan.mp4")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda:0"
    t0 = time.time()
    pipe = build_pipeline(device)
    print(f"[build] pipeline assembled in {time.time() - t0:.0f}s")
    print(f"[build] torch.cuda.max_memory_allocated after build: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    image = load_seed_image()
    prompt = (
        "A radioscopy inspection system scanning a metal weld joint, the X-ray detector head "
        "sweeping slowly across the seam, revealing internal structure. Ambient industrial "
        "sound: a steady conveyor motor hum and faint mechanical clicks from the scan head."
    )
    negative_prompt = "shaky, glitchy, low quality, blurry, static, distorted"

    frame_rate = 24.0
    print("[generate] running LTX2ImageToVideoPipeline")
    t1 = time.time()
    video, audio = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        frame_rate=frame_rate,
        num_inference_steps=args.steps,
        guidance_scale=3.0,
        output_type="np",
        return_dict=False,
    )
    print(f"[generate] done in {time.time() - t1:.0f}s")
    print(f"[generate] torch.cuda.max_memory_allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    encode_video(
        video[0],
        fps=frame_rate,
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        output_path=str(out_path),
    )
    print(f"[save] wrote {out_path}")


if __name__ == "__main__":
    main()
