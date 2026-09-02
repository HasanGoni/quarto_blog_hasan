"""Part 2: does *video* conditioning (many real frames) hold the X-ray domain longer than
Part 1's single-image conditioning did?

Same model, same quantization strategy, same prompt/negative prompt, same output settings as
Part 1 (`generate_video_sound.py`) -- the only variable changed is the conditioning signal itself:
`LTX2ConditionPipeline` + `LTX2VideoCondition(frames=<N copies of the real seed image>)` instead
of `LTX2ImageToVideoPipeline`'s single-frame conditioning. This directly tests the "Video
Strength" / "Context Duration" idea documented for LTX-2's own video-conditioning feature.
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
    LTX2ConditionPipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.utils import encode_video
from transformers import AutoTokenizer, BitsAndBytesConfig as TransformersBnBConfig, Gemma3ForConditionalGeneration

REPO_ID = "Lightricks/LTX-2"

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "synthetic-data-xray" / "src"))
from synth_xray.diffusion_data import load_gray, to_rgb_pil  # noqa: E402
from synth_xray.groundtruth import parse_gdxray_bboxes  # noqa: E402

GDXRAY_CACHE = pathlib.Path(__file__).parent.parent / "synthetic-data-xray" / ".data_cache"


def load_seed_image() -> "PIL.Image.Image":
    """Same real GDXray Welds frame Part 1 used."""
    series_dir = GDXRAY_CACHE / "Welds" / "W0001"
    gt_path = next(series_dir.glob("*.txt"))
    boxes = parse_gdxray_bboxes(gt_path)
    image_id = next(iter(boxes))
    img_path = next(series_dir.glob(f"*{image_id:03d}*.png"))
    gray = load_gray(img_path)
    return to_rgb_pil(gray)


def build_pipeline(device: str) -> LTX2ConditionPipeline:
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

    pipe = LTX2ConditionPipeline(
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
    parser.add_argument("--out", default="out/xray_scan_conditioned.mp4")
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--condition-frames", type=int, default=41, help="Real frames of the repeated still used as the video condition (~1.7s at 24fps).")
    args = parser.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda:0"
    t0 = time.time()
    pipe = build_pipeline(device)
    print(f"[build] pipeline assembled in {time.time() - t0:.0f}s")
    print(f"[build] torch.cuda.max_memory_allocated after build: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    seed_image = load_seed_image()
    video_condition_frames = [seed_image] * args.condition_frames
    condition = LTX2VideoCondition(frames=video_condition_frames, index=0, strength=1.0)

    prompt = (
        "A radioscopy inspection system scanning a metal weld joint, the X-ray detector head "
        "sweeping slowly across the seam, revealing internal structure. Ambient industrial "
        "sound: a steady conveyor motor hum and faint mechanical clicks from the scan head."
    )
    negative_prompt = "shaky, glitchy, low quality, blurry, static, distorted"

    frame_rate = 24.0
    print(f"[generate] running LTX2ConditionPipeline, video condition = {args.condition_frames} real frames")
    t1 = time.time()
    video, audio = pipe(
        conditions=condition,
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
