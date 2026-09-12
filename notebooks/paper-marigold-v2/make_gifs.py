#!/usr/bin/env python3
"""Build the lightweight modality GIFs embedded in the Paper of the Week post.

Keeps each GIF well under quarto.pub's ~1 MB upload comfort zone via resize +
palette quantize. Re-run after regenerating output/{depth,normals,albedo}*.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent.parent / "posts" / "series" / "papers" / "images"
W = 480


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def fit(im: Image.Image, w: int = W) -> Image.Image:
    h = int(round(im.height * (w / im.width)))
    return im.resize((w, h), Image.Resampling.LANCZOS)


def label_frame(im: Image.Image, text: str, hold: int = 2) -> list[Image.Image]:
    im = fit(im).copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except OSError:
        font = ImageFont.load_default()
    pad = 8
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bar_h = th + pad * 2
    draw.rectangle([0, im.height - bar_h, im.width, im.height], fill=(20, 20, 20))
    draw.text(
        ((im.width - tw) // 2, im.height - bar_h + pad),
        text,
        fill=(255, 255, 255),
        font=font,
    )
    return [im] * hold


def save_gif(
    frames: list[Image.Image], path: Path, duration: int = 700, colors: int = 48
) -> None:
    q = [
        f.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors) for f in frames
    ]
    q[0].save(
        path,
        save_all=True,
        append_images=q[1:],
        duration=duration,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"{path.name}: {path.stat().st_size / 1024:.0f} KB, {len(frames)} frames")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    crop = "C0008_0011_defect"
    xray = load_rgb(ROOT / "xray_inputs" / f"{crop}.png")
    depth = load_rgb(
        ROOT
        / "output/depth/images/visualizations/depth_spectral"
        / f"{crop}.png"
    )
    normals = load_rgb(
        ROOT / "output/normals/images/visualizations/normals" / f"{crop}.png"
    )
    albedo = load_rgb(
        ROOT / "output/albedo/images/visualizations/albedo" / f"{crop}.png"
    )
    depth_base = load_rgb(
        ROOT
        / "output/depth_base/images/visualizations/depth_spectral"
        / f"{crop}.png"
    )
    normals_base = load_rgb(
        ROOT
        / "output/normals_base/images/visualizations/normals"
        / f"{crop}.png"
    )
    # stills used by the Case 2 layout
    for name, im in [
        ("marigold-v2-case2-xray.jpg", xray),
        ("marigold-v2-case2-depth.jpg", depth),
        ("marigold-v2-case2-normals.jpg", normals),
        ("marigold-v2-case2-albedo.jpg", albedo),
    ]:
        fit(im, 768).save(OUT / name, "JPEG", quality=92, optimize=True)

    frames1: list[Image.Image] = []
    frames1 += label_frame(xray, "X-ray input")
    frames1 += label_frame(depth, "Depth (Marigold V2 LoRA)")
    frames1 += label_frame(normals, "Normals (Marigold V2 LoRA)")
    frames1 += label_frame(albedo, "Albedo (Marigold V2 LoRA)")
    save_gif(frames1, OUT / "marigold-v2-modalities.gif", duration=800, colors=64)

    frames2: list[Image.Image] = []
    frames2 += label_frame(xray, "X-ray input")
    frames2 += label_frame(depth_base, "Depth — base model (no LoRA)")
    frames2 += label_frame(depth, "Depth — fine-tuned LoRA")
    frames2 += label_frame(normals_base, "Normals — base model (no LoRA)")
    frames2 += label_frame(normals, "Normals — fine-tuned LoRA")
    save_gif(frames2, OUT / "marigold-v2-lora-vs-base.gif", duration=750, colors=48)


if __name__ == "__main__":
    main()
