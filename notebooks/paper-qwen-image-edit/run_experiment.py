"""Real experiment: can Qwen-Image-2.1's mask-conditioned local editing synthesize
convincing, controllable, well-placed defects on real GDXray Castings X-ray images?

Tests (per the actual brief this script answers):
  1. Placement control  -- does the model confine the defect to an annotated mask region?
  2. Diversity           -- same mask+prompt, different seeds: real variation or near-duplicates?
  3. Multiple defect types -- void / crack / inclusion, does quality hold across types?
  4. Severity control    -- does prompt wording (faint vs pronounced) change output meaningfully?
  5. Texture realism     -- do synthesized regions match real GDXray grayscale/noise statistics?
  6. Failure modes       -- deliberately adversarial / underspecified prompts, logged honestly.

Mask convention used: Qwen-Image-2.1 has no `mask_image` argument (confirmed by reading the
actual diffusers `QwenImage21Pipeline.__call__` signature on the git-main branch -- there is no
such parameter). Its README instead says local edits are specified via "circles, painted
annotations, or separate masks" baked into the input image itself. We test the painted-annotation
convention: draw a thin red circle directly on a copy of the source image marking the target
region, and instruct the model to edit inside it.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic-zero-to-advanced"))
from baseline_detector import DATA_DIR, load_ground_truth, object_mask  # noqa: E402

from diffusers import QwenImage21Pipeline  # noqa: E402

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)
POST_IMAGES_DIR = Path(__file__).parent.parent.parent / "posts/series/papers/images"
POST_IMAGES_DIR.mkdir(exist_ok=True)

MASK_RADIUS = 22
CLEAN_MARGIN = 45  # min pixel distance from any real GT defect box, so masks land on genuinely clean metal
NUM_STEPS = 28

DEFECT_PROMPTS = {
    "void": (
        "Inside the thin red circle marked on this X-ray image, add a small dark circular void "
        "defect matching the grayscale film-grain noise texture of the surrounding aluminum casting. "
        "Do not change anything outside the circle. Remove the red circle marking itself from the "
        "final image -- it must not be visible in the output."
    ),
    "crack": (
        "Inside the thin red circle marked on this X-ray image, add a thin, dark, linear crack "
        "defect a few pixels wide, matching the grayscale film-grain noise texture of the surrounding "
        "aluminum casting. Do not change anything outside the circle. Remove the red circle marking "
        "itself from the final image -- it must not be visible in the output."
    ),
    "inclusion": (
        "Inside the thin red circle marked on this X-ray image, add a small elongated dark "
        "inclusion defect (an irregular blob, not a perfect circle), matching the grayscale "
        "film-grain noise texture of the surrounding aluminum casting. Do not change anything "
        "outside the circle. Remove the red circle marking itself from the final image -- it must "
        "not be visible in the output."
    ),
}

SEVERITY_PROMPTS = {
    "faint": (
        "Inside the thin red circle marked on this X-ray image, add a very faint, barely visible "
        "dark void defect, subtle and low-contrast, matching the grayscale film-grain noise texture "
        "of the surrounding aluminum casting. Do not change anything outside the circle. Remove the "
        "red circle marking itself from the final image."
    ),
    "pronounced": (
        "Inside the thin red circle marked on this X-ray image, add a strongly visible, high-contrast "
        "dark void defect, clearly darker than the surrounding metal, matching the grayscale "
        "film-grain noise texture of the surrounding aluminum casting. Do not change anything outside "
        "the circle. Remove the red circle marking itself from the final image."
    ),
}

FAILURE_PROMPTS = {
    "no_location_given": (
        "Add a small dark void defect to this X-ray image, matching the grayscale film-grain noise "
        "texture of the surrounding aluminum casting."
    ),  # deliberately omits any mask/location instruction -- tests what happens with no spatial anchor
    "contradictory": (
        "Inside the thin red circle marked on this X-ray image, make the region brighter and remove "
        "any defects there, but also add a large obvious crack across the whole casting."
    ),  # deliberately self-contradictory instruction
}

IMAGES = ["C0001_0001.png", "C0001_0030.png"]
SEEDS = [0, 1, 2]


def pick_clean_point(gray: np.ndarray, gt_boxes, rng: np.random.Generator, n_needed: int):
    mask = object_mask(gray)
    ys, xs = np.where(mask > 0)
    points = []
    attempts = 0
    while len(points) < n_needed and attempts < 5000:
        attempts += 1
        i = rng.integers(0, len(xs))
        x, y = int(xs[i]), int(ys[i])
        if x < MASK_RADIUS + 5 or y < MASK_RADIUS + 5 or x > gray.shape[1] - MASK_RADIUS - 5 or y > gray.shape[0] - MASK_RADIUS - 5:
            continue
        ok = True
        for (x1, y1, x2, y2) in gt_boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if abs(cx - x) < CLEAN_MARGIN and abs(cy - y) < CLEAN_MARGIN:
                ok = False
                break
        for (px, py) in points:
            if abs(px - x) < MASK_RADIUS * 3 and abs(py - y) < MASK_RADIUS * 3:
                ok = False
                break
        if ok:
            points.append((x, y))
    return points


def draw_mask_annotation(img_rgb: Image.Image, cx: int, cy: int, radius: int = MASK_RADIUS) -> Image.Image:
    arr = np.array(img_rgb).copy()
    cv2.circle(arr, (cx, cy), radius, (255, 0, 0), 2)
    return Image.fromarray(arr)


def texture_stats(gray: np.ndarray, cx: int, cy: int, radius: int) -> dict:
    y0, y1 = max(0, cy - radius), min(gray.shape[0], cy + radius)
    x0, x1 = max(0, cx - radius), min(gray.shape[1], cx + radius)
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    lap = cv2.Laplacian(patch, cv2.CV_32F)
    return {
        "mean": float(patch.mean()),
        "std": float(patch.std()),
        "laplacian_var": float(lap.var()),
    }


def main():
    results = []
    gt = load_ground_truth(DATA_DIR)
    rng = np.random.default_rng(42)

    print("Loading Qwen-Image-2.1 ...")
    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1", dtype=torch.bfloat16)
    pipe.to("cuda")
    load_time = time.time() - t0
    load_mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded in {load_time:.1f}s, {load_mem_gb:.2f} GB allocated")

    def run_edit(source_img_rgb, prompt, seed, tag):
        gen = torch.Generator("cuda").manual_seed(seed)
        t = time.time()
        torch.cuda.reset_peak_memory_stats()
        out = pipe(
            prompt=prompt,
            image=source_img_rgb,
            num_inference_steps=NUM_STEPS,
            generator=gen,
        ).images[0]
        dt = time.time() - t
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        out_path = OUT_DIR / f"{tag}.png"
        out.save(out_path)
        print(f"  [{tag}] {dt:.1f}s, peak {peak_mem:.2f} GB -> {out_path.name}")
        return out, dt, peak_mem

    for img_name in IMAGES:
        img_path = DATA_DIR / img_name
        idx = int(img_name.split("_")[1].split(".")[0])
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        boxes = gt.get(idx, [])
        source_rgb = Image.open(img_path).convert("RGB")

        n_points_needed = len(DEFECT_PROMPTS) + 1  # +1 for severity test location
        points = pick_clean_point(gray, boxes, rng, n_points_needed)
        if len(points) < n_points_needed:
            print(f"WARNING: only found {len(points)} clean points for {img_name}, need {n_points_needed}")

        point_iter = iter(points)

        # --- 1 & 3: defect type x seed diversity grid ---
        for defect_type, prompt in DEFECT_PROMPTS.items():
            cx, cy = next(point_iter)
            masked_input = draw_mask_annotation(source_rgb, cx, cy)
            masked_input.save(OUT_DIR / f"{img_name}_{defect_type}_masked_input.png")
            before_stats = texture_stats(gray, cx, cy, MASK_RADIUS)

            for seed in SEEDS:
                tag = f"{Path(img_name).stem}_{defect_type}_seed{seed}"
                out_img, dt, peak_mem = run_edit(masked_input, prompt, seed, tag)
                out_gray = np.array(out_img.convert("L"))
                # output may be resized by the pipeline; rescale mask coords proportionally
                scale_x = out_img.size[0] / source_rgb.size[0]
                scale_y = out_img.size[1] / source_rgb.size[1]
                ocx, ocy, orad = int(cx * scale_x), int(cy * scale_y), int(MASK_RADIUS * max(scale_x, scale_y))
                after_stats = texture_stats(out_gray, ocx, ocy, orad)
                results.append({
                    "image": img_name,
                    "test": "defect_type_diversity",
                    "defect_type": defect_type,
                    "seed": seed,
                    "mask_center_src": [cx, cy],
                    "inference_s": round(dt, 2),
                    "peak_mem_gb": round(peak_mem, 2),
                    "before_stats": before_stats,
                    "after_stats": after_stats,
                    "out_file": f"{tag}.png",
                })

        # --- 4: severity control (void type, fixed seed) ---
        cx, cy = next(point_iter)
        masked_input = draw_mask_annotation(source_rgb, cx, cy)
        masked_input.save(OUT_DIR / f"{img_name}_severity_masked_input.png")
        for severity, prompt in SEVERITY_PROMPTS.items():
            tag = f"{Path(img_name).stem}_severity_{severity}"
            out_img, dt, peak_mem = run_edit(masked_input, prompt, seed=7, tag=tag)
            out_gray = np.array(out_img.convert("L"))
            scale_x = out_img.size[0] / source_rgb.size[0]
            scale_y = out_img.size[1] / source_rgb.size[1]
            ocx, ocy, orad = int(cx * scale_x), int(cy * scale_y), int(MASK_RADIUS * max(scale_x, scale_y))
            after_stats = texture_stats(out_gray, ocx, ocy, orad)
            results.append({
                "image": img_name,
                "test": "severity_control",
                "severity": severity,
                "seed": 7,
                "mask_center_src": [cx, cy],
                "inference_s": round(dt, 2),
                "peak_mem_gb": round(peak_mem, 2),
                "after_stats": after_stats,
                "out_file": f"{tag}.png",
            })

    # --- 6: failure modes, run once each on the first image ---
    img_path = DATA_DIR / IMAGES[0]
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    source_rgb = Image.open(img_path).convert("RGB")
    boxes = gt.get(int(IMAGES[0].split("_")[1].split(".")[0]), [])
    fail_points = pick_clean_point(gray, boxes, rng, 1)
    cx, cy = fail_points[0]
    masked_input = draw_mask_annotation(source_rgb, cx, cy)
    for fail_tag, prompt in FAILURE_PROMPTS.items():
        img_for_prompt = source_rgb if fail_tag == "no_location_given" else masked_input
        tag = f"failure_{fail_tag}"
        out_img, dt, peak_mem = run_edit(img_for_prompt, prompt, seed=0, tag=tag)
        results.append({
            "image": IMAGES[0],
            "test": "failure_mode",
            "case": fail_tag,
            "mask_center_src": [cx, cy] if fail_tag != "no_location_given" else None,
            "inference_s": round(dt, 2),
            "peak_mem_gb": round(peak_mem, 2),
            "out_file": f"{tag}.png",
        })

    manifest = {
        "model": "Qwen/Qwen-Image-2.1",
        "pipeline_class": "QwenImage21Pipeline",
        "diffusers_version": __import__("diffusers").__version__,
        "num_inference_steps": NUM_STEPS,
        "mask_radius_px": MASK_RADIUS,
        "load_time_s": round(load_time, 1),
        "load_mem_gb": round(load_mem_gb, 2),
        "results": results,
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(results)} results to {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
