"""Real comparison: naive frame-by-frame SAM2 tracking-by-detection (point derived from the
previous frame's own predicted mask) vs. ENEAS-style tracking with the real semantic
verification layer (DINOv2 embedding match, falling back to real Qwen2-VL when ambiguous) --
on a real DAVIS-2017 sequence with 6 visually similar instances (gold-fish), scored against the
real ground-truth instance mask every single frame.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image

from common import load_frame, load_gt_mask, mask_centroid, mask_bbox, iou, N_FRAMES
from model import sam2_segment, dino_embed, vlm_same_object

IMG_OUT = "../../posts/series/papers/images"
OUT = "out"
os.makedirs(OUT, exist_ok=True)

HIGH_THRESH = 0.80   # confident match -- accept without asking the VLM
LOW_THRESH = 0.55    # confident mismatch -- reject without asking the VLM
                      # in between: ambiguous, real Qwen2-VL call decides


def crop_for_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    box = mask_bbox(mask)
    if box is None:
        return image
    return image.crop(box)


if __name__ == "__main__":
    frame0 = load_frame(0)
    gt0 = load_gt_mask(0)
    point0 = mask_centroid(gt0)
    print(f"Frame 0 target centroid: {point0}")

    init_mask, init_conf = sam2_segment(frame0, point0)
    print(f"Frame 0 SAM2 IoU vs GT: {iou(init_mask, gt0):.3f} (confidence {init_conf:.3f})")
    ref_crop = crop_for_mask(frame0, init_mask)
    ref_embedding = dino_embed(ref_crop)

    naive_mask = init_mask
    eneas_mask = init_mask
    naive_ious, eneas_ious, similarities, vlm_calls = [], [], [], []
    frames_viz = []
    n_vlm_calls = 0
    n_rejections = 0

    for t in range(N_FRAMES):
        frame = load_frame(t)
        gt = load_gt_mask(t)

        # --- naive: always accept whatever SAM2 returns from the previous point ---
        pt = mask_centroid(naive_mask)
        if pt is not None:
            naive_mask, _ = sam2_segment(frame, pt)
        naive_ious.append(iou(naive_mask, gt))

        # --- ENEAS-style: verify identity before accepting ---
        pt2 = mask_centroid(eneas_mask)
        accepted_mask = eneas_mask
        sim = None
        if pt2 is not None:
            candidate, _ = sam2_segment(frame, pt2)
            cand_crop = crop_for_mask(frame, candidate)
            cand_embedding = dino_embed(cand_crop)
            sim = float((ref_embedding @ cand_embedding).item())
            if sim >= HIGH_THRESH:
                accepted_mask = candidate
            elif sim <= LOW_THRESH:
                n_rejections += 1  # confident mismatch -- hold last confirmed position
            else:
                n_vlm_calls += 1
                if vlm_same_object(ref_crop, cand_crop):
                    accepted_mask = candidate
                else:
                    n_rejections += 1
        eneas_mask = accepted_mask
        eneas_ious.append(iou(eneas_mask, gt))
        similarities.append(sim)

        if t == 64:  # the drift frame -- clean overlay pair for the interactive slider
            arr = np.array(frame).astype(np.float32)
            naive_overlay = arr.copy()
            naive_overlay[naive_mask] = naive_overlay[naive_mask] * 0.4 + np.array([255, 0, 0]) * 0.6
            Image.fromarray(naive_overlay.clip(0, 255).astype(np.uint8)).save(f"{IMG_OUT}/eneas-slider-before.jpg", quality=90)
            eneas_overlay = arr.copy()
            eneas_overlay[eneas_mask] = eneas_overlay[eneas_mask] * 0.4 + np.array([0, 180, 255]) * 0.6
            Image.fromarray(eneas_overlay.clip(0, 255).astype(np.uint8)).save(f"{IMG_OUT}/eneas-slider-after.jpg", quality=90)

        if t % 8 == 0 or t == N_FRAMES - 1:
            print(f"frame {t:2d}  naive IoU {naive_ious[-1]:.3f}  ENEAS IoU {eneas_ious[-1]:.3f}  "
                  f"sim {sim}")
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
            axes[0].imshow(frame); axes[0].contour(gt, colors="lime", linewidths=2)
            axes[0].set_title("ground truth", fontsize=9); axes[0].axis("off")
            axes[1].imshow(frame); axes[1].contour(naive_mask, colors="red", linewidths=2)
            axes[1].set_title(f"naive, IoU={naive_ious[-1]:.2f}", fontsize=9); axes[1].axis("off")
            axes[2].imshow(frame); axes[2].contour(eneas_mask, colors="deepskyblue", linewidths=2)
            axes[2].set_title(f"ENEAS-style, IoU={eneas_ious[-1]:.2f}", fontsize=9); axes[2].axis("off")
            plt.tight_layout()
            fig.canvas.draw()
            frame_img = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                                          fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")
            frames_viz.append(frame_img)
            plt.close(fig)

    print(f"\nVLM verification calls: {n_vlm_calls}/{N_FRAMES} frames  "
          f"({n_rejections} candidates rejected outright by embedding check)")
    print(f"Mean IoU -- naive: {np.mean(naive_ious):.3f}  ENEAS-style: {np.mean(eneas_ious):.3f}")

    with open(f"{OUT}/results.json", "w") as f:
        json.dump({"naive_ious": naive_ious, "eneas_ious": eneas_ious, "similarities": similarities,
                   "n_vlm_calls": n_vlm_calls, "n_rejections": n_rejections}, f, indent=2)

    frames_np = [np.array(f) for f in frames_viz]
    imageio.mimsave(f"{IMG_OUT}/eneas-tracking-progress.mp4", frames_np, fps=3)
    print("Saved eneas-tracking-progress.mp4")

    plt.figure(figsize=(8, 4.5))
    plt.plot(naive_ious, label=f"Naive tracking-by-detection (mean {np.mean(naive_ious):.3f})", color="red")
    plt.plot(eneas_ious, label=f"ENEAS-style w/ semantic verification (mean {np.mean(eneas_ious):.3f})", color="deepskyblue")
    plt.xlabel("frame"); plt.ylabel("IoU vs real ground truth")
    plt.title("Real per-frame tracking accuracy, DAVIS 'gold-fish' (6 similar instances)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{IMG_OUT}/eneas-iou-curve.png", dpi=130)
    print("Saved eneas-iou-curve.png")
