"""Plain, non-agentic classical-CV baseline for GDXray Castings defect detection.

No LLM, no agent, no learning at all -- top-hat morphology + Otsu object masking +
connected components, scored against GDXray's real ground_truth.txt for series C0001.
This is Arc 1 Part 2's baseline: the number every later, agentic post gets compared against.
"""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(
    "/home/hasan-spark/.cache/kagglehub/datasets/samarthgoel2604/"
    "gdxray-without-synthetic-for-defectd/versions/1/Castings kaggle/Castings (1)/Castings/C0001"
)
OUT_DIR = Path(__file__).parent / "out"
POST_IMAGES_DIR = Path(__file__).parent.parent.parent / "posts/series/agentic-zero-to-advanced/images"
OUT_DIR.mkdir(exist_ok=True)

TOPHAT_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
CLEAN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
MIN_BLOB_AREA = 9
TOPHAT_PERCENTILE = 99.5
IOU_HIT_THRESHOLD = 0.05


def load_ground_truth(series_dir: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    gt: dict[int, list[tuple[float, float, float, float]]] = {}
    for line in (series_dir / "ground_truth.txt").read_text().splitlines():
        if not line.strip():
            continue
        idx, x1, x2, y1, y2 = (float(v) for v in line.split())
        gt.setdefault(int(idx), []).append((x1, y1, x2, y2))
    return gt


def object_mask(gray: np.ndarray) -> np.ndarray:
    """Otsu-threshold the casting body away from the surrounding air, eroded a bit
    so the very edge of the casting (a strong, boring top-hat response) doesn't
    get treated as a candidate defect."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask.mean() > 255 / 2:
        mask = 255 - mask
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    return mask


def detect_candidates(gray: np.ndarray, return_stages: bool = False):
    white_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, TOPHAT_KERNEL)
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, TOPHAT_KERNEL)
    response = np.maximum(white_hat, black_hat)

    mask = object_mask(gray)
    response = cv2.bitwise_and(response, response, mask=mask)

    thresh_val = np.percentile(response[mask > 0], TOPHAT_PERCENTILE)
    binary = (response >= thresh_val).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, CLEAN_KERNEL)

    n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= MIN_BLOB_AREA:
            boxes.append((x, y, x + w, y + h))

    if return_stages:
        return boxes, {"mask": mask, "response": response, "binary": binary}
    return boxes


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def score_image(gt_boxes, candidates):
    matched_gt, matched_cand = set(), set()
    for gi, g in enumerate(gt_boxes):
        for ci, c in enumerate(candidates):
            if ci in matched_cand:
                continue
            if iou(g, c) >= IOU_HIT_THRESHOLD:
                matched_gt.add(gi)
                matched_cand.add(ci)
                break
    tp = len(matched_gt)
    fn = len(gt_boxes) - tp
    fp = len(candidates) - len(matched_cand)
    return tp, fp, fn


def draw_overlay(gray, gt_boxes, candidates) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in gt_boxes:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
    for x1, y1, x2, y2 in candidates:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return vis


def save_pipeline_figure(gray, stages, gt_boxes, candidates, out_path):
    vis_final = draw_overlay(gray, gt_boxes, candidates)
    panels = [
        (gray, "1. Raw X-ray", "gray"),
        (stages["response"], "2. Top-hat response\n(bright + dark blobs)", "gray"),
        (stages["binary"], "3. Thresholded mask\n(99.5th percentile)", "gray"),
        (cv2.cvtColor(vis_final, cv2.COLOR_BGR2RGB), "4. Candidates (red) vs\nground truth (green)", None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("The classical baseline pipeline, on one real GDXray image", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    gt = load_ground_truth(DATA_DIR)
    total_tp = total_fp = total_fn = 0
    per_image = []

    for idx in sorted(gt):
        img_path = DATA_DIR / f"C0001_{idx:04d}.png"
        if not img_path.exists():
            continue
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        candidates, stages = detect_candidates(gray, return_stages=True)
        gt_boxes = gt[idx]
        tp, fp, fn = score_image(gt_boxes, candidates)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_image.append({"image": idx, "gt": len(gt_boxes), "candidates": len(candidates), "tp": tp, "fp": fp, "fn": fn})

        if idx == 1:
            cv2.imwrite(str(OUT_DIR / "c0001_0001_raw.png"), gray)
            cv2.imwrite(str(OUT_DIR / "c0001_0001_overlay.png"), draw_overlay(gray, gt_boxes, candidates))
            cv2.imwrite(str(POST_IMAGES_DIR / "baseline-slider-before.png"), gray)
            cv2.imwrite(str(POST_IMAGES_DIR / "baseline-slider-after.png"), draw_overlay(gray, gt_boxes, candidates))
            save_pipeline_figure(gray, stages, gt_boxes, candidates, POST_IMAGES_DIR / "baseline-pipeline-diagram.png")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0

    summary = {
        "series": "C0001",
        "n_images": len(per_image),
        "total_gt_defects": total_tp + total_fn,
        "total_candidates_raised": total_tp + total_fp,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "iou_hit_threshold": IOU_HIT_THRESHOLD,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "per_image.json").write_text(json.dumps(per_image, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
