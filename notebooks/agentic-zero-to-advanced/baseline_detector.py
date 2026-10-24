"""Plain, non-agentic classical-CV baseline for the synthetic void dataset.

No LLM, no agent, no learning from labels -- white top-hat morphology, a part mask, one noise-scaled
threshold, connected components. Scored against the exact void boxes. This is Part 2's baseline:
the number every later, agentic addition in this series gets compared against.

Protocol (void_data.py): the one free number `K` is chosen on DEV (val split); the HELD-OUT
split (test) is scored once with it.
"""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import void_data as vd

OUT_DIR = Path(__file__).parent / "out"
POST_IMAGES_DIR = Path(__file__).parent.parent.parent / "posts/series/agentic-zero-to-advanced/images"
OUT_DIR.mkdir(exist_ok=True)
POST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

TOPHAT_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))  # bigger than the biggest void (24 px)
CLEAN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
EDGE_ERODE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
MIN_BLOB_AREA = 4
IOU_HIT_THRESHOLD = 0.2
K_GUESS = 4.0  # my first guess, before looking at any score
K_DEV = 8.0  # the value the dev sweep picked (out/baseline_output.txt); used by every later part
K_GRID = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)


def part_mask(gray: np.ndarray) -> np.ndarray:
    """Otsu-threshold the part (dark) away from the air (bright), then erode so the part's
    own edge, a very strong and very boring top-hat response, is not a candidate defect."""
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.erode(m, EDGE_ERODE)


def response_map(gray: np.ndarray):
    """White top-hat: small regions brighter than their surroundings, which is what a void is
    (less material in the way, so more X-rays get through)."""
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.morphologyEx(smooth, cv2.MORPH_TOPHAT, TOPHAT_KERNEL).astype(np.float32)


def detect_candidates(gray: np.ndarray, k: float = K_GUESS, return_stages: bool = False):
    mask = part_mask(gray)
    resp = response_map(gray) * (mask > 0)
    inside = resp[mask > 0]
    # robust noise scale of the response inside the part (median absolute deviation)
    sigma = 1.4826 * np.median(np.abs(inside - np.median(inside))) if inside.size else 1.0
    binary = (resp >= k * max(sigma, 1e-3)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, CLEAN_KERNEL)

    n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    boxes = [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, a in stats[1:] if a >= MIN_BLOB_AREA]
    if return_stages:
        return boxes, {"mask": mask, "response": resp, "binary": binary, "sigma": float(sigma)}
    return boxes


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt_boxes, candidates):
    """Greedy one-to-one match; returns (matched_gt_indices, matched_candidate_indices)."""
    pairs = sorted(
        ((iou(g, c), gi, ci) for gi, g in enumerate(gt_boxes) for ci, c in enumerate(candidates)),
        reverse=True,
    )
    mg, mc = set(), set()
    for s, gi, ci in pairs:
        if s < IOU_HIT_THRESHOLD:
            break
        if gi in mg or ci in mc:
            continue
        mg.add(gi)
        mc.add(ci)
    return mg, mc


def run_split(split: str, k: float, keep_images: tuple[int, ...] = ()):
    """Score one split. Per void: hit or miss plus its contrast and size, so the misses can be sliced."""
    tp = fp = fn = 0
    void_rows, per_image = [], []
    px_inter = px_pred = px_true = 0
    examples = {}
    for i in range(len(vd.split_ids(split))):
        gray = vd.load_image(split, i)
        voids = vd.load_voids(split, i)
        gt_boxes = [tuple(v["bbox"]) for v in voids]
        cands, st = detect_candidates(gray, k, return_stages=True)
        mg, mc = match(gt_boxes, cands)
        tp += len(mg)
        fn += len(gt_boxes) - len(mg)
        fp += len(cands) - len(mc)
        for gi, v in enumerate(voids):
            void_rows.append({"image": i, "hit": gi in mg, "contrast": v["contrast"], "eq_diam_px": v["eq_diam_px"]})
        per_image.append({"image": i, "gt": len(gt_boxes), "candidates": len(cands), "tp": len(mg)})
        vm, _ = vd.load_masks(split, i)
        pred = st["binary"] > 0
        px_inter += int((pred & (vm > 0)).sum())
        px_pred += int(pred.sum())
        px_true += int((vm > 0).sum())
        if i in keep_images:
            examples[i] = (gray, gt_boxes, cands, st)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    dice = 2 * px_inter / (px_pred + px_true) if px_pred + px_true else 0.0
    summary = {
        "split": split, "k": k, "n_images": len(per_image), "gt_voids": tp + fn, "candidates": tp + fp,
        "tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 3), "recall": round(rec, 3),
        "f1": round(f1, 3), "pixel_dice": round(dice, 3),
    }
    return summary, void_rows, per_image, examples


def recall_by(void_rows, key, edges, labels):
    out = {}
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        rows = [r for r in void_rows if lo <= r[key] < hi]
        out[lab] = {"n": len(rows), "recall": round(sum(r["hit"] for r in rows) / len(rows), 3) if rows else None}
    return out


def draw_overlay(gray, gt_boxes, candidates, scale=3):
    vis = cv2.cvtColor(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in gt_boxes:
        cv2.rectangle(vis, (x1 * scale - 2, y1 * scale - 2), (x2 * scale + 2, y2 * scale + 2), (0, 200, 0), 2)
    for x1, y1, x2, y2 in candidates:
        cv2.rectangle(vis, (x1 * scale, y1 * scale), (x2 * scale, y2 * scale), (0, 0, 255), 1)
    return vis


def save_pipeline_figure(gray, stages, gt_boxes, candidates, out_path, title):
    vis = cv2.cvtColor(draw_overlay(gray, gt_boxes, candidates), cv2.COLOR_BGR2RGB)
    panels = [
        (gray, "1. The image", "gray"),
        (stages["mask"], "2. Part mask\n(edge eroded away)", "gray"),
        (stages["response"], "3. White top-hat response\n(brighter than surroundings)", "gray"),
        (stages["binary"], "4. Above K x noise level", "gray"),
        (vis, "5. Candidates (red)\nvs exact voids (green)", None),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    for ax, (img, t, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def save_recall_figure(by_contrast, by_size, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, data, xl in ((axes[0], by_contrast, "void contrast (how much brighter than around it)"),
                         (axes[1], by_size, "void equivalent diameter")):
        labs = list(data)
        vals = [data[l]["recall"] or 0 for l in labs]
        ax.bar(range(len(labs)), vals, color="#3b82c4")
        for j, l in enumerate(labs):
            ax.text(j, vals[j] + 0.02, f"{vals[j]:.0%}\nn={data[l]['n']}", ha="center", fontsize=9)
        ax.set_xticks(range(len(labs)), labs, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel(xl)
        ax.set_ylabel("recall")
    fig.suptitle("Where the baseline misses (held-out test split)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def pick_example(split="val"):
    """A deterministic example with 3 voids of mixed contrast, so the figure is not cherry-picked
    by outcome: the first dev image with exactly three voids."""
    for i in range(len(vd.split_ids(split))):
        if len(vd.load_voids(split, i)) == 3:
            return i
    return 0


def main():
    # 1. choose K on DEV
    dev_grid = {}
    for k in K_GRID:
        s, *_ = run_split("val", k)
        dev_grid[k] = s
        print(f"dev  K={k:<4} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} cand={s['candidates']}")
    k_best = max(dev_grid, key=lambda k: dev_grid[k]["f1"])
    print("chosen K on dev:", k_best)

    # 2. HELD-OUT, once each for the guess and the dev-chosen K
    ex_idx = pick_example("test")
    test_guess, *_ = run_split("test", K_GUESS)
    test_best, void_rows, per_image, examples = run_split("test", k_best, keep_images=(ex_idx,))
    print("test (K guess)  :", test_guess)
    print("test (K on dev) :", test_best)

    by_contrast = recall_by(void_rows, "contrast", [0, 0.05, 0.15, 10], ["< 0.05", "0.05-0.15", ">= 0.15"])
    by_size = recall_by(void_rows, "eq_diam_px", [0, 6, 12, 100], ["< 6 px", "6-12 px", ">= 12 px"])
    print("recall by contrast:", by_contrast)
    print("recall by size    :", by_size)
    save_recall_figure(by_contrast, by_size, POST_IMAGES_DIR / "baseline-recall-slices.png")

    gray, gt_boxes, cands, st = examples[ex_idx]
    save_pipeline_figure(gray, st, gt_boxes, cands, POST_IMAGES_DIR / "baseline-pipeline-diagram.png",
                         f"The classical baseline on one held-out image (test_{ex_idx:05d}), K = {k_best}")
    cv2.imwrite(str(POST_IMAGES_DIR / "baseline-slider-before.png"),
                cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(POST_IMAGES_DIR / "baseline-slider-after.png"), draw_overlay(gray, gt_boxes, cands))

    out = {
        "protocol": "K chosen on val (dev); test scored once per K",
        "iou_hit_threshold": IOU_HIT_THRESHOLD,
        "dev_grid": {str(k): v for k, v in dev_grid.items()},
        "k_guess": K_GUESS, "k_best": k_best,
        "test_k_guess": test_guess, "test_k_dev": test_best,
        "recall_by_contrast": by_contrast, "recall_by_size": by_size,
        "example_image": f"test_{ex_idx:05d}",
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / "per_image.json").write_text(json.dumps(per_image, indent=2))


if __name__ == "__main__":
    main()
