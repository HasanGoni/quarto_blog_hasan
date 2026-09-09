"""Real SAM3-O2D2 reimplementation: object-class prompting of the real
facebook/sam3 model as a zero-shot OOD witness for a real torchvision
detector, run on (1) a real natural image, (2) a constructed OOD probe on
that same image, and (3) two real semiconductor X-ray images, zero-shot,
with no fine-tuning on either domain.

Outputs: printed real IoU / SAM3 confidence / ID-OOD decisions, plus
overlay PNGs saved to ./out/.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import requests
import torch
from PIL import Image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from transformers import Sam3Model, Sam3Processor

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {DEVICE}")


@dataclass
class Detection:
    box: np.ndarray
    label: str
    score: float


@dataclass
class Sam3Candidate:
    box: np.ndarray
    score: float


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Eq. 1 of the paper."""
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def matching_set(det: Detection, cands: list[Sam3Candidate], theta_iou=0.2):
    """Eq. 2 of the paper."""
    return [c for c in cands if iou(det.box, c.box) >= theta_iou]


def is_in_distribution(det, cands, theta_iou=0.2, theta_score=0.5):
    """Eq. 3 of the paper."""
    matches = matching_set(det, cands, theta_iou)
    best = max((c.score for c in matches), default=0.0)
    return best >= theta_score, best


print("loading facebook/sam3 ...")
sam3 = Sam3Model.from_pretrained("facebook/sam3").eval().to(DEVICE)
processor = Sam3Processor.from_pretrained("facebook/sam3")

print("loading torchvision Faster R-CNN (COCO-pretrained) ...")
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
detector = fasterrcnn_resnet50_fpn_v2(weights=weights).eval().to(DEVICE)
det_transforms = weights.transforms()
coco_names = weights.meta["categories"]


@torch.no_grad()
def run_detector(image: Image.Image, score_thresh: float = 0.7) -> list[Detection]:
    x = det_transforms(image).to(DEVICE)
    out = detector([x])[0]
    dets = []
    for box, label_idx, score in zip(out["boxes"], out["labels"], out["scores"]):
        if score < score_thresh:
            continue
        dets.append(Detection(box=box.cpu().numpy(), label=coco_names[label_idx], score=float(score)))
    return dets


@torch.no_grad()
def prompt_sam3(image: Image.Image, class_name: str) -> list[Sam3Candidate]:
    inputs = processor(images=image, text=class_name, return_tensors="pt").to(DEVICE)
    outputs = sam3(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=0.0, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    return [
        Sam3Candidate(box=b.cpu().numpy(), score=float(s))
        for b, s in zip(results["boxes"], results["scores"])
    ]


def draw_case(image, det: Detection, cands: list[Sam3Candidate], is_id: bool, matched_score: float, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    x1, y1, x2, y2 = det.box
    ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                     edgecolor="#c2410c", linewidth=2, linestyle="--",
                                     label=f'detector "{det.label}" ({det.score:.2f})'))
    for c in cands:
        cx1, cy1, cx2, cy2 = c.box
        ax.add_patch(mpatches.Rectangle((cx1, cy1), cx2 - cx1, cy2 - cy1, fill=False,
                                         edgecolor="#0d7c82", linewidth=1.5,
                                         label=f"SAM3 candidate (u={c.score:.2f})"))
    verdict = "ID" if is_id else "OOD"
    color = "#116a45" if is_id else "#b6420f"
    ax.set_title(f"{title}\nverdict={verdict}   matched SAM3 score={matched_score:.2f}",
                 color=color, fontsize=10, wrap=True)
    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=110)
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Case 1: real natural image, genuine detector + genuine SAM3 agreement (ID)
# ---------------------------------------------------------------------------
print("\n=== case 1: real ID example (natural image) ===")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # two cats
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
dets = run_detector(image, score_thresh=0.8)
print(f"detector found {len(dets)} boxes: {[(d.label, round(d.score,2)) for d in dets]}")
det = dets[0]
cands = prompt_sam3(image, det.label)
print(f"SAM3 prompted with {det.label!r} returned {len(cands)} candidates, scores={[round(c.score,2) for c in cands]}")
is_id, matched = is_in_distribution(det, cands)
print(f"-> Eq.3 verdict: {'ID' if is_id else 'OOD'} (matched SAM3 score={matched:.3f})")
cands_plot = [c for c in cands if c.score >= 0.3]  # declutter: SAM3 returns 200 raw queries/prompt
draw_case(image, det, cands_plot, is_id, matched, f'Case 1 — real detector + real SAM3 agree on "{det.label}"', "case1_id_natural.png")

# ---------------------------------------------------------------------------
# Case 2: same real image, constructed OOD probe — detector "claims" a class
# that is not actually present in that box, real SAM3 call decides for real
# ---------------------------------------------------------------------------
print("\n=== case 2: constructed OOD probe (real SAM3 call, fabricated claim) ===")
fake_claim = "elephant"  # genuinely absent from this image
fake_det = Detection(box=det.box, label=fake_claim, score=0.83)
cands2 = prompt_sam3(image, fake_claim)
print(f"SAM3 prompted with {fake_claim!r} returned {len(cands2)} candidates, scores={[round(c.score,2) for c in cands2]}")
is_id2, matched2 = is_in_distribution(fake_det, cands2)
print(f"-> Eq.3 verdict: {'ID' if is_id2 else 'OOD'} (matched SAM3 score={matched2:.3f})")
cands2_plot = [c for c in cands2 if c.score >= 0.3]
draw_case(image, fake_det, cands2_plot, is_id2, matched2, f'Case 2 — detector falsely claims "{fake_claim}" (real SAM3 refutes it)', "case2_ood_constructed.png")

# ---------------------------------------------------------------------------
# Case 3: real, openly-licensed semiconductor X-ray image, zero-shot domain
# transfer test — genuinely uncertain outcome, reported honestly either way.
#
# Image: "Using X-ray for authentication and quality control in electronics
# industry.jpg" by SarahLaserEng, Wikimedia Commons, CC BY-SA 4.0:
# https://commons.wikimedia.org/wiki/File:Using_X-ray_for_authentication_and_quality_control_in_electronics_industry.jpg
# Cropped to the single "Authentic" X-ray sub-panel (bond wires + die, no
# overlaid text/annotations) so the prompt tests the X-ray content itself.
# ---------------------------------------------------------------------------
print("\n=== case 3: real, open-licensed X-ray zero-shot domain-transfer probe ===")
xray_url = (
    "https://upload.wikimedia.org/wikipedia/commons/6/65/"
    "Using_X-ray_for_authentication_and_quality_control_in_electronics_industry.jpg"
)
headers = {"User-Agent": "paper-sam3-o2d2-repro/0.1 (https://hasangoni.quarto.pub/hasan-blog-post; contact via GitHub HasanGoni)"}
full_xray = Image.open(requests.get(xray_url, headers=headers, stream=True).raw).convert("RGB")
w, h = full_xray.size
xray_image = full_xray.crop((int(w * 0.52), int(h * 0.42), int(w * 0.99), int(h * 0.83)))

for prompt in ["chip", "substrate", "circuit board"]:
    cands3 = prompt_sam3(xray_image, prompt)
    top = max((c.score for c in cands3), default=0.0)
    print(f"prompt={prompt!r} -> {len(cands3)} SAM3 candidates, best score={top:.3f}")
    if cands3:
        best_cand = max(cands3, key=lambda c: c.score)
        fake_xray_det = Detection(box=best_cand.box, label=prompt, score=1.0)
        draw_case(
            xray_image, fake_xray_det, [best_cand], best_cand.score >= 0.5, best_cand.score,
            f'Case 3 — zero-shot SAM3 prompt "{prompt}" on a real, open-licensed X-ray',
            f"case3_xray_{prompt.replace(' ', '_')}.png",
        )

print("\ndone.")
