"""How good is the part_mask() tool against the exact part masks of the DEV split? (Run before the agent runs.)"""
import sys
from pathlib import Path
import numpy as np
exec((Path(__file__).parent / "kernel_tools.py").read_text())
d = np.load(Path(__file__).resolve().parents[2] / "agentic-factory-segmentation/data/void_dataset_v2/val.npz")
iou = []
for i in range(600):
    gt = d["part_masks"][i] > 0
    pm = part_mask(d["images"][i])
    iou.append((pm & gt).sum() / (pm | gt).sum())
iou = np.array(iou)
print(f"part_mask vs exact part mask, 600 DEV images: mean IoU {iou.mean():.4f}, median {np.median(iou):.4f}, min {iou.min():.2f}")
print(f"images with IoU < 0.95: {(iou < 0.95).sum()} of 600 ({100*(iou<0.95).mean():.1f} %)")
print("first 12 images (the ones the agent sees): IoU", np.round(iou[:12], 3).tolist())
print("worst 3: val_%05d %.2f, val_%05d %.2f, val_%05d %.2f" % tuple(x for i in np.argsort(iou)[:3] for x in (i, iou[i])))
# the cause: the exact part mask has big through-holes (cut-outs); fill_holes closes them
import scipy.ndimage as ndi
def has_cutout(g):
    return ndi.binary_fill_holes(g).sum() - g.sum() > 200
gt_has_hole = np.array([has_cutout(d["part_masks"][i] > 0) for i in range(600)])
print(f"exact part masks with an enclosed hole of > 200 px (cut-out the silhouette does not include): {gt_has_hole.sum()} of 600")
print(f"  of the {(iou<0.95).sum()} low-IoU images, {(gt_has_hole & (iou<0.95)).sum()} have such a hole")
