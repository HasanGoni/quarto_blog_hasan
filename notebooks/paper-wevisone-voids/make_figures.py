"""Figures for the post. Reads out/*.json(l) only; no model needed.  .venv/bin/python make_figures.py"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import common as c

IMG = Path(__file__).parents[2] / "posts/series/papers/images"
IMG.mkdir(exist_ok=True, parents=True)
cmp = json.load(open(c.OUT / "compare_test.json"))


def sketch():
    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(9, 3.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
        def box(x, y, w, h, t, col):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=col, ec="k")); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=10)
        def arr(x1, y1, x2, y2):
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=15, lw=1.5))
        box(0.2, 1.5, 2.0, 1.3, "X-ray image\n+ the word\n\"bright spot\"", "#e8f1ff")
        box(3.2, 1.5, 2.2, 1.3, "one language model\nwrites the answer\nas TEXT", "#fff3d6")
        box(6.5, 2.4, 3.2, 1.1, "<ref>void</ref><box>\n<218><575><369><652>", "#e6f7e6")
        box(6.5, 0.6, 3.2, 1.1, "or: None\n(\"no void here\")", "#fde8e8")
        arr(2.3, 2.15, 3.1, 2.15); arr(5.5, 2.3, 6.4, 2.9); arr(5.5, 2.0, 6.4, 1.2)
        ax.text(5, 0.1, "boxes are just numbers 0-1000 written like words", ha="center", fontsize=9)
        fig.savefig(IMG / "wevisone-voids-eli5.png", dpi=90, bbox_inches="tight"); plt.close(fig)


def bars():
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, m in zip(axs, ("precision", "recall", "f1")):
        for k, (lab, col) in enumerate((("zero_shot", "#c9a227"), ("fine_tuned", "#2a7fbf"))):
            r = cmp[lab]["iou0.2"]; lo, hi = r["ci"][m]
            ax.bar(k, r[m], color=col); ax.errorbar(k, r[m], yerr=[[r[m] - lo], [hi - r[m]]], color="k", capsize=4)
            ax.text(k, hi + 0.02, f"{r[m]:.2f}", ha="center", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["zero-shot", "fine-tuned"], fontsize=8); ax.set_ylim(0, 1.05); ax.set_title(m, fontsize=10)
    fig.suptitle("200 held-out test images, hit = IoU >= 0.2, bars = 95% interval over images", fontsize=9)
    fig.tight_layout(); fig.savefig(IMG / "wevisone-voids-results.png", dpi=90); plt.close(fig)


def loss():
    txt = open(c.HERE / "out_ft_stdout.txt").read()
    pts = [(int(a), float(b)) for a, b in re.findall(r"step (\d+)/700 loss ([\d.]+)", txt)]
    fig, ax = plt.subplots(figsize=(5, 3)); ax.plot(*zip(*pts), marker="o", ms=3)
    ax.axvline(350, ls=":", c="gray"); ax.text(352, max(p[1] for p in pts) * 0.9, "end of epoch 1", fontsize=8)
    ax.set_xlabel("optimizer step (8 images each)"); ax.set_ylabel("training loss"); fig.tight_layout(); fig.savefig(IMG / "wevisone-voids-loss.png", dpi=90); plt.close(fig)


def gallery(ids=(0, 3, 7, 12, 18, 25)):
    zs = {r["i"]: r for r in map(json.loads, open(c.OUT / "zs_test_raw.jsonl"))}
    ft = {r["i"]: r for r in map(json.loads, open(c.OUT / "ft_test_raw.jsonl"))}
    fig, axs = plt.subplots(3, len(ids), figsize=(2.2 * len(ids), 6.9))
    for j, i in enumerate(ids):
        img = c.load_image("test", i, 1)
        for row, (name, src) in enumerate((("truth", None), ("zero-shot", zs), ("fine-tuned", ft))):
            ax = axs[row, j]; ax.imshow(img, cmap="gray"); ax.axis("off")
            boxes = c.gt_boxes("test", i) if src is None else c.parse_boxes(src[i]["response"], img.width * 2, img.height * 2, 2)
            col = "lime" if src is None else ("orange" if name == "zero-shot" else "cyan")
            for b in boxes:
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=col, lw=1.6))
            if j == 0: ax.set_title(name, fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(IMG / "wevisone-voids-gallery.png", dpi=80); plt.close(fig)


sketch(); bars(); loss(); gallery()
print("ok")
