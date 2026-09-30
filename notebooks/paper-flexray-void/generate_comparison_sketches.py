"""Two matching xkcd-style sketches for the before/after slider -- this is a conceptual
applicability post with no captured experiment, so per this blog's established pattern
(see Agentic Part 1), the interactive element is two illustrative sketches, not real screenshots.

Left: FleXray's actual data engine (whole-body CT -> anatomy labels).
Right: the same engine re-imagined for casting/semiconductor void detection, with the one
missing ingredient called out explicitly (no abundant labeled 3D source in this domain).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def panel(ax, title, boxes, missing_idx=None):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    patches = []
    for i, (x, y, w, label) in enumerate(boxes):
        h = 1.7
        face = "#fbe1e1" if i == missing_idx else "#e6f0fd"
        edge = "#a33" if i == missing_idx else "black"
        style = "dashed" if i == missing_idx else "solid"
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              linewidth=2.2, edgecolor=edge, facecolor=face,
                              linestyle=style)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5)
        patches.append((x, y, w, h))
    for i in range(len(patches) - 1):
        x0, y0, w0, h0 = patches[i]
        x1, y1, w1, h1 = patches[i + 1]
        arrow = FancyArrowPatch((x0 + w0 / 2, y0), (x1 + w1 / 2, y1 + h1),
                                 arrowstyle="-|>", mutation_scale=18, linewidth=2,
                                 connectionstyle="arc3,rad=0.0")
        ax.add_patch(arrow)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)


with plt.xkcd():
    fig1, ax1 = plt.subplots(figsize=(6, 7))
    panel(ax1, "FleXray, as published",
          [
              (1.0, 4.1, 8.0, "1,597 labeled whole-body CTs\n(organs already outlined)"),
              (1.0, 2.2, 8.0, "Project at random angles\n-> labeled synthetic X-ray"),
              (1.0, 0.3, 8.0, "Generalist segmenter,\n60 anatomical structures"),
          ])
    plt.tight_layout()
    plt.savefig("../../posts/series/papers/images/flexray-void-slider-before.png", dpi=140)
    print("Saved before panel")

    fig2, ax2 = plt.subplots(figsize=(6, 7))
    panel(ax2, "Adapted for void/casting X-ray",
          [
              (1.0, 4.1, 8.0, "Micro-CT of a handful of\ncastings, voids outlined once"),
              (1.0, 2.2, 8.0, "Project at random angles\n-> labeled synthetic X-ray"),
              (1.0, 0.3, 8.0, "Fine-tune on real casting\nX-rays with the void class"),
          ], missing_idx=0)
    plt.tight_layout()
    plt.savefig("../../posts/series/papers/images/flexray-void-slider-after.png", dpi=140)
    print("Saved after panel (red dashed box = the part almost nobody in industrial X-ray has)")
