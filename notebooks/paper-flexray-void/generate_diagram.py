"""ELI5 whiteboard-style sketch -- matplotlib plt.xkcd(), this blog's established method.

FleXray's data engine, as a left-to-right pipeline: labeled 3D CT -> random-angle ray
projection -> synthetic X-ray (DRR) -> generative appearance editing -> QC filter ->
harmonized dense labels feeding one generalist segmenter.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "../../posts/series/papers/images/flexray-eli5-sketch.png"

with plt.xkcd():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.2, 1.6, 2.1, "Labeled 3D CT\n(every organ\nalready outlined)"),
        (2.6, 1.6, 2.1, "Shoot rays\nthrough it from\na random angle"),
        (5.0, 1.6, 2.1, "Flat 2D X-ray\n+ free dense\nlabels"),
        (7.4, 1.6, 2.1, "Generative editor\nadds scanner grime,\ntext, scatter"),
        (9.8, 1.6, 2.9, "Quality check\ndrops the ~4%\nthat drifted"),
    ]
    patches = []
    for x, y, w, label in boxes:
        h = 1.9
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              linewidth=2, edgecolor="black", facecolor="#e6f0fd")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5)
        patches.append((x, y, w, h))

    for i in range(len(patches) - 1):
        x0, y0, w0, h0 = patches[i]
        x1, y1, w1, h1 = patches[i + 1]
        arrow = FancyArrowPatch((x0 + w0, y0 + h0 / 2), (x1, y1 + h1 / 2),
                                 arrowstyle="-|>", mutation_scale=20, linewidth=2)
        ax.add_patch(arrow)

    fig.text(0.5, 0.96, "FleXray: one 3D CT becomes infinite labeled 2D X-rays",
              ha="center", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.04,
              "the trick: you only ever need the CT once -- every angle, every crop is free supervision",
              ha="center", fontsize=10, style="italic")

    plt.tight_layout()
    plt.savefig(OUT, dpi=140)
    print("Saved", OUT)
