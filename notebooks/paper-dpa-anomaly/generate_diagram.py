"""ELI5 whiteboard-style sketch -- matplotlib plt.xkcd(), this blog's established method."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "../../posts/series/papers/images/dpa-eli5-sketch.png"

with plt.xkcd():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.3, 1.8, 2.6, "Scratch seen\non a red\ntoy car"),
        (3.3, 1.8, 2.8, "\"Scratch-ness\"\nlearned, not tied\nto red or car-shaped"),
        (6.5, 1.8, 2.8, "Blue toy (never\nscratched before)\n+ a mask"),
        (9.7, 1.8, 2.0, "Believable\nscratch,\nnow on blue"),
    ]
    patches = []
    for x, y, w, label in boxes:
        h = 1.8
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              linewidth=2, edgecolor="black", facecolor="#fdeee6")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
        patches.append((x, y, w, h))

    for i in range(len(patches) - 1):
        x0, y0, w0, h0 = patches[i]
        x1, y1, w1, h1 = patches[i + 1]
        arrow = FancyArrowPatch((x0 + w0, y0 + h0 / 2), (x1, y1 + h1 / 2),
                                 arrowstyle="-|>", mutation_scale=20, linewidth=2)
        ax.add_patch(arrow)

    fig.text(0.5, 0.95, "DPA: transferring anomalies across products",
              ha="center", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT, dpi=140)
    print("Saved", OUT)
