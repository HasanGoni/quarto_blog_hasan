"""ELI5 whiteboard-style sketch -- matplotlib plt.xkcd(), this blog's established method."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "../../posts/series/papers/images/sam2cd-eli5-sketch.png"

with plt.xkcd():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.3, 1.8, 2.6, "Before + after\nphoto, a few\ndots on changes"),
        (3.3, 1.8, 2.6, "SAM2 turns\neach dot into\na full outline"),
        (6.3, 1.8, 2.7, "Teacher & student\ntake turns\nimproving guesses"),
        (9.4, 1.8, 2.3, "Clean, full\nchange map"),
    ]
    patches = []
    for x, y, w, label in boxes:
        h = 1.8
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              linewidth=2, edgecolor="black", facecolor="#eafbea")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10.5)
        patches.append((x, y, w, h))

    for i in range(len(patches) - 1):
        x0, y0, w0, h0 = patches[i]
        x1, y1, w1, h1 = patches[i + 1]
        arrow = FancyArrowPatch((x0 + w0, y0 + h0 / 2), (x1, y1 + h1 / 2),
                                 arrowstyle="-|>", mutation_scale=20, linewidth=2)
        ax.add_patch(arrow)

    fig.text(0.5, 0.95, "Point-Supervised Change Detection with SAM2",
              ha="center", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT, dpi=140)
    print("Saved", OUT)
