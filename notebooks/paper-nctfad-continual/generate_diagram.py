"""ELI5 whiteboard-style sketch -- matplotlib plt.xkcd(), this blog's established method."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "../../posts/series/papers/images/nctfad-eli5-sketch.png"

with plt.xkcd():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    boxes = [
        (0.3, 1.8, 2.7, "New product\nshows up on the\nline (no warning!)"),
        (3.3, 1.8, 2.9, "Give it its own\nfixed corner of\nthe room (prototype)"),
        (6.5, 1.8, 2.8, "Old products keep\ntheir OWN corners\n-- room never reshuffles"),
        (9.6, 1.8, 2.0, "Nothing\nforgotten"),
    ]
    patches = []
    for x, y, w, label in boxes:
        h = 1.8
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", linewidth=2,
                              edgecolor="black", facecolor="#e6f7ff")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
        patches.append((x, y, w, h))
    for i in range(len(patches) - 1):
        x0, y0, w0, h0 = patches[i]; x1, y1, _, h1 = patches[i + 1]
        ax.add_patch(FancyArrowPatch((x0 + w0, y0 + h0 / 2), (x1, y1 + h1 / 2),
                                      arrowstyle="-|>", mutation_scale=20, linewidth=2))
    fig.text(0.5, 0.95, "NC-TFAD: a fixed room, so old lessons don't get bumped",
              ha="center", fontsize=13.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT, dpi=140)
    print("Saved", OUT)
