"""ELI5 whiteboard-style sketch -- matplotlib plt.xkcd(), this blog's established method."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "../../posts/series/papers/images/eneas-eli5-sketch.png"

with plt.xkcd():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.2); ax.axis("off")
    boxes = [
        (0.3, 1.8, 2.5, "Point at YOUR\ngoldfish in\nframe 1"),
        (3.1, 1.8, 2.8, "Tracker follows\nthe blob every\nframe"),
        (6.1, 1.8, 3.0, "\"Wait -- does this\nstill look like\nMY fish?\""),
        (9.4, 1.8, 2.3, "If not: stop,\ndon't switch\nfish!"),
    ]
    patches = []
    for x, y, w, label in boxes:
        h = 1.8
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", linewidth=2,
                              edgecolor="black", facecolor="#fff0e6")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
        patches.append((x, y, w, h))
    for i in range(len(patches) - 1):
        x0, y0, w0, h0 = patches[i]; x1, y1, _, h1 = patches[i + 1]
        ax.add_patch(FancyArrowPatch((x0 + w0, y0 + h0 / 2), (x1, y1 + h1 / 2),
                                      arrowstyle="-|>", mutation_scale=20, linewidth=2))
    ax.text(6, 4.3, "5 other lookalike fish swimming nearby the whole time!",
            fontsize=10, ha="center", style="italic")
    fig.text(0.5, 0.98, "ENEAS: don't just follow the blob, check it's still YOUR fish",
              ha="center", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT, dpi=140)
    print("Saved", OUT)
