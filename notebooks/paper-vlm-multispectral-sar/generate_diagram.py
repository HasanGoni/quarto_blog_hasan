"""ELI5 whiteboard-style sketch -- matplotlib plt.xkcd(), this blog's established method."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "../../posts/series/papers/images/vlm-sar-eli5-sketch.png"

with plt.xkcd():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    boxes = [
        (0.3, 1.8, 2.8, "Weird sensor data:\n5 spectral bands\n+ 1 radar view"),
        (3.5, 1.8, 2.9, "Chop into 6\nnamed pictures\nwith labels"),
        (6.8, 1.3, 2.7, "Hand all 6 to a\nVLM that already\nknows how to look\nat several photos"),
        (9.9, 1.8, 1.9, "\"Sparsely\nbuilt\""),
    ]
    patches = []
    for x, y, w, label in boxes:
        h = 1.8
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              linewidth=2, edgecolor="black", facecolor="#e8f4ff")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10.5)
        patches.append((x, y, w, h))

    for i in range(len(patches) - 1):
        x0, y0, w0, h0 = patches[i]
        x1, y1, w1, h1 = patches[i + 1]
        arrow = FancyArrowPatch((x0 + w0, y0 + h0 / 2), (x1, y1 + h1 / 2),
                                 arrowstyle="-|>", mutation_scale=20, linewidth=2)
        ax.add_patch(arrow)

    ax.text(6.15, 4.3, "no new eyes needed --\njust a little extra practice (LoRA)!",
            fontsize=10, ha="center", style="italic")
    fig.text(0.5, 0.95, "Lightweight VLM Adaptation for Multispectral + SAR",
              ha="center", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT, dpi=140)
    print("Saved", OUT)
