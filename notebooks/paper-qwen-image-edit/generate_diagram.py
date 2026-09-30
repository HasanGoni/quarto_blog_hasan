"""ELI5 whiteboard-style diagram for the Qwen-Image-2.1 mask-edit post.
Standard method: plt.xkcd() + FancyBboxPatch + FancyArrowPatch (see repo memory
eli5-whiteboard-diagram-method).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).parent.parent.parent / "posts/series/papers/images/qwen-image-edit-eli5-sketch.png"

with plt.xkcd():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    boxes = [
        (0.4, "A real clean\nX-ray patch\n(no defect here)"),
        (3.1, "Draw a circle:\n'put a void\nin here'"),
        (5.9, "Qwen-Image-2.1\nfills in the\ncircle"),
        (8.6, "One new\n'defective'\ntraining image"),
    ]
    w, h, y = 2.1, 2.0, 1.1
    centers = []
    for x, text in boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                              linewidth=2, edgecolor="black", facecolor="#fef6e4")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)
        centers.append((x, x + w, y + h / 2))

    for (x0, x1, cy0), (x2, x3, cy1) in zip(centers[:-1], centers[1:]):
        arrow = FancyArrowPatch((x1 + 0.05, cy0), (x2 - 0.05, cy1),
                                 arrowstyle="-|>", mutation_scale=20, linewidth=2)
        ax.add_patch(arrow)

    ax.text(5.5, 3.7, "It's like handing someone a photo, circling a spot with a crayon,\n"
                      "and saying 'draw a scratch right there' -- they don't touch anything else.",
            ha="center", va="center", fontsize=10, style="italic")
    ax.text(5.5, 0.55, "The catch (this post's whole point): does that new image actually\n"
                       "look like a REAL defect, or just a plausible-looking guess?",
            ha="center", va="center", fontsize=10, color="#a33")

    fig.suptitle("Mask-conditioned editing as a rare-defect data generator", fontsize=14, y=0.99)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"Saved {OUT}")
