"""ELI5 whiteboard-style sketch for the InstEditSeg post -- matplotlib plt.xkcd(), the
established method for this blog's hand-drawn diagrams (see synthetic-data-xray/generate_diagram.py)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "../../posts/series/papers/images/insteditseg-eli5-sketch.png"

with plt.xkcd():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.5, 2, 2.4, "Endoscopy\nphoto"),
        (3.4, 2, 2.6, "Instruction:\n\"highlight the\npolyp in red\""),
        (6.5, 1.5, 2.6, "Diffusion artist\n(U-Net) with\nDINO eyes for\nextra detail"),
        (9.6, 2, 2.2, "Same photo,\npolyp glowing\nred"),
    ]
    patches = []
    for x, y, w, label in boxes:
        h = 1.6
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              linewidth=2, edgecolor="black", facecolor="#fff7e6")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)
        patches.append((x, y, w, h))

    for i in range(len(patches) - 1):
        x0, y0, w0, h0 = patches[i]
        x1, y1, w1, h1 = patches[i + 1]
        arrow = FancyArrowPatch((x0 + w0, y0 + h0 / 2), (x1, y1 + h1 / 2),
                                 arrowstyle="-|>", mutation_scale=20, linewidth=2)
        ax.add_patch(arrow)

    ax.text(7.8, 4.3, "no lines to color inside --\nit just repaints the photo!",
            fontsize=10, ha="center", style="italic")
    fig.text(0.5, 0.95, "InstEditSeg: segmentation as instructed editing",
              ha="center", fontsize=15, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT, dpi=140)
    print("Saved", OUT)
