# generate_diagram_part5.py
"""Generate the whiteboard-style (matplotlib xkcd) ELI5 diagram for Part 5."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_PATH = Path(__file__).parents[2] / "posts" / "series" / "synthetic-data" / "images" / "synth-xray-joint-eli5-sketch.png"

STEPS = [
    "Real (image, mask)\npairs, stacked",
    "Small UNet,\nfrom scratch",
    "Learn to denoise\nboth together",
    "Start from\npure noise",
    "Reverse-diffuse:\nno mask input at all",
    "New image +\nits own new mask",
]


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.set_xlim(0, len(STEPS))
        ax.set_ylim(0, 2)
        ax.axis("off")

        fig.text(0.5, 0.95, "Joint Image + Mask Diffusion", fontsize=20, fontweight="bold", ha="center")

        box_w, box_h = 0.8, 0.9
        for i, label in enumerate(STEPS):
            x = i + 0.5
            box = FancyBboxPatch(
                (x - box_w / 2, 1.0 - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.05,rounding_size=0.08",
                linewidth=2, edgecolor="black", facecolor="white",
            )
            ax.add_patch(box)
            ax.text(x, 1.0, label, ha="center", va="center", fontsize=10)
            if i < len(STEPS) - 1:
                arrow = FancyArrowPatch(
                    (x + box_w / 2, 1.0), (x + 1 - box_w / 2, 1.0),
                    arrowstyle="-|>", mutation_scale=20, linewidth=2, color="black",
                )
                ax.add_patch(arrow)

        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(OUT_PATH, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
