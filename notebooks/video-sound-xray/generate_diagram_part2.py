"""Generate the whiteboard-style (matplotlib xkcd) ELI5 diagram for Part 2."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_PATH = (
    Path(__file__).parents[2]
    / "posts"
    / "series"
    / "video-sound-generation"
    / "images"
    / "vsxray-part2-eli5-sketch.png"
)

STEPS = [
    "Part 1: one real\nframe, then noise",
    "-> drifts to\ncolor photo",
    "Part 2: repeat real\nframe 41x as 'video'",
    "-> stays grayscale...",
    "...but still drifts\nto machinery photo",
]


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.set_xlim(0, len(STEPS))
        ax.set_ylim(0, 2)
        ax.axis("off")

        fig.text(0.5, 0.95, "Part 2: Video Conditioning vs. Part 1's Single Frame", fontsize=18, fontweight="bold", ha="center")

        box_w, box_h = 0.82, 0.95
        colors = ["white", "#f8d7da", "white", "#d4edda", "#fff3cd"]
        for i, label in enumerate(STEPS):
            x = i + 0.5
            box = FancyBboxPatch(
                (x - box_w / 2, 1.0 - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.05,rounding_size=0.08",
                linewidth=2, edgecolor="black", facecolor=colors[i],
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
