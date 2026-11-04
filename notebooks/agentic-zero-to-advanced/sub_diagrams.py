"""Part 10: the hand-drawn picture of the three ways to use contexts (plt.xkcd, like the rest of the series)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from generate_arc2_diagrams import OUT_DIR, arrow, box


def room(ax, x, y, w, h, color="#f8f9fa"):
    """A dashed rounded rectangle: one model context, the room in which one agent can see and remember things."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05,rounding_size=0.25", fc=color, ec="#868e96",
                                lw=1.6, ls="--", zorder=0))


def three_ways():
    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(15, 7.2))
        ax.set_xlim(0, 30); ax.set_ylim(0.2, 9.4); ax.axis("off")
        # --- A: one agent, one room
        ax.text(4.2, 8.7, "A. one agent", ha="center", fontsize=14, fontweight="bold")
        room(ax, 4.2, 4.6, 7.2, 6.4)
        box(ax, 4.2, 6.9, 5.4, 1.1, "decides what to do next", "#a5d8ff", fs=11)
        box(ax, 2.3, 4.5, 3.0, 1.0, "find\ncandidates", "#d3f9d8", fs=10)
        box(ax, 6.1, 4.5, 3.0, 1.0, "measure\neach box", "#d3f9d8", fs=10)
        box(ax, 4.2, 2.4, 5.6, 1.0, "judges every box", "#ffd8a8", fs=11)
        ax.text(4.2, 0.65, "one context holds\neverything that happened", ha="center", fontsize=10, color="#495057")
        # --- B: supervisor and specialists
        ax.text(15.0, 8.7, "B. a supervisor and specialists", ha="center", fontsize=14, fontweight="bold")
        room(ax, 15.0, 7.0, 5.6, 2.2)
        box(ax, 15.0, 7.0, 4.2, 1.0, "supervisor\n(passes notes)", "#a5d8ff", fs=10)
        room(ax, 12.0, 3.3, 4.6, 3.0)
        room(ax, 18.0, 3.3, 4.6, 3.0)
        box(ax, 12.0, 3.8, 3.8, 1.0, "measurer\n(has the tools)", "#d3f9d8", fs=10)
        box(ax, 18.0, 3.8, 3.8, 1.0, "judge\n(sees only the note)", "#ffd8a8", fs=10)
        ax.text(12.0, 2.3, "own context", ha="center", fontsize=9, color="#495057")
        ax.text(18.0, 2.3, "own context", ha="center", fontsize=9, color="#495057")
        arrow(ax, (14.2, 6.3), (12.4, 4.6), rad=0.15)
        arrow(ax, (15.8, 6.3), (17.6, 4.6), rad=-0.15)
        ax.text(15.0, 0.65, "what crosses between rooms is\ntext the supervisor wrote", ha="center", fontsize=10, color="#495057")
        # --- C: fan-out
        ax.text(25.8, 8.7, "C. one sub-agent per box", ha="center", fontsize=14, fontweight="bold")
        box(ax, 25.8, 7.4, 4.4, 1.0, "code: detect + measure", "#d3f9d8", fs=10)
        for i, y in enumerate((5.6, 4.0, 2.4)):
            room(ax, 25.8, y, 4.0, 1.2)
            box(ax, 25.8, y, 3.4, 0.8, f"judge box {i}" if i < 2 else "judge box n", "#ffd8a8", fs=10)
            arrow(ax, (27.9 if i else 25.8, 6.85), (27.9 if i else 25.8, y + 0.7), rad=-0.25 if i else 0.0)
        ax.text(25.8, 0.65, "code collects the answers,\nrules and verdict stay plain Python", ha="center", fontsize=10, color="#495057")
        fig.text(0.5, 0.97, "Three ways to spend model contexts on the same inspection (a dashed room = one context)", ha="center", fontsize=15, fontweight="bold")
        fig.savefig(f"{OUT_DIR}/subagents-three-ways.png", dpi=100, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    three_ways()
