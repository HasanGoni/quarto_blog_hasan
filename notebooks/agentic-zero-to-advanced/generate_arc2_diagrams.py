"""Arc 2 diagrams, hand-drawn style (plt.xkcd) like the rest of the series."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = "../../posts/series/agentic-zero-to-advanced/images"
RED = "#c92a2a"


def box(ax, cx, cy, w, h, text, fc, fs=12):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.05,rounding_size=0.25",
                                fc=fc, ec="#1e1e1e", lw=2, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4)


def arrow(ax, p, q, label=None, off=(0, 0.3), rad=0.0, dashed=False, fs=10):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=22, lw=2, color="#1e1e1e",
                                 ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], label, ha="center", va="center",
                fontsize=fs, color=RED, zorder=5, bbox=dict(fc="white", ec="none", pad=1.5))


def agent_loop():
    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(14, 7.5))
        ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
        box(ax, 8, 7.4, 5.6, 1.5, "messages = [...]\neverything said and seen so far", "#fff3bf")
        box(ax, 2.6, 3.6, 4.0, 1.8, "client.chat.completions\n.create(messages, tools)\nthe model decides", "#a5d8ff")
        box(ax, 8.0, 1.7, 4.2, 1.6, "run the tool it asked for\nTOOL_FUNCS[name](**args)", "#b2f2bb")
        box(ax, 13.4, 3.6, 3.8, 1.8, "did it call\nsubmit_report?\nor hit MAX_STEPS?", "#ffd8a8")
        arrow(ax, (6.2, 7.0), (3.4, 4.55), "read it all", off=(-0.9, 0.5), rad=0.15)
        arrow(ax, (3.6, 2.7), (5.9, 1.9), "tool call", off=(-0.1, -0.55))
        arrow(ax, (10.1, 1.9), (12.4, 2.7), "result", off=(0.2, -0.5))
        arrow(ax, (13.4, 4.5), (10.2, 6.5), "no: append result,\nask again", off=(1.6, 0.4), rad=-0.15)
        ax.text(15.2, 1.6, "yes: stop,\nreturn the report", fontsize=11, color=RED, ha="center")
        arrow(ax, (14.6, 2.7), (15.0, 2.0), None)
        for x, y, t in [(0.3, 1.6, "PLANNING\nthe model picks the next step"), (0.3, 8.2, "MEMORY\nthe messages list, nothing else"),
                        (5.9, 0.35, "TOOL USE: plain Python functions"), (12.0, 7.9, "CONTROL FLOW\na for loop and an if")]:
            ax.text(x, y, t, fontsize=10, color="#1971c2", fontweight="bold", va="center")
        fig.text(0.5, 0.96, "The whole agent is this loop", ha="center", fontsize=16, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/agent-loop-diagram.png", dpi=110, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    agent_loop()
