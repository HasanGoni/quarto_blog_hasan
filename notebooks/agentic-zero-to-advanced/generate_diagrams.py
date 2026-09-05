"""ELI5 whiteboard-style sketches -- matplotlib plt.xkcd(), this blog's established method.

Generates a before/after pair (plain LLM call vs. agentic loop) sized identically so they
work in the repo's standard interactive before/after slider.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = "../../posts/series/agentic-zero-to-advanced/images"
FIGSIZE = (9, 6)
XLIM, YLIM = (0, 9), (0, 6)


def box(ax, x, y, w, h, label, face="#fdeee6"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                        linewidth=2, edgecolor="black", facecolor=face)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)
    return x, y, w, h


def arrow(ax, p0, p1, **kw):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=22, linewidth=2, **kw)
    ax.add_patch(a)


def plain_call():
    with plt.xkcd():
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.axis("off")

        u = box(ax, 0.4, 2.6, 1.8, 1.2, "You")
        p = box(ax, 3.0, 2.6, 2.2, 1.2, "\"Is this\nX-ray a\ndefect?\"")
        l = box(ax, 6.0, 2.6, 2.4, 1.2, "LLM writes\nan answer")
        arrow(ax, (u[0] + u[2], u[1] + u[3] / 2), (p[0], p[1] + p[3] / 2))
        arrow(ax, (p[0] + p[2], p[1] + p[3] / 2), (l[0], l[1] + l[3] / 2))

        ax.text(7.2, 1.1, "That's it -- one shot.\nNo action, no check,\nno next step.",
                ha="center", va="center", fontsize=10, style="italic")

        fig.text(0.5, 0.94, "A plain LLM call", ha="center", fontsize=15, fontweight="bold")
        plt.tight_layout()
        out = f"{OUT_DIR}/agentic-loop-slider-before.png"
        plt.savefig(out, dpi=140)
        print("Saved", out)


def agentic_loop():
    with plt.xkcd():
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.axis("off")

        decide = box(ax, 3.4, 3.6, 2.2, 1.2, "LLM\ndecides", face="#e6f0fd")
        act = box(ax, 6.2, 1.6, 2.2, 1.2, "Acts:\ncalls real\ndetector tool")
        observe = box(ax, 0.6, 1.6, 2.2, 1.2, "Observes\nthe tool's\nreal result")

        arrow(ax, (decide[0] + decide[2], decide[1] + decide[3] / 2 - 0.2),
              (act[0] + act[2] / 2, act[1] + act[3]))
        arrow(ax, (act[0], act[1] + act[3] / 2), (observe[0] + observe[2], observe[1] + observe[3] / 2))
        arrow(ax, (observe[0] + observe[2] / 2, observe[1] + observe[3]),
              (decide[0], decide[1] + decide[3] / 2 - 0.2))

        ax.text(4.5, 5.4, "loops until the decision\nis \"good enough, stop\"",
                ha="center", va="center", fontsize=10, style="italic")

        fig.text(0.5, 0.94, "An agentic loop", ha="center", fontsize=15, fontweight="bold")
        plt.tight_layout()
        out = f"{OUT_DIR}/agentic-loop-slider-after.png"
        plt.savefig(out, dpi=140)
        print("Saved", out)


if __name__ == "__main__":
    plain_call()
    agentic_loop()
