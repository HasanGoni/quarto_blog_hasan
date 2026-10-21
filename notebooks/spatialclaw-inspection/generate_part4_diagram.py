"""Part 4 diagram (separate script: plt.xkcd jitter is random, so re-running generate_diagrams.py would change the older images)."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = pathlib.Path(__file__).resolve().parents[2] / "posts/series/spatialclaw-inspection/images"
OUT.mkdir(parents=True, exist_ok=True)

INK, BLUE, GREEN, RED, GOLD = "#222", "#1971c2", "#2f9e44", "#e03131", "#f08c00"

def box(ax, x, y, w, h, text, color=INK, fill="white", fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                ec=color, fc=fill, lw=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=INK)

def arrow(ax, p, q, color=INK, rad=0.0, label=None, lx=0, ly=0.18):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=2,
                                 color=color, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p[0]+q[0])/2 + lx, (p[1]+q[1])/2 + ly, label, ha="center", fontsize=9, color=color)

def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    ax.text(w/2, h - 0.35, title, ha="center", fontsize=15, weight="bold", color=INK)
    return fig, ax

with plt.xkcd():
    fig, ax = canvas(11, 5.8, "Part 4: eyes and hands for the same loop")
    box(ax, 0.3, 3.7, 2.2, 1.0, "model writes\none python cell", INK)
    box(ax, 3.2, 3.7, 2.4, 1.0, "persistent kernel\n(img, np, ndi)", INK)
    box(ax, 6.3, 3.7, 1.9, 1.0, "text:\nstdout, vars", GREEN)
    box(ax, 8.7, 3.7, 2.0, 1.0, "PICTURE:\nshow(array)", GOLD)
    arrow(ax, (2.5, 4.2), (3.2, 4.2)); arrow(ax, (5.6, 4.2), (6.3, 4.2)); arrow(ax, (8.2, 4.2), (8.7, 4.2))
    arrow(ax, (9.7, 3.7), (1.4, 3.7), color=GOLD, rad=-0.3, label="both go back to the model", ly=-1.6)
    box(ax, 0.3, 0.5, 5.0, 1.3, "hands: part_mask, label_blobs,\noverlay, crop  (numpy + scipy)\nbuilding blocks, no detector", BLUE, fs=10)
    box(ax, 5.9, 0.5, 4.8, 1.3, "the rule layer is unchanged:\nthe model submits a number,\nplain code says PASS / FAIL / REVIEW", RED, fs=10)
    fig.savefig(OUT / "part4-eyes-and-hands.png", dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote part4-eyes-and-hands.png")
