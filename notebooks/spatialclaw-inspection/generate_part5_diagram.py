"""Part 5 diagram (separate script: plt.xkcd jitter is random, so re-running generate_diagrams.py would change the older images)."""
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
    fig, ax = canvas(11, 5.8, "Part 5: the model's code runs in a box")
    ax.add_patch(FancyBboxPatch((0.3, 0.6), 4.3, 4.0, boxstyle="round,pad=0.02,rounding_size=0.15", ec=BLUE, fc="white", lw=2))
    ax.text(2.45, 4.25, "HOST (trusted)", ha="center", fontsize=12, color=BLUE, weight="bold")
    box(ax, 0.6, 3.1, 3.7, 0.8, "model + loop", INK, fs=10.5)
    box(ax, 0.6, 2.0, 3.7, 0.8, "checks every answer:\nid, shape, hard timeout", GREEN, fs=10)
    box(ax, 0.6, 0.9, 3.7, 0.8, "decide(): PASS / FAIL / REVIEW", RED, fs=10)
    ax.add_patch(FancyBboxPatch((6.2, 0.6), 4.5, 4.0, boxstyle="round,pad=0.02,rounding_size=0.15", ec=RED, fc="#fff5f5", lw=3, ls="--"))
    ax.text(8.45, 4.25, "CONTAINER (untrusted code)", ha="center", fontsize=12, color=RED, weight="bold")
    ax.text(8.45, 3.2, "no network   read-only filesystem\nno capabilities   1 GB   1 CPU\n64 processes   64 MB scratch\nnon-root user", ha="center", va="center", fontsize=10.5, color=INK)
    ax.text(8.45, 1.55, "persistent cell runner\n(numpy, scipy, pillow)", ha="center", va="center", fontsize=10.5, color=INK)
    arrow(ax, (4.6, 3.5), (6.2, 3.5), color=INK, label="one cell", ly=0.15)
    arrow(ax, (6.2, 2.4), (4.6, 2.4), color=GOLD, label="answer (untrusted)", ly=-0.4)
    fig.savefig(OUT / "part5-sandbox.png", dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote part5-sandbox.png")
