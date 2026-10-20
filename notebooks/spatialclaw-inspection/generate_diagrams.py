"""Whiteboard-style (plt.xkcd) diagrams for the SpatialClaw-for-inspection series."""
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
    # 1 -- the SpatialClaw loop
    fig, ax = canvas(11, 5.6, "SpatialClaw: one cell at a time, with a memory")
    box(ax, 0.3, 3.6, 2.0, 0.9, "Question\n+ image(s)", BLUE)
    box(ax, 3.0, 3.6, 2.0, 0.9, "I. Planner\n(no images)", GOLD)
    box(ax, 5.8, 3.6, 2.2, 0.9, "II. VLM writes\nONE python cell", INK)
    box(ax, 8.6, 3.6, 2.1, 0.9, "III. AST check,\nthen run", INK)
    box(ax, 8.6, 1.5, 2.1, 0.9, "IV. Feedback:\nprint + vars + show()", GREEN)
    box(ax, 5.4, 1.5, 2.6, 0.9, "optional reflection\n(OFF by default)", RED)
    box(ax, 1.6, 1.5, 3.0, 0.9, "V. ReturnAnswer()\nends the loop", BLUE)
    arrow(ax, (2.3, 4.05), (3.0, 4.05)); arrow(ax, (5.0, 4.05), (5.8, 4.05))
    arrow(ax, (8.0, 4.05), (8.6, 4.05)); arrow(ax, (9.65, 3.6), (9.65, 2.4))
    arrow(ax, (8.6, 1.95), (8.0, 1.95)); arrow(ax, (5.4, 1.95), (4.6, 1.95))
    arrow(ax, (6.7, 2.4), (6.9, 3.6), color=GOLD, rad=0.25, label="not done: next cell", lx=0.9, ly=0)
    ax.text(5.5, 0.6, "The kernel keeps every variable between cells.  That is the whole trick.",
            ha="center", fontsize=11, color=GREEN)
    fig.savefig(OUT / "spatialclaw-loop.png", dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)

    # 2 -- three action interfaces
    fig, ax = canvas(11, 5.2, "Three ways to let a model use tools")
    cols = [("Single-pass code", RED, ["write the WHOLE\nprogram", "run it once", "hope it was right"]),
            ("Structured tool call", GOLD, ["pick a tool by name", "get JSON back", "pick the next tool"]),
            ("Code in a kernel\n(SpatialClaw)", GREEN, ["write a cell", "look at the result", "write the next cell\nusing earlier variables"])]
    for i, (name, c, steps) in enumerate(cols):
        x = 0.4 + i * 3.6
        ax.text(x + 1.45, 4.0, name, ha="center", fontsize=12, weight="bold", color=c)
        for j, s in enumerate(steps):
            box(ax, x, 2.9 - j * 1.05, 2.9, 0.75, s, c, fs=10)
            if j: arrow(ax, (x + 1.45, 2.9 - (j-1) * 1.05), (x + 1.45, 2.9 - j * 1.05 + 0.75), c)
    ax.text(0.4 + 0.0, 0.1, "cannot react", color=RED, fontsize=10)
    ax.text(4.1, 0.1, "reacts, but only with the menu", color=GOLD, fontsize=10)
    ax.text(7.7, 0.1, "reacts, and can invent new glue", color=GREEN, fontsize=10)
    fig.savefig(OUT / "three-interfaces.png", dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)

    # 3 -- what we keep / swap / drop
    fig, ax = canvas(11, 5.6, "Adapting it to image inspection: keep, swap, drop")
    items = [("KEEP", GREEN, ["persistent kernel", "plan -> cell -> feedback", "show() images back", "reflection gate (turn it ON)"]),
             ("SWAP", GOLD, ["SAM3 -> our detector / SAM", "big hosted VLM -> one local VLM", "ReturnAnswer -> typed verdict", "AST check -> real sandbox"]),
             ("DROP", RED, ["Depth-Anything-3 / Pi3", "video + multi-view code", "SLURM chains, 3 services", "20 benchmark loaders"])]
    for i, (h, c, rows) in enumerate(items):
        x = 0.4 + i * 3.6
        ax.text(x + 1.45, 4.3, h, ha="center", fontsize=16, weight="bold", color=c)
        for j, r in enumerate(rows):
            box(ax, x, 3.3 - j * 0.95, 2.9, 0.7, r, c, fs=9.5)
    fig.savefig(OUT / "keep-swap-drop.png", dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
