"""Excalidraw-style architecture diagrams for the HPMA post (matplotlib xkcd mode)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "../../posts/series/papers/images/"
C = {"sam": "#a5d8ff", "hpma": "#b2f2bb", "bank": "#ffd8a8", "in": "#f1f3f5", "loss": "#ffc9c9"}
RED = "#c92a2a"


def box(ax, cx, cy, w, h, text, fc, fs=12):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.05,rounding_size=0.25",
                                fc=fc, ec="#1e1e1e", lw=2, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4)


def zone(ax, x0, y0, x1, y1, title, color):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.05,rounding_size=0.4",
                                fc="none", ec=color, lw=2.2, ls="--", zorder=1))
    ax.text((x0 + x1) / 2, y1 + 0.3, title, ha="center", fontsize=14, fontweight="bold", color=color)


def arrow(ax, p, q, label=None, off=(0, 0.3), dashed=False, rad=0.0, fs=10):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=22, lw=2, color="#1e1e1e",
                                 ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], label, ha="center", va="center",
                fontsize=fs, color=RED, zorder=5, bbox=dict(fc="white", ec="none", pad=1.5))


def canvas(w, h, xl, yl):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xl); ax.set_ylim(0, yl); ax.axis("off")
    return fig, ax


with plt.xkcd():
    # ---------- Diagram 1: where each adapter plugs into SAM3 ----------
    fig, ax = canvas(17, 10, 20, 12)
    zone(ax, 0.3, 5.4, 19.7, 10.9, "facebook/sam3  (frozen, 840M params)", "#1971c2")
    zone(ax, 0.3, 0.1, 19.7, 4.4, "HPMA additions  (trainable)", "#2f9e44")

    box(ax, 1.9, 9.4, 2.8, 1.0, "category text", C["in"])
    box(ax, 5.0, 9.4, 3.0, 1.2, "text_encoder +\ntext_projection", C["sam"])
    box(ax, 8.6, 9.4, 3.4, 1.2, "text tokens Tc\n(B, 32, 256)", C["sam"])
    box(ax, 1.9, 6.6, 2.8, 1.0, "pixel_values\n1008×1008", C["in"], fs=11)
    box(ax, 5.0, 6.6, 3.0, 1.2, "vision_encoder\n(Sam3VisionModel)", C["sam"], fs=11)
    box(ax, 8.6, 6.6, 3.4, 1.7, "FPN features\nF0 288² local\nF1 144² structural\nF2 72² global", C["sam"], fs=11)
    box(ax, 12.0, 8.0, 2.6, 1.2, "detr_encoder", C["sam"])
    box(ax, 15.0, 8.0, 2.6, 1.2, "detr_decoder\n(200 queries)", C["sam"], fs=11)
    box(ax, 17.95, 8.0, 2.2, 1.2, "mask_\ndecoder", C["sam"])
    box(ax, 17.95, 6.1, 2.4, 0.9, "pred_masks", C["in"])
    for p, q in [((3.3, 9.4), (3.5, 9.4)), ((6.5, 9.4), (6.9, 9.4)), ((3.3, 6.6), (3.5, 6.6)), ((6.5, 6.6), (6.9, 6.6)),
                 ((13.3, 8.0), (13.7, 8.0)), ((16.3, 8.0), (16.85, 8.0)), ((17.95, 7.4), (17.95, 6.55))]:
        arrow(ax, p, q)
    arrow(ax, (10.3, 9.4), (11.3, 8.6), "Tc", off=(-0.2, 0.4))
    arrow(ax, (10.3, 6.6), (11.3, 7.4), "F2", off=(-0.2, -0.4))

    box(ax, 7.4, 3.5, 4.4, 1.4, "local_alignment_loss\ncosine, train-time only\n+0.01·L_align → total loss", C["loss"], fs=11)
    box(ax, 12.0, 3.5, 3.4, 1.4, "GlobalCoupling-\nAdapter\ncross-attn, Tc = queries", C["hpma"], fs=10)
    box(ax, 15.9, 3.5, 3.4, 1.4, "StructuralCoupling-\nAdapter\nadditive MLP delta", C["hpma"], fs=10)
    box(ax, 9.8, 1.0, 17.0, 1.0, "Frozen Prototype Bank   P: 3 scales × 7 categories × K=4 × 256d   (built offline via K-Means)", C["bank"], fs=12)
    for x in (7.4, 12.0, 15.9):
        arrow(ax, (x, 1.5), (x, 2.8))
    arrow(ax, (8.0, 5.8), (7.6, 4.2), "F0, masked-pooled", off=(-2.3, 0.0), dashed=True)
    arrow(ax, (12.0, 4.2), (11.9, 7.4), "T̃c replaces Tc\n(patched get_text_features)", off=(1.65, -0.8), dashed=True)
    arrow(ax, (16.2, 4.2), (15.1, 7.4), "query_embed.weight + delta\n(subclassed forward)", off=(2.4, -0.8), dashed=True)
    fig.tight_layout(); fig.savefig(OUT + "hpma-architecture.png", dpi=110, facecolor="white"); plt.close(fig)

    # ---------- Diagram 2: prototype construction ----------
    fig, ax = canvas(17, 6.2, 20, 7)
    box(ax, 1.9, 5.0, 3.0, 1.5, "Real EndoVis2017\nimages\n(tyluan/Endovis2017)", C["in"], fs=11)
    box(ax, 5.6, 5.0, 3.0, 1.5, "Real SAM3\nvision_encoder\nforward pass", C["sam"], fs=11)
    box(ax, 9.3, 5.0, 3.0, 1.5, "3 FPN feature\nmaps per image", C["sam"], fs=11)
    box(ax, 12.9, 5.0, 3.2, 1.5, "Masked average pool\nper category,\nper scale", C["hpma"], fs=11)
    box(ax, 16.4, 5.0, 2.8, 1.5, "K-Means\nK=4\n(scikit-learn)", C["hpma"], fs=11)
    box(ax, 1.9, 1.8, 3.0, 1.5, "Real per-pixel\nground-truth masks", C["in"], fs=11)
    box(ax, 6.5, 1.8, 3.6, 1.5, "Resize mask to each\nscale's H×W (nearest)", C["hpma"], fs=11)
    box(ax, 16.4, 1.8, 2.8, 1.5, "Frozen\nPrototype Bank\n3 × 7 × 4 × 256", C["bank"], fs=11)
    box(ax, 11.0, 1.8, 3.2, 1.5, "Coupling adapters\n(read-only access)", C["hpma"], fs=11)
    for p, q in [((3.4, 5.0), (4.1, 5.0)), ((7.1, 5.0), (7.8, 5.0)), ((10.8, 5.0), (11.3, 5.0)),
                 ((14.5, 5.0), (15.0, 5.0)), ((3.4, 1.8), (4.7, 1.8)), ((16.4, 4.2), (16.4, 2.6))]:
        arrow(ax, p, q)
    arrow(ax, (7.5, 2.5), (11.9, 4.2), rad=-0.15)
    arrow(ax, (14.95, 1.8), (12.6, 1.8), "never updated by\ngradient descent", off=(0, 0.75), dashed=True)
    fig.tight_layout(); fig.savefig(OUT + "hpma-prototype-construction.png", dpi=110, facecolor="white"); plt.close(fig)
