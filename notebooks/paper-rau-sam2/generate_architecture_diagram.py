"""Excalidraw-style architecture diagram for the RAU post (matplotlib xkcd mode)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "../../posts/series/papers/images/rau-architecture.png"
C = {"vlm": "#a5d8ff", "fuse": "#b2f2bb", "sam": "#ffec99", "mem": "#ffd8a8", "in": "#f1f3f5"}


def box(ax, cx, cy, w, h, text, fc, fs=13, ec="#1e1e1e", lw=2):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.05,rounding_size=0.25",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4)


def zone(ax, x0, y0, x1, y1, title, color):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.05,rounding_size=0.4",
                                fc="none", ec=color, lw=2.2, ls="--", zorder=1))
    ax.text((x0 + x1) / 2, y1 + 0.35, title, ha="center", fontsize=14, fontweight="bold", color=color)


def arrow(ax, p, q, label=None, off=(0, 0.3), rad=0.0, fs=11):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=22, lw=2, color="#1e1e1e",
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], label, ha="center", fontsize=fs,
                color="#c92a2a", zorder=5, bbox=dict(fc="white", ec="none", pad=1.5))


with plt.xkcd():
    fig, ax = plt.subplots(figsize=(17, 9.5))
    ax.set_xlim(0, 20); ax.set_ylim(0, 11.2); ax.axis("off")

    zone(ax, 0.3, 0.6, 7.2, 10.2, "Qwen2.5-VL-3B  (frozen, except one embedding row)", "#1971c2")
    zone(ax, 7.8, 0.6, 13.6, 10.2, "RAU fusion  (trainable)", "#2f9e44")
    zone(ax, 14.2, 0.6, 19.8, 10.2, "SAM2  (encoders frozen)", "#e67700")

    # Qwen
    box(ax, 1.9, 8.8, 2.8, 1.0, "reference image", C["in"])
    box(ax, 1.9, 6.9, 2.8, 1.0, "target image", C["in"])
    box(ax, 1.9, 4.8, 2.8, 1.5, "prompt:\n'find the {label}\nin image 2'", C["in"], fs=12)
    box(ax, 5.4, 6.8, 2.4, 4.2, "Qwen2.5-VL\nforward pass", C["vlm"], fs=14)
    for y in (8.8, 6.9, 4.8):
        arrow(ax, (3.3, y), (4.2, 6.8 + (y - 6.8) * 0.35))
    box(ax, 5.4, 2.2, 3.0, 1.3, "h_seg\n(hidden state at <SEG>)", C["vlm"], fs=12)
    arrow(ax, (5.4, 4.7), (5.4, 2.85), "teacher-forced\n'... <SEG>'", off=(1.15, 0), fs=10)

    # Fusion
    box(ax, 10.7, 2.2, 3.6, 1.3, "SegQueryProjection\nMLP: vlm_dim → 256", C["fuse"], fs=12)
    box(ax, 10.7, 5.4, 3.6, 1.4, "dot-product\nattention (Eq. 7)", C["fuse"], fs=13)
    box(ax, 10.7, 8.6, 4.4, 1.9, "Reference memory\nmasked-avg-pooled SAM2\nfeatures per label\n(built offline)", C["mem"], fs=12)
    arrow(ax, (6.95, 2.2), (8.9, 2.2))
    arrow(ax, (10.7, 2.85), (10.7, 4.7), "q", off=(0.35, 0))
    arrow(ax, (10.7, 7.65), (10.7, 6.1), "memory slots m_j", off=(1.5, 0))

    # SAM2
    box(ax, 17, 9.0, 3.6, 0.9, "target image", C["in"])
    box(ax, 17, 7.5, 3.6, 0.9, "vision_encoder", C["sam"])
    box(ax, 17, 6.0, 3.6, 0.9, "image_embeddings", C["sam"])
    box(ax, 17, 3.9, 3.8, 1.5, "mask_decoder\n(Sam2TwoWayTransformer)\ntrainable", C["sam"], fs=12)
    box(ax, 17, 1.6, 3.6, 0.9, "pred_masks", C["in"])
    arrow(ax, (17, 8.55), (17, 7.95)); arrow(ax, (17, 7.05), (17, 6.45))
    arrow(ax, (17, 5.55), (17, 4.65)); arrow(ax, (17, 3.15), (17, 2.05))
    arrow(ax, (12.5, 5.4), (15.1, 4.2), "z → target_embedding\n(public Sam2Model arg,\nused by PerSAM)", off=(-0.3, 1.05), rad=-0.1, fs=10)

    fig.tight_layout()
    fig.savefig(OUT, dpi=110, facecolor="white")
