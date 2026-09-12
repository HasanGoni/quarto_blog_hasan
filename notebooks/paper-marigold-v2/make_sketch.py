#!/usr/bin/env python3
"""Hand-drawn/xkcd-style sketch of the Marigold V2 single-step pipeline, for
the ELI5 section -- matches the established style for this post type (see
CLAUDE.md: 'an optional hand-drawn/xkcd-style sketch diagram for the ELI5
section')."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch

OUT = "images/marigold-v2-eli5-sketch.png"

with plt.xkcd():
    fig, ax = plt.subplots(figsize=(11.6, 3.6))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    boxes = [
        (0.3, "Your\nX-ray\nphoto"),
        (2.5, "Squish into\nQwen's VAE\n'memory space'"),
        (5.0, "ONE guess\nfrom the big\nedit-brain\n(t = 0.499)"),
        (7.5, "Unsquish\nback into\na picture"),
        (9.7, "Depth\nmap!"),
    ]
    for x, label in boxes:
        box = FancyBboxPatch(
            (x, 1.1), 1.6, 1.6, boxstyle="round,pad=0.08,rounding_size=0.15",
            linewidth=2, edgecolor="black", facecolor="#fff7e0",
        )
        ax.add_patch(box)
        ax.text(x + 0.8, 1.9, label, ha="center", va="center", fontsize=10)

    xs = [b[0] for b in boxes]
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.add_patch(
            FancyArrow(
                x0 + 1.65, 1.9, (x1 - x0) - 1.75, 0, width=0.03,
                head_width=0.22, head_length=0.15, length_includes_head=True,
                color="black",
            )
        )

    ax.text(
        5.0, 3.3,
        "\"I already know what a huge pile of edited photos looks like --\n"
        "let me guess the depth in ONE shot instead of denoising 50 times.\"",
        ha="center", va="top", fontsize=10, style="italic",
    )
    ax.text(
        5.0, 0.55,
        "No iterative denoising. One forward pass. That's the whole trick.",
        ha="center", va="top", fontsize=10.5, fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(OUT, dpi=150)
    print("saved", OUT)
