"""Context-engineering post: the hand-drawn desk-and-library diagram (plt.xkcd, like the rest of the series)."""
import matplotlib.pyplot as plt

from generate_arc2_diagrams import OUT_DIR, RED, arrow, box


def desk_and_library():
    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(14, 7.5))
        ax.set_xlim(0, 16); ax.set_ylim(-1.2, 9); ax.axis("off")
        box(ax, 3.0, 5.2, 4.6, 3.6, "THE DESK\n(the context window)\n\nsmall: 65,536 tokens\neverything the model\ncan see right now", "#fff3bf")
        box(ax, 13.0, 5.2, 4.6, 3.6, "THE LIBRARY\n(the archive)\n\n4,000-row CSV\n200 reports, 2 policies\nfar bigger than the desk", "#a5d8ff")
        box(ax, 8.0, 7.9, 4.2, 1.0, "the agent's hands: tools", "#d0bfff", fs=12)
        # the three failures and their fixes, one row each
        rows = [(2.35, "1. SEARCHES BADLY", "grep / glob by text, semantic search by meaning", "#ffc9c9"),
                (1.25, "2. TOO MUCH AT ONCE", "page 100 lines, offload big results to a file, pass paths", "#ffd8a8"),
                (0.15, "3. WRONG OR STALE", "a sub-agent reads each file, returns only facts + source", "#b2f2bb")]
        for y, head, fix, col in rows:
            box(ax, 8.0, y + 0.1, 12.6, 0.8, f"{head}:  {fix}", col, fs=11)
        arrow(ax, (4.2, 7.1), (5.9, 7.7), "the model asks", off=(-0.2, 0.55), rad=-0.2)
        arrow(ax, (10.1, 7.7), (11.8, 7.1), "exactly what\nis needed", off=(0.5, 0.75), rad=-0.2)
        arrow(ax, (10.7, 4.4), (5.3, 4.4), "only small, relevant pieces come back", off=(0, -0.55), dashed=True)
        fig.text(0.5, 0.96, "Retrieval: carrying the right pages from the library to a small desk", ha="center",
                 fontsize=16, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/ctx-desk-and-library.png", dpi=110, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    desk_and_library()
