"""Part 9: the hand-drawn pause-and-ask diagram (plt.xkcd, like the rest of the series)."""
import matplotlib.pyplot as plt

from generate_arc2_diagrams import OUT_DIR, arrow, box


def pause_and_ask():
    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(14, 7.6))
        ax.set_xlim(0, 16); ax.set_ylim(-0.6, 9); ax.axis("off")
        # the machine does the work
        box(ax, 2.0, 6.6, 3.2, 1.5, "detect +\nmeasure +\nclassify", "#a5d8ff")
        box(ax, 6.0, 6.6, 3.0, 1.5, "rules +\ndecide", "#a5d8ff")
        arrow(ax, (3.7, 6.6), (4.4, 6.6))
        # three hand-off words
        box(ax, 11.4, 8.1, 3.8, 1.0, "pass: nothing kept\n(straight on; 1 in 10 audited)", "#b2f2bb", fs=11)
        box(ax, 11.4, 6.6, 3.4, 0.9, "review: confident defects", "#ffd8a8", fs=11)
        box(ax, 11.4, 5.1, 3.4, 0.9, "escalate: pipeline unsure", "#ffc9c9", fs=11)
        arrow(ax, (7.6, 6.9), (9.6, 8.0), rad=0.1)
        arrow(ax, (7.6, 6.6), (9.6, 6.6))
        arrow(ax, (7.6, 6.3), (9.6, 5.2), rad=-0.1)
        # the pause
        box(ax, 11.4, 3.1, 4.6, 1.6, "interrupt(payload)\nthe graph STOPS here\nstate saved in SQLite", "#fff3bf")
        arrow(ax, (11.4, 4.6), (11.4, 3.95), "ticket number\n(thread_id)", off=(1.5, 0.0))
        arrow(ax, (13.2, 6.3), (13.6, 3.95), rad=-0.35)
        box(ax, 4.6, 3.1, 4.6, 1.6, "a person looks:\napprove / reject / edit\n(hours or days later)", "#d0bfff")
        arrow(ax, (9.0, 3.1), (6.95, 3.1), "payload out", off=(0, 0.45))
        arrow(ax, (6.95, 2.7), (9.0, 2.7), "Command(resume=answer)", off=(0, -0.85), dashed=True)
        # bad answer loop
        arrow(ax, (3.4, 2.3), (3.4, 0.9), "a bad answer is\nrefused, asked again", off=(-1.6, 0.0), dashed=True)
        box(ax, 8.0, 0.4, 9.4, 1.0, "audit trail: who decided, what they saw, every refused answer", "#e9ecef", fs=11)
        arrow(ax, (11.4, 2.3), (11.0, 0.95), rad=-0.15)
        fig.text(0.5, 0.96, "A person in the loop: the graph pauses, saves its place, and waits for an answer", ha="center", fontsize=15, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/hitl-pause-and-ask.png", dpi=110, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    pause_and_ask()
