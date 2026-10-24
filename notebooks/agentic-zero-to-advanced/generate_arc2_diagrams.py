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


def graph_map():
    """Part 6: the Part 5 loop as a graph. Three nodes, five edges, one shared state."""
    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(14, 7.5))
        ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
        box(ax, 1.4, 4.6, 1.8, 1.0, "START", "#e9ecef", fs=12)
        box(ax, 5.2, 4.6, 3.0, 1.8, "agent\nask the model", "#a5d8ff")
        box(ax, 11.0, 6.9, 3.2, 1.6, "tools\nrun what it asked", "#b2f2bb")
        box(ax, 11.0, 2.3, 3.2, 1.6, "nudge\n\"use a tool, or submit\"", "#ffd8a8")
        box(ax, 14.9, 4.6, 1.6, 1.0, "END", "#e9ecef", fs=12)
        arrow(ax, (2.3, 4.6), (3.7, 4.6))
        arrow(ax, (6.85, 5.7), (9.35, 7.2), "asked for a tool", off=(-0.5, 0.75))
        arrow(ax, (9.35, 6.4), (6.85, 5.0), "result goes back", off=(0.9, -0.55))
        arrow(ax, (6.85, 3.5), (9.35, 2.0), "just talked", off=(-0.4, -0.75))
        arrow(ax, (9.35, 2.7), (6.85, 4.2), "then asks again", off=(1.0, 0.5))
        arrow(ax, (12.6, 6.5), (14.4, 5.2), "report\nsubmitted", off=(0.9, 0.55))
        arrow(ax, (12.6, 2.5), (14.4, 4.0), "step\nlimit", off=(0.95, -0.35), dashed=True)
        ax.text(8.2, 0.7, "every node reads and writes the\nSAME shared state", ha="center", fontsize=11, color="#1971c2")
        fig.text(0.5, 0.96, "Part 5's loop, drawn as a graph", ha="center", fontsize=16, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/graph-map-diagram.png", dpi=110, facecolor="white")
        plt.close(fig)


def state_board():
    """Part 6: the state is a whiteboard; a node returns only the pieces it changes."""
    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(14, 7.0))
        ax.set_xlim(0, 16); ax.set_ylim(0, 8.5); ax.axis("off")
        keys = [("messages", "everything said and seen", "#fff3bf"), ("trace", "what happened, per step", "#fff3bf"),
                ("step", "how many times we asked", "#e9ecef"), ("tokens", "running cost", "#e9ecef"), ("report", "None until submit_report", "#e9ecef")]
        ax.add_patch(FancyBboxPatch((0.4, 0.9), 7.0, 6.6, boxstyle="round,pad=0.05,rounding_size=0.25", fc="white", ec="#1e1e1e", lw=3, zorder=1))
        ax.text(3.9, 7.1, "the state (shared whiteboard)", ha="center", fontsize=14, fontweight="bold")
        for i, (k, d, c) in enumerate(keys):
            box(ax, 3.9, 6.0 - i * 1.15, 6.2, 0.85, f"{k}: {d}", c, fs=11)
        box(ax, 12.2, 5.6, 5.0, 1.8, "a node runs\nreads the board", "#a5d8ff")
        box(ax, 12.2, 2.2, 5.0, 1.8, "returns ONLY what changed\n{messages: [one new], step: 2}", "#b2f2bb", fs=11)
        arrow(ax, (7.5, 5.6), (9.7, 5.6), "read", off=(0, 0.35))
        arrow(ax, (12.2, 4.7), (12.2, 3.2), None)
        arrow(ax, (9.7, 2.2), (7.5, 2.2), "merge", off=(0, 0.35))
        ax.text(12.2, 0.7, "messages and trace are APPENDED (reducer = operator.add)\nstep, tokens, report are REPLACED", ha="center", fontsize=11, color="#c92a2a")
        fig.text(0.5, 0.96, "State: nodes don't pass values around, they write on one board", ha="center", fontsize=15, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/graph-state-board.png", dpi=110, facecolor="white")
        plt.close(fig)


def pipeline_map():
    """Part 7: the inspection pipeline. A straight line with two exits; the model is used in two boxes only."""
    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(15, 7.0))
        ax.set_xlim(0, 18); ax.set_ylim(0, 8.5); ax.axis("off")
        det, llm = "#b2f2bb", "#a5d8ff"
        xs = [1.6, 4.5, 7.4, 10.3, 13.2, 16.1]
        nodes = [(xs[0], "detect\nfind_candidates", det), (xs[1], "measure\nmeasure_candidate", det),
                 (xs[2], "classify\nLLM, one batched call", llm), (xs[3], "rules\nborder, min area,\nseverity", det),
                 (xs[4], "decide\npass / review /\nescalate", det), (xs[5], "report\nLLM writes prose", llm)]
        for x, t, c in nodes:
            box(ax, x, 4.4, 2.3, 1.9, t, c, fs=11)
        for x1, x2 in zip(xs, xs[1:]):
            arrow(ax, (x1 + 1.2, 4.4), (x2 - 1.2, 4.4))
        ax.annotate("", xy=(xs[5], 5.5), xytext=(xs[0], 5.5), arrowprops=dict(arrowstyle="-|>", lw=2, color=RED, ls="--",
                    connectionstyle="arc3,rad=-0.22", mutation_scale=22))
        ax.text(8.85, 7.3, "no candidates: skip straight to report, no model call", ha="center", fontsize=11, color=RED)
        ax.annotate("", xy=(xs[4], 3.3), xytext=(xs[2], 3.3), arrowprops=dict(arrowstyle="-|>", lw=2, color=RED, ls="--",
                    connectionstyle="arc3,rad=0.35", mutation_scale=22))
        ax.text(9.6, 1.5, "classifier output invalid after a retry:\nskip the rules, escalate to a human", ha="center", fontsize=11, color=RED)
        ax.text(0.4, 0.45, "green = plain Python        blue = a model call", fontsize=11, color="#1971c2", fontweight="bold")
        fig.text(0.5, 0.96, "The pipeline: code owns the order, the model owns two judgments", ha="center", fontsize=16, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/pipeline-map-diagram.png", dpi=110, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    agent_loop()
    graph_map()
    state_board()
    pipeline_map()
