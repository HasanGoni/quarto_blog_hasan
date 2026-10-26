"""Part 9: draw what the reviewer is shown. One X-ray image, every candidate box numbered, colour = what the pipeline did to it."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from inspection_tools import load_gray

GREEN, RED, GREY = "#2f9e44", "#e03131", "#868e96"


def show_review(payload: dict, truth: dict | None = None, ax=None):
    """truth (id -> bool) is drawn as a thin outline ONLY when you pass it: a real reviewer would not have it. Use it to check yourself."""
    gray = load_gray(payload["image"])
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    for c in payload["candidates"]:
        x1, y1, x2, y2 = c["box"]
        kept = c["pipeline"].startswith("kept")
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=2.2 if kept else 1.2,
                               ec=GREEN if kept else GREY, ls="-" if kept else "--"))
        if truth is not None:
            ax.add_patch(Rectangle((x1 - 3, y1 - 3), x2 - x1 + 6, y2 - y1 + 6, fill=False, lw=1.5, ec="#fcc419" if truth[c["id"]] else "none"))
        ax.text(x1, max(8, y1 - 4), str(c["id"]), color=GREEN if kept else "#dee2e6", fontsize=11, fontweight="bold")
    ax.set_title(f"{payload['image']}: pipeline said '{payload['pipeline_verdict']}'  (solid green = kept, dashed grey = rejected)", fontsize=9)
    ax.axis("off")
    if own:
        plt.show()
