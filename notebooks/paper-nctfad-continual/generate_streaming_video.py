"""Real streaming visualization: replay the actual measured forgetting curve (already computed
by run_eval.py, cached in out/results.json) one category-arrival at a time, so watching the video
matches exactly watching the real evaluation stream new categories in with no announced task
boundary.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import imageio

IMG_OUT = "../../posts/series/papers/images"
CLASSES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
           "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]

with open("out/results.json") as f:
    results = json.load(f)

nctfad = results["nctfad"]
baseline = results["baseline"]

frames = []
for k in range(1, len(CLASSES) + 1):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x_n = [c[0] for c in nctfad[:k]]
    y_n = [c[1] for c in nctfad[:k]]
    x_b = [c[0] for c in baseline[:k]]
    y_b = [c[1] for c in baseline[:k]]
    ax.plot(x_n, y_n, marker="o", color="#1f77b4", label="NC-TFAD (fixed ETF prototypes)")
    ax.plot(x_b, y_b, marker="s", color="#ff7f0e", label="Baseline (drifting mean prototypes)")
    ax.set_xlim(0.5, len(CLASSES) + 0.5)
    ax.set_ylim(0.55, 1.02)
    ax.set_xlabel("categories streamed so far (no task boundary announced)")
    ax.set_ylabel("mean AUROC over all categories seen so far")
    ax.set_title(f"Real streaming eval -- category {k}/15 just arrived: '{CLASSES[k-1]}'")
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    fig.canvas.draw()
    from PIL import Image
    frame_img = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                                  fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")
    frames.append(np.array(frame_img))
    plt.close(fig)
    if k == len(CLASSES):  # hold the final frame a bit longer
        frames.extend([frames[-1]] * 6)

imageio.mimsave(f"{IMG_OUT}/nctfad-streaming-video.mp4", frames, fps=2, macro_block_size=16)
print("Saved nctfad-streaming-video.mp4")
