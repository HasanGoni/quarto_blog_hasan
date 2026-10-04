"""Figures from the saved real results.

  plscea-auroc-comparison.png  : per-category 8-shot AUROC, original QK attention vs PL-SCEA (setup B)
  plscea-gamma-sweep.png       : issue-style dual-axis gamma sweep, setup B (official protocol, 1-shot)
  plscea-gamma-sweep-setupA.png: same chart for setup A (test-split holdout, 1-shot)
"""
import json
import numpy as np
import matplotlib.pyplot as plt

IMG_OUT = "../../posts/series/papers/images"
BLUE, ORANGE = "#0072B2", "#D55E00"


def key(res, g):  # setup A wrote "g2.0", setup B writes "g2"
    return next(k for k in (f"g{g:g}", f"g{g}") if k in res)


def macro(res, v):
    """(n_seeds,) macro-average over categories."""
    cl = sorted(res[v])
    return np.array([[res[v][c][s] for c in cl] for s in range(len(res[v][cl[0]]))]).mean(1)


def sweep_plot(path, out, title):
    r = json.load(open(path)); g = r["gammas"]
    au = [100 * macro(r["auroc"], key(r["auroc"], x)).mean() for x in g]
    ap = [100 * macro(r["aupr"], key(r["aupr"], x)).mean() for x in g]
    au_qk, ap_qk = 100 * macro(r["auroc"], "qk").mean(), 100 * macro(r["aupr"], "qk").mean()
    fig, ax1 = plt.subplots(figsize=(8, 5.2)); ax2 = ax1.twinx()
    ax1.axvspan(1.9, 2.1, color="#cfe8f7", alpha=0.9, zorder=0); ax1.axvline(2, color=BLUE, ls="--", lw=1.5, zorder=1)
    ax1.plot(g, au, "-o", color=BLUE, lw=3, ms=9, mfc="white", mew=2.5, label="Classification AUROC", zorder=3)
    ax2.plot(g, ap, "-o", color=ORANGE, lw=3, ms=9, mfc="white", mew=2.5, label="Classification AUPR", zorder=3)
    ax1.axhline(au_qk, color=BLUE, ls=":", lw=1.5); ax2.axhline(ap_qk, color=ORANGE, ls=":", lw=1.5)
    ax1.set_xticks(g); ax1.set_xticklabels([f"{x:g}" for x in g]); ax1.set_xlabel(r"Power exponent $\gamma$", fontsize=13)
    ax1.set_ylabel("Classification AUROC (%)", color=BLUE, fontsize=13)
    ax2.set_ylabel("Classification AUPR (%)", color=ORANGE, fontsize=13)
    ax1.tick_params(axis="y", colors=BLUE); ax2.tick_params(axis="y", colors=ORANGE); ax1.grid(axis="y", alpha=0.3)
    h = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    h += [plt.Line2D([], [], color="gray", ls=":", label="original QK attention (dotted)")]
    ax1.legend(handles=h, loc="lower right", fontsize=9); ax1.set_title(title, fontsize=10)
    plt.tight_layout(); plt.savefig(f"{IMG_OUT}/{out}", dpi=130); plt.close()
    print(out, "AUROC", np.round(au, 2).tolist(), "QK", round(au_qk, 2), "| AUPR", np.round(ap, 2).tolist(), "QK", round(ap_qk, 2))


# per-category bars, setup B 8-shot
r8 = json.load(open("out/setup_official/results_8shot.json"))
classes = sorted(r8["auroc"]["qk"]); x = np.arange(len(classes))
q, p = macro(r8["auroc"], "qk"), macro(r8["auroc"], "g2")
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - 0.2, [np.mean(r8["auroc"]["qk"][c]) for c in classes], 0.4, color=BLUE, label=f"Original QK attention (mean {100*q.mean():.1f})")
ax.bar(x + 0.2, [np.mean(r8["auroc"]["g2"][c]) for c in classes], 0.4, color=ORANGE, label=rf"PL-SCEA, $\gamma$=2 (mean {100*p.mean():.1f})")
ax.set_xticks(x); ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_ylim(0.78, 1.01)
ax.set_ylabel("image AUROC (8-shot, official MVTec-AD protocol, 3 seeds)"); ax.legend(loc="lower left")
ax.set_title("Frozen C-RADIOv3-B: original attention vs. PL-SCEA")
plt.tight_layout(); plt.savefig(f"{IMG_OUT}/plscea-auroc-comparison.png", dpi=130); plt.close()

sweep_plot("out/setup_official/results_1shot.json", "plscea-gamma-sweep.png",
           "Setup B (official train/test split): 1-shot, 15 MVTec-AD categories, 3 seeds")
sweep_plot("out/setupA_test_holdout/sweep_1shot.json", "plscea-gamma-sweep-setupA.png",
           "Setup A (test-split holdout): 1-shot, 15 MVTec-AD categories, 3 seeds")


# ---- pixel-level: per-category change in pixel F1-max (PL-SCEA gamma=2 minus original QK), setup B
rp = json.load(open("out/setup_official/results_pixel.json")); px = rp["pixel"]; cl = rp["classes"]
def dF1(K):
    q = np.array([[px[K]["qk"]["p_f1max"][c][s] for c in cl] for s in range(3)])
    g = np.array([[px[K]["g2"]["p_f1max"][c][s] for c in cl] for s in range(3)])
    return 100 * (g - q).mean(0)
d1, d8 = dF1("1"), dF1("8"); order = np.argsort(d8)[::-1]; x = np.arange(len(cl))
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.bar(x - 0.2, d1[order], 0.4, color="#56B4E9", label=f"1-shot (macro {d1.mean():+.1f})")
ax.bar(x + 0.2, d8[order], 0.4, color=BLUE, label=f"8-shot (macro {d8.mean():+.1f})")
ax.axhline(0, color="black", lw=1)
ax.set_xticks(x); ax.set_xticklabels([cl[i] for i in order], rotation=45, ha="right")
ax.set_ylabel("change in pixel F1-max (points)\nPL-SCEA (γ=2) minus original attention")
ax.set_title("Pixel-level: where PL-SCEA helps and hurts (official protocol, mean of 3 seeds)")
ax.legend(); plt.tight_layout(); plt.savefig(f"{IMG_OUT}/plscea-pixel-f1-delta.png", dpi=130); plt.close()
print("saved plscea-pixel-f1-delta.png")
