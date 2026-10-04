"""Pixel-level anomaly-localization metrics, computed per category over all test images
(good + defective): pixel AUROC (exact), pixel F1-max and PRO (both from fine threshold histograms).

PRO = per-region overlap (Bergmann et al.): for each threshold, the mean over ground-truth connected
components of the fraction of the component that is predicted anomalous; integrated against the
false-positive rate on normal pixels up to FPR 0.3 and normalized by 0.3.
"""
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score

N_BINS = 1000
FPR_LIMIT = 0.3
EIGHT = np.ones((3, 3), dtype=int)


def pixel_metrics(score_maps, masks):
    """score_maps: list of (H, W) float arrays; masks: list of (H, W) bool arrays (same shapes)."""
    scores = np.concatenate([s.ravel() for s in score_maps]).astype(np.float32)
    gt = np.concatenate([m.ravel() for m in masks])
    auroc = roc_auc_score(gt, scores)

    lo, hi = float(scores.min()), float(scores.max())
    edges = np.linspace(lo, hi, N_BINS + 1)
    pos_h, _ = np.histogram(scores[gt], bins=edges)
    neg_h, _ = np.histogram(scores[~gt], bins=edges)
    # threshold i = edges[i]: predicted positive = score >= edges[i]
    tp = np.cumsum(pos_h[::-1])[::-1].astype(np.float64); fp = np.cumsum(neg_h[::-1])[::-1].astype(np.float64)
    fn = pos_h.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    f1_max = float(f1.max())

    # PRO: per-component histograms
    comp_cum = []
    for s, m in zip(score_maps, masks):
        if not m.any():
            continue
        lab, n = label(m, structure=EIGHT)
        for c in range(1, n + 1):
            h, _ = np.histogram(s[lab == c], bins=edges)
            comp_cum.append(np.cumsum(h[::-1])[::-1] / h.sum())
    if comp_cum:
        pro_curve = np.mean(comp_cum, axis=0)               # per threshold
        fpr = fp / max(neg_h.sum(), 1)
        order = np.argsort(fpr)                             # ascending FPR
        f, p = fpr[order], pro_curve[order]
        keep = f <= FPR_LIMIT
        f, p = np.append(f[keep], FPR_LIMIT), np.append(p[keep], np.interp(FPR_LIMIT, fpr[order], pro_curve[order]))
        pro = float(np.trapezoid(p, f) / FPR_LIMIT)
    else:
        pro = float("nan")
    return {"p_auroc": float(auroc), "p_f1max": f1_max, "pro": pro}
