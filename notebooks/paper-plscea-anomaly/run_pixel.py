"""Pixel-level evaluation of PL-SCEA vs the original QK attention (setup B: official protocol).

Same frozen C-RADIOv3-B, same VAE head, same shots/seeds as run_eval.py; the anomaly map
(standardized reconstruction error, upsampled to 448 px, Gaussian sigma=4) is compared with the official
MVTec-AD ground-truth masks. Metrics per category over all test images: pixel AUROC, pixel F1-max, PRO.
Macro-averaged over the 15 categories; mean +- std over 3 support-set seeds.

usage: python run_pixel.py [K ...]    (default: 1 8)
"""
import json, os, sys, time
import numpy as np, torch
from PIL import Image

import data
from common import Radio, fit_lnm, patch_score_map, IMG_SIZE
from pixel_metrics import pixel_metrics

KS = [int(k) for k in sys.argv[1:]] or [1, 8]
SEEDS = [0, 1, 2]
VARIANTS = {"qk": ("qk", 2.0), "g2": ("plscea", 2.0)}
OUT = "out/setup_official"; PARTIAL = f"{OUT}/pixel_partial.json"
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_mask(path):
    parts = path.split("/")                      # data/mvtec_ad/<cls>/test/<defect>/<name>.png
    cls, defect, name = parts[-4], parts[-2], parts[-1][:-4]
    if defect == "good":
        return np.zeros((IMG_SIZE, IMG_SIZE), bool)
    m = Image.open(f"data/mvtec_ad/{cls}/ground_truth/{defect}/{name}_mask.png").convert("L")
    return np.asarray(m.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 0


if __name__ == "__main__":
    classes = data.official_classes(); radio = Radio(device)
    res = {}  # res[str(K)][variant][metric][cls] = [per seed]
    if os.path.exists(PARTIAL):
        res = json.load(open(PARTIAL)); print("resuming; done:", {k: sorted(v["qk"]["p_auroc"]) for k, v in res.items()}, flush=True)
    for K in KS:
        res.setdefault(str(K), {v: {m: {} for m in ("p_auroc", "p_f1max", "pro")} for v in VARIANTS})
    for cls in classes:
        todo = [K for K in KS if cls not in res[str(K)]["qk"]["p_auroc"]]
        if not todo:
            continue
        t0 = time.time(); d = data.official_class(cls)
        test = [k for k in d["cands"]]
        masks = [load_mask(k) for k in test]
        feats = {v: {} for v in VARIANTS}
        def F(v, k):
            if k not in feats[v]:
                feats[v][k] = radio.patch_tokens(d["load"](k), *VARIANTS[v])
            return feats[v][k]
        for K in todo:
            for seed in SEEDS:
                order = list(np.random.default_rng(seed).permutation(d["pool"]))
                support = order[:K]
                for v in VARIANTS:
                    m, mean, std = fit_lnm(torch.cat([F(v, k) for k in support]), seed=seed)
                    maps = [patch_score_map(m, mean, std, F(v, k), "paper") for k in test]
                    pm = pixel_metrics(maps, masks)
                    for name, val in pm.items():
                        res[str(K)][v][name].setdefault(cls, []).append(val)
            line = "  ".join(f"{v} AUROC {100*np.mean(res[str(K)][v]['p_auroc'][cls]):.1f} F1 {100*np.mean(res[str(K)][v]['p_f1max'][cls]):.1f} PRO {100*np.mean(res[str(K)][v]['pro'][cls]):.1f}" for v in VARIANTS)
            print(f"{cls:12s} K={K}  {line}", flush=True)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(PARTIAL, "w"))
    json.dump({"seeds": SEEDS, "classes": classes, "pixel": res}, open(f"{OUT}/results_pixel.json", "w"), indent=2)
    for K in KS:
        print(f"\nPixel-level, K={K}: macro over {len(classes)} categories, mean +- std over seeds")
        for v in VARIANTS:
            for name in ("p_auroc", "p_f1max", "pro"):
                ps = np.array([[res[str(K)][v][name][c][s] for c in classes] for s in range(len(SEEDS))]).mean(1)
                print(f"  {v:3s} {name:8s} {100*ps.mean():.2f} +- {100*ps.std(ddof=1):.2f}")
