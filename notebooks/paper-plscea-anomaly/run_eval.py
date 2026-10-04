"""Real evaluation of PL-SCEA (arXiv:2609.03655), corrected reimplementation.

Frozen C-RADIOv3-B; the ONLY thing that changes between variants is the final block's attention:
  qk      : original QK attention (the paper's baseline)
  g<gam>  : PL-SCEA with power exponent gamma
Same LNM-VAE head, same k-shot support set, same seeds for every variant. Metrics: image-level AUROC
and AUPR (anomaly = positive class), macro-averaged over the 15 categories; mean +- std over 3
support-set seeds.

usage: python run_eval.py PROTOCOL K GAMMAS      e.g.  python run_eval.py official 8 2,4
  PROTOCOL: holdout (setup A: test-split holdout, grid smoothing) | official (setup B: official
            train/good support + full official test set, sigma=4 upsampled smoothing)
"""
import json, os, sys, time
import numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score

import data
from common import Radio, fit_lnm, patch_score_map, image_score

PROTOCOL, K_SHOT = sys.argv[1], int(sys.argv[2])
GAMMAS = [float(g) for g in sys.argv[3].split(",")]
SEEDS = [0, 1, 2]
SMOOTH = "paper" if PROTOCOL == "official" else "grid"
OUT = f"out/setup_{PROTOCOL}"; os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
VARIANTS = {"qk": ("qk", 2.0), **{f"g{g:g}": ("plscea", g) for g in GAMMAS}}

if __name__ == "__main__":
    if PROTOCOL == "holdout":
        ds = data.load_mvtec_ds(); classes = data.holdout_classes(ds); get = lambda c: data.holdout_class(ds, c)
    else:
        classes = data.official_classes(); get = data.official_class
    radio = Radio(device)
    print(f"{PROTOCOL}: {len(classes)} categories, {K_SHOT}-shot, seeds {SEEDS}, variants {list(VARIANTS)}", flush=True)
    auroc = {v: {} for v in VARIANTS}; aupr = {v: {} for v in VARIANTS}
    PARTIAL = f"{OUT}/partial_{K_SHOT}shot.json"  # per-category checkpoint: a killed run resumes here
    if os.path.exists(PARTIAL):
        part = json.load(open(PARTIAL)); auroc, aupr = part["auroc"], part["aupr"]
        print("resuming, already done:", sorted(auroc["qk"]), flush=True)
    for cls in classes:
        if cls in auroc["qk"]:
            continue
        t0 = time.time(); d = get(cls); cache = {v: {} for v in VARIANTS}
        def feats(v, k):
            if k not in cache[v]:
                cache[v][k] = radio.patch_tokens(d["load"](k), *VARIANTS[v])
            return cache[v][k]
        for seed in SEEDS:
            order = list(np.random.default_rng(seed).permutation(d["pool"]))
            support = [order[i] if PROTOCOL == "official" else int(order[i]) for i in range(K_SHOT)]
            ev = [k for k in d["cands"] if k not in set(support)]
            labels = [d["label"][k] for k in ev]
            for v in VARIANTS:
                m, mean, std = fit_lnm(torch.cat([feats(v, k) for k in support]), seed=seed)
                sc = [image_score(patch_score_map(m, mean, std, feats(v, k), SMOOTH)) for k in ev]
                auroc[v].setdefault(cls, []).append(roc_auc_score(labels, sc))
                aupr[v].setdefault(cls, []).append(average_precision_score(labels, sc))
        line = "  ".join(f"{v} {np.mean(auroc[v][cls]):.3f}" for v in VARIANTS)
        print(f"{cls:12s} n_eval={len(ev):3d}  {line}   ({time.time()-t0:.0f}s)", flush=True)
        cache.clear()
        json.dump({"auroc": auroc, "aupr": aupr}, open(PARTIAL, "w"))
    json.dump({"protocol": PROTOCOL, "k_shot": K_SHOT, "seeds": SEEDS, "gammas": GAMMAS, "classes": classes,
               "auroc": auroc, "aupr": aupr}, open(f"{OUT}/results_{K_SHOT}shot.json", "w"), indent=2)
    print(f"\nMacro-mean over {len(classes)} categories ({PROTOCOL}, {K_SHOT}-shot), mean +- std over seeds:")
    for name, res in [("AUROC", auroc), ("AUPR", aupr)]:
        for v in VARIANTS:
            ps = np.array([[res[v][c][s] for c in classes] for s in range(len(SEEDS))]).mean(1)
            print(f"  {name} {v:6s} {100*ps.mean():.2f} +- {100*ps.std(ddof=1):.2f}")
