"""Part 8: how sure is the box-call ranking? Paired bootstrap over held-out IMAGES (boxes on one image are not independent).
Reads the saved probabilities of the guided runs. Prints AUC with a 95% interval and each judge's paired difference from the rule."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "out" / "typed"
rows = [json.loads(l) for l in (OUT / "candidates.jsonl").read_text().splitlines()]
test_idx = [i for i, r in enumerate(rows) if r["split"] == "test"]
y = np.array([rows[i]["y"] for i in test_idx])
img = np.array([rows[i]["image"] for i in test_idx])
JUDGES = {"rule (contrast >= 8)": ("band", "guided"), "laya-english guided": ("laya-english", "guided"),
          "llm-verbalized guided": ("llm-verbalized", "guided"), "llm-logprob guided": ("llm-logprob", "guided"),
          "llm-logprob plain": ("llm-logprob", "plain")}
P = {k: np.array([json.load(open(OUT / f"results_{j}.json"))["A"]["p"][t][i] for i in test_idx]) for k, (j, t) in JUDGES.items()}


def auc(p, yy):
    pos, neg = p[yy == 1], p[yy == 0]
    if not len(pos) or not len(neg):
        return np.nan
    return ((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


imgs = sorted(set(img))
by_img = {m: np.where(img == m)[0] for m in imgs}
rng = np.random.default_rng(0)
B = 1000
boot = {k: [] for k in P}
for _ in range(B):
    idx = np.concatenate([by_img[m] for m in rng.choice(imgs, len(imgs))])
    for k in P:
        boot[k].append(auc(P[k][idx], y[idx]))
out = {}
base = "rule (contrast >= 8)"
for k in P:
    a = auc(P[k], y)
    lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
    d = np.array(boot[k]) - np.array(boot[base])
    dlo, dhi = np.nanpercentile(d, [2.5, 97.5])
    out[k] = {"auc": round(float(a), 3), "ci": [round(float(lo), 3), round(float(hi), 3)],
              "diff_vs_rule": None if k == base else [round(float(auc(P[k], y) - auc(P[base], y)), 3), round(float(dlo), 3), round(float(dhi), 3)]}
    print(f"{k:24s} AUC {a:.3f} [{lo:.3f}, {hi:.3f}]" + ("" if k == base else f" | minus rule {out[k]['diff_vs_rule'][0]:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"))
(OUT / "bootstrap.json").write_text(json.dumps(out, indent=1))
