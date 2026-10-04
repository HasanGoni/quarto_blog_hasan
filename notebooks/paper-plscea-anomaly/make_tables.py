"""Markdown tables for the post, straight from the saved results (no hand-copied numbers)."""
import json, tempfile
import numpy as np

def macro(res, v):
    cl = sorted(res[v]); return np.array([[res[v][c][s] for c in cl] for s in range(len(res[v][cl[0]]))]).mean(1)

def table(path, variants, names, ap=True):
    r = json.load(open(path)); cl = sorted(r["auroc"]["qk"])
    hdr = "| Category | " + " | ".join(names) + " | Δ (PL-SCEA γ=2 − QK) |"
    out = [hdr, "|---|" + "---|" * (len(names) + 1)]
    for c in cl:
        vals = [np.mean(r["auroc"][v][c]) for v in variants]
        d = vals[1] - vals[0]
        out.append(f"| {c} | " + " | ".join(f"{100*x:.1f}" for x in vals) + f" | {100*d:+.1f} |")
    row = ["**macro mean ± std over seeds**"]
    for v in variants:
        m = 100 * macro(r["auroc"], v); row.append(f"**{m.mean():.2f} ± {m.std(ddof=1):.2f}**")
    row.append(f"**{100*(macro(r['auroc'], variants[1]) - macro(r['auroc'], variants[0])).mean():+.2f}**")
    out.append("| " + " | ".join(row) + " |")
    if ap and "aupr" in r:
        row = ["AUPR macro mean"] + [f"{100*macro(r['aupr'], v).mean():.2f}" for v in variants] + [f"{100*(macro(r['aupr'], variants[1]) - macro(r['aupr'], variants[0])).mean():+.2f}"]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)

names = ["QK (original)", "PL-SCEA γ=2", "PL-SCEA γ=4"]
print("### SETUP B 8-shot\n" + table("out/setup_official/results_8shot.json", ["qk", "g2", "g4"], names))
rA = json.load(open("out/setupA_test_holdout/results_8shot.json"))
# setup A stored {"auroc": {variant: {cls: [...]}}} with variants qk/plscea_g2/plscea_g4 (no AUPR)
tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump({"auroc": {"qk": rA["auroc"]["qk"], "g2": rA["auroc"]["plscea_g2"], "g4": rA["auroc"]["plscea_g4"]}}, tmp); tmp.close()
print("\n### SETUP A 8-shot\n" + table(tmp.name, ["qk", "g2", "g4"], names, ap=False))


# ---- pixel-level macro table
rp = json.load(open("out/setup_official/results_pixel.json")); px = rp["pixel"]; cl = rp["classes"]
def mac(K, v, m):
    a = 100 * np.array([[px[K][v][m][c][s] for c in cl] for s in range(3)]).mean(1)
    return a
print("\n### PIXEL macro (setup B)\n| Shots | Metric | QK (original) | PL-SCEA γ=2 | Δ (mean of per-seed differences) |\n|---|---|---|---|---|")
for K in ("1", "8"):
    for m, name in (("p_auroc", "pixel AUROC"), ("p_f1max", "pixel F1-max"), ("pro", "PRO")):
        q, g = mac(K, "qk", m), mac(K, "g2", m)
        print(f"| {K}-shot | {name} | {q.mean():.2f} ± {q.std(ddof=1):.2f} | {g.mean():.2f} ± {g.std(ddof=1):.2f} | {(g-q).mean():+.2f} |")
