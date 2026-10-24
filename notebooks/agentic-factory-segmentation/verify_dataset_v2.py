"""Checks for void_dataset_v2.  python verify_dataset_v2.py"""
import csv, json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from void_dataset_v2 import SEED, image_rng, load_fmnist, make_sample

root = Path(__file__).parent
d = root / "data" / "void_dataset_v2"
X, y = load_fmnist(root / "data" / "fashion_mnist")
S = {s: dict(np.load(d / f"{s}.npz")) for s in ("train", "val", "test")}   # arrays loaded once (a lazy npz re-reads on every access)
meta = list(csv.DictReader(open(d / "metadata.csv")))
defects = [json.loads(l) for l in (d / "defects.jsonl").read_text().splitlines()]

print("1. containment: defect pixels outside the part, and within 8 px of the part's edge")
out = near = 0
for s, a in S.items():
    m, p = a["defect_masks"] > 0, a["part_masks"] > 0
    out += int((m & ~p).sum())
    near += int((m & ~np.stack([ndi.binary_erosion(q, iterations=8) for q in p])).sum())
print(f"   outside the part: {out}   within 8 px of the edge: {near}  {'PASS' if out == near == 0 else 'FAIL'}")

print("2. the mask matches the metadata: components per image == n_voids + n_cracks, and component type matches")
bad = 0
for r in meta:
    sp, i = r["split"], int(r["id"].split("_")[1]); m = S[sp]["defect_masks"][i]
    nv, nc = ndi.label(m == 1)[1], ndi.label(m == 2)[1]
    bad += (nv, nc) != (int(r["n_voids"]), int(r["n_cracks"]))
print(f"   images where the mask disagrees with the metadata: {bad}  {'PASS' if bad == 0 else 'FAIL'}")

print("3. splits share no source shape")
src = {s: {int(r["src_index"]) for r in meta if r["split"] == s} for s in S}
sh = len(src["train"] & src["val"]) + len(src["train"] & src["test"]) + len(src["val"] & src["test"])
print(f"   source items shared between splits: {sh}  {'PASS' if sh == 0 else 'FAIL'}")

print("4. visibility: every labelled defect scored >= 8 when it was drawn")
sc = np.array([x["score"] for x in defects])
print(f"   defects {len(sc)}: min score {sc.min():.2f}, p5 {np.percentile(sc, 5):.1f}, median {np.median(sc):.1f}  {'PASS' if sc.min() >= 8 else 'FAIL'}")

print("5. deterministic: regenerate the first 30 train images and compare byte for byte")
ok = 0
for r in [m for m in meta if m["split"] == "train"][:30]:
    i = int(r["id"].split("_")[1]); s = int(r["src_index"])
    img, mask, part, prm, recs = make_sample(X[s], int(y[s]), image_rng(0, i))
    ok += bool((img == S["train"]["images"][i]).all() and (mask == S["train"]["defect_masks"][i]).all())
print(f"   identical: {ok} of 30  {'PASS' if ok == 30 else 'FAIL'}")

print("6. what is in it")
n = len(meta); nv = sum(x["type"] == "void" for x in defects)
cnt = {k: sum(int(r["n_defects"]) == k for r in meta) for k in range(6)}
print(f"   images {n}: defects per image {cnt}")
print(f"   voids {nv}, cracks {len(defects) - nv}; images with a crack: {sum(int(r['n_cracks']) > 0 for r in meta)}")
a = np.array([x["area_px"] for x in defects if x["type"] == "void"]); b = np.array([x["area_px"] for x in defects if x["type"] == "crack"])
print(f"   void area px p5={np.percentile(a, 5):.0f} p50={np.percentile(a, 50):.0f} p95={np.percentile(a, 95):.0f}; crack area px p5={np.percentile(b, 5):.0f} p50={np.percentile(b, 50):.0f} p95={np.percentile(b, 95):.0f}")
mu = np.array([float(r["mu"]) for r in meta])
print(f"   part opacity mu p5={np.percentile(mu, 5):.2f} p50={np.percentile(mu, 50):.2f} p95={np.percentile(mu, 95):.2f}")
