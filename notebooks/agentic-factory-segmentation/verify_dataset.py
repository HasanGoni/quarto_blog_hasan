"""Checks on the built dataset. Prints a report; every number in the post comes from here. python verify_dataset.py"""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from void_dataset import N, image_rng, load_fmnist, make_sample

root = Path(__file__).parent
d = root / "data" / "void_dataset_v1"
X, y = load_fmnist(root / "data" / "fashion_mnist")
meta = list(csv.DictReader(open(d / "metadata.csv")))
voids = [json.loads(l) for l in (d / "voids.jsonl").read_text().splitlines()]
sp = {s: {k: v for k, v in np.load(d / f"{s}.npz").items()} for s in ("train", "val", "test")}   # load each array ONCE: indexing an npz file decompresses the whole array every time
rows = {s: [m for m in meta if m["split"] == s] for s in sp}
ok = lambda b: "PASS" if b else "FAIL"

print("1. containment: voids outside the part")
bad = sum(int((sp[s]["void_masks"].astype(bool) & ~sp[s]["part_masks"].astype(bool)).sum()) for s in sp)
print(f"   void pixels outside the part mask: {bad}  {ok(bad == 0)}")

print("2. mask matches the metadata: connected components per image == n_voids; no mask iff no voids")
mism = empty_bad = 0
for s in sp:
    for i, m in enumerate(rows[s]):
        n = ndi.label(sp[s]["void_masks"][i])[1]
        mism += n != int(m["n_voids"]); empty_bad += (sp[s]["void_masks"][i].any()) != (int(m["n_voids"]) > 0)
print(f"   images where the mask component count differs from n_voids: {mism}  {ok(mism == 0)}")
print(f"   images where 'has a mask' disagrees with 'has voids': {empty_bad}  {ok(empty_bad == 0)}")

print("3. splits share no source shape")
src = {s: {int(m["src_index"]) for m in rows[s]} for s in sp}
shared = sum(len(src[a] & src[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
print(f"   source items shared between splits: {shared}  {ok(shared == 0)}")

print("4. deterministic: regenerate the first 30 train images and compare byte for byte")
sid = 0
same = sum(np.array_equal(make_sample(X[int(rows['train'][i]['src_index'])], image_rng(sid, i))[0], sp["train"]["images"][i]) for i in range(30))
print(f"   identical: {same} of 30  {ok(same == 30)}")

print("5. near-duplicate SHAPES (Fashion-MNIST has many look-alike items): nearest-neighbour distance, test source -> train source")
def vec(idx): return X[np.array(sorted(idx))].reshape(len(idx), -1).astype(np.float32) / 255
tr, te = vec(src["train"]), vec(src["test"])
def nn(a, b, same=False):
    D = np.sqrt(np.maximum((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None] - 2 * a @ b.T, 0))
    if same: np.fill_diagonal(D, np.inf)
    return D.min(1)
t2t, w = nn(te, tr), nn(tr[:600], tr, same=True)
q = lambda a: " ".join(f"p{p}={np.percentile(a, p):.2f}" for p in (5, 25, 50, 75))
print(f"   test->train: {q(t2t)}   | train->train (600 items, excluding self): {q(w)}")
thr = float(np.percentile(w, 5))
print(f"   test shapes with a train twin closer than the 5th percentile of train->train ({thr:.2f}): {int((t2t < thr).sum())} of {len(t2t)}")

print("6. what is in it")
tm = [m for m in meta if m["split"] == "train"]
nv = np.array([int(m["n_voids"]) for m in meta])
print(f"   images {len(meta)}: with 0 voids {int((nv == 0).sum())} ({100 * (nv == 0).mean():.0f}%), per-image void counts " + str({k: int((nv == k).sum()) for k in range(6)}))
ed = np.array([v["eq_diam_px"] for v in voids]); ar = np.array([v["area_px"] for v in voids]); ct = np.array([v["contrast"] for v in voids])
print(f"   voids {len(voids)}: equivalent diameter px p5={np.percentile(ed,5):.1f} p50={np.percentile(ed,50):.1f} p95={np.percentile(ed,95):.1f}; "
      f"smaller than 20 px area: {100 * (ar < 20).mean():.0f}%; physical contrast p5={np.percentile(ct,5):.3f} p50={np.percentile(ct,50):.3f} p95={np.percentile(ct,95):.3f}; "
      f"contrast below 0.05: {100 * (ct < 0.05).mean():.0f}%")
print(f"   part classes in train: " + str({m: sum(1 for t in tm if t['part_name'] == m) for m in sorted({t['part_name'] for t in tm})}))
print(f"   images with clustered voids: {sum(m['cluster'] == 'True' for m in meta)}")
