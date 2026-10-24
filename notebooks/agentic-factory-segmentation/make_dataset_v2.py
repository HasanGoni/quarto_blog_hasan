"""Build v2: python make_dataset_v2.py [out_dir]  (a few minutes; deterministic)"""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
from void_dataset_v2 import CLASS_NAMES, N, SEED, image_rng, load_fmnist, make_sample

root = Path(__file__).parent
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data" / "void_dataset_v2"
out.mkdir(parents=True, exist_ok=True)
X, y = load_fmnist(root / "data" / "fashion_mnist")
sizes = {"train": 2800, "val": 600, "test": 600}
# sources are visited in a fixed order per split; unusable ones are skipped, so splits stay disjoint by source item
perm = np.random.default_rng(SEED + 2).permutation(len(X)); pos = 0
meta, defects, digest, skipped = [], [], hashlib.sha256(), 0
for sid, (split, n) in enumerate(sizes.items()):
    imgs, mk, pm = (np.zeros((n, N, N), np.uint8) for _ in range(3)); i = 0
    while i < n:
        s = int(perm[pos]); pos += 1
        r = make_sample(X[s], int(y[s]), image_rng(sid, i))
        if r is None:
            skipped += 1; continue
        img, mask, part, prm, recs = r
        imgs[i], mk[i], pm[i] = img, mask, part
        iid = f"{split}_{i:05d}"
        meta.append(dict(id=iid, split=split, src_index=s, part_class=int(y[s]), part_name=CLASS_NAMES[y[s]],
                         **{k: (round(float(v), 5) if isinstance(v, float) else v) for k, v in prm.items()}))
        defects += [dict(id=iid, split=split, **d) for d in recs]
        i += 1
    np.savez_compressed(out / f"{split}.npz", images=imgs, defect_masks=mk, part_masks=pm)
    for a in (imgs, mk, pm):
        digest.update(a.tobytes())
    print(split, n, "images,", int((mk > 0).any((1, 2)).sum()), "with a defect")
with open(out / "metadata.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(meta[0])); w.writeheader(); w.writerows(meta)
(out / "defects.jsonl").write_text("\n".join(json.dumps(d) for d in defects))
(out / "README.txt").write_text(
    "Synthetic defect dataset v2: voids (mask 1) and cracks (mask 2) on Fashion-MNIST silhouettes, simulated X-ray attenuation.\n"
    "Source shapes: Fashion-MNIST (MIT licence, Copyright (c) 2017 Zalando SE, https://github.com/zalandoresearch/fashion-mnist).\n"
    f"Seed {SEED + 2}. Every defect has a visibility score >= 8 at generation time. Parts are flattened (no inner texture). Skipped sources: {skipped}.\n"
    f"sha256 of all arrays: {digest.hexdigest()}\n")
nv = sum(d["type"] == "void" for d in defects)
print("voids", nv, "cracks", len(defects) - nv, "skipped sources", skipped, "| sha256", digest.hexdigest()[:16])
