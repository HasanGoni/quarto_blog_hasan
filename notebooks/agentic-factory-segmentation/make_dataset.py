"""Build the dataset: python make_dataset.py [out_dir]   (about a minute; deterministic)"""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
from void_dataset import CLASS_NAMES, N, SEED, image_rng, load_fmnist, make_sample, split_indices

root = Path(__file__).parent
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data" / "void_dataset_v1"
out.mkdir(parents=True, exist_ok=True)
X, y = load_fmnist(root / "data" / "fashion_mnist")
sizes = {"train": 2800, "val": 600, "test": 600}
idx = split_indices(len(X), sizes)
meta, voids_all, digest = [], [], hashlib.sha256()
for sid, (split, src) in enumerate(idx.items()):
    imgs, vm, pm = (np.zeros((len(src), N, N), np.uint8) for _ in range(3))
    for i, s in enumerate(src):
        img, void, part, prm, voids = make_sample(X[s], image_rng(sid, i))
        imgs[i], vm[i], pm[i] = img, void, part
        iid = f"{split}_{i:05d}"
        meta.append(dict(id=iid, split=split, src_index=int(s), part_class=int(y[s]), part_name=CLASS_NAMES[y[s]],
                         **{k: (round(float(v), 5) if isinstance(v, float) else v) for k, v in prm.items()}))
        voids_all += [dict(id=iid, split=split, **v) for v in voids]
    np.savez_compressed(out / f"{split}.npz", images=imgs, void_masks=vm, part_masks=pm)
    for a in (imgs, vm, pm):
        digest.update(a.tobytes())
    print(split, len(src), "images,", int(vm.any((1, 2)).sum()), "with at least one void")
with open(out / "metadata.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(meta[0])); w.writeheader(); w.writerows(meta)
(out / "voids.jsonl").write_text("\n".join(json.dumps(v) for v in voids_all))
(out / "README.txt").write_text(
    "Synthetic void dataset v1: Fashion-MNIST silhouettes as parts, simulated X-ray attenuation, synthetic voids with exact masks.\n"
    "Source shapes: Fashion-MNIST (MIT licence, Copyright (c) 2017 Zalando SE, https://github.com/zalandoresearch/fashion-mnist).\n"
    f"Seed {SEED}. Splits are disjoint by source item. sha256 of all arrays: {digest.hexdigest()}\n")
print("voids:", len(voids_all), "| sha256", digest.hexdigest()[:16], "| written to", out)
