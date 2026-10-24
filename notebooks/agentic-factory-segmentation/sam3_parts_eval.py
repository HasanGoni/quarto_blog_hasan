"""Choose how to turn SAM3's raw instances into ONE part mask per image (on val), then report on test, against the exact part masks.
Candidate rules: score threshold T in {0.05, 0.1, 0.2, 0.3, 0.5}; 'best' = the highest-scoring instance of the first prompt that has any,
'union' = all instances of that prompt at or above T; 'any prompt' = pool all prompts of the class. Writes out/sam3/part_masks_<split>.npz
for the chosen rule (every split that has a raw file). python sam3_parts_eval.py   (plain numpy)"""
import csv, json, pickle
from pathlib import Path
import numpy as np

root = Path(__file__).parent
D = (root / "data" / "void_dataset_v2").resolve()
O = root / "out/sam3"
PROMPTS = {"T-shirt/top": ["t-shirt", "shirt", "top"], "Trouser": ["trousers", "pants", "jeans"], "Pullover": ["sweater", "pullover", "sweatshirt"],
           "Dress": ["dress"], "Coat": ["coat", "jacket"], "Sandal": ["sandal", "shoe"], "Shirt": ["shirt", "blouse"],
           "Sneaker": ["sneaker", "shoe"], "Bag": ["bag", "handbag"], "Ankle boot": ["boot", "shoe"]}
meta = {r["id"]: r for r in csv.DictReader(open(D / "metadata.csv"))}
unpack = lambda b: np.unpackbits(b)[:224 * 224].reshape(224, 224).astype(bool)


def mask_for(raw, i, cls, T, mode):
    prompts = PROMPTS[cls]
    if mode.endswith("pool"):                                    # every prompt of the class, instances >= T, best or union over all
        inst = [(s, b) for p in prompts for s, b in raw.get((i, p), []) if s >= T]
        if not inst:
            return None, 0.0
        if mode == "best_pool":
            s, b = max(inst, key=lambda x: x[0]); return unpack(b), s
        return np.any([unpack(b) for s, b in inst], 0), max(s for s, b in inst)
    for p in prompts:                                            # first prompt with an instance >= T
        inst = [(s, b) for s, b in raw.get((i, p), []) if s >= T]
        if inst:
            if mode == "best":
                s, b = max(inst, key=lambda x: x[0]); return unpack(b), s
            return np.any([unpack(b) for s, b in inst], 0), max(s for s, b in inst)
    return None, 0.0


def evaluate(split, T, mode):
    raw = pickle.load(open(O / f"raw_{split}.pkl", "rb")); P = np.load(D / f"{split}.npz")["part_masks"] > 0
    ious, found = [], 0
    for i in range(len(P)):
        m, s = mask_for(raw, i, meta[f"{split}_{i:05d}"]["part_name"], T, mode)
        if m is None:
            ious.append(0.0); continue
        found += 1; ious.append((m & P[i]).sum() / max((m | P[i]).sum(), 1))
    ious = np.array(ious)
    return dict(found=found / len(P), miou=float(ious.mean()), iou50=float((ious >= .5).mean()), iou90=float((ious >= .9).mean()),
                miou_found=float(ious[ious > 0].mean()) if found else 0.0)


if __name__ == "__main__":
    grid = [(T, m) for T in (0.05, 0.1, 0.2, 0.3, 0.5) for m in ("best", "union", "best_pool", "union_pool")]
    res = {f"{m}@{T}": evaluate("val", T, m) for T, m in grid}
    print("VAL (choose here): rule, found, mean IoU (missing = 0), IoU>=0.5, IoU>=0.9")
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["miou"]):
        print(f"  {k:16s} found {v['found']:.3f}  mIoU {v['miou']:.3f}  >=.5 {v['iou50']:.3f}  >=.9 {v['iou90']:.3f}")
    best = max(res, key=lambda k: res[k]["miou"]); mode, T = best.split("@")[0], float(best.split("@")[1])
    print("chosen on val:", best)
    json.dump(dict(chosen=best, val=res[best]), open(O / "chosen.json", "w"))
    for split in ("val", "test", "train"):
        if (O / f"raw_{split}.pkl").exists():
            r = evaluate(split, T, mode); print(split, {k: round(v, 3) for k, v in r.items()})
            raw = pickle.load(open(O / f"raw_{split}.pkl", "rb"))
            n = len(np.load(D / f"{split}.npz")["part_masks"]); M = np.zeros((n, 224, 224), np.uint8); S = np.zeros(n, np.float32)
            for i in range(n):
                m, s = mask_for(raw, i, meta[f"{split}_{i:05d}"]["part_name"], T, mode)
                if m is not None:
                    M[i], S[i] = m, s
            np.savez_compressed(O / f"part_masks_{split}.npz", sam3_part_masks=M, score=S)
