"""Run SAM3 (facebook/sam3, text prompts) on every v2 image and keep every instance it returns at score >= 0.05.
The raw instances go to out/sam3/raw_<split>.pkl; sam3_parts_eval.py chooses a rule on val and reports on test.
Needs transformers with Sam3 (run with notebooks/paper-hpma-sam3/.venv/bin/python).  Prompts per Fashion-MNIST class are my guesses."""
import pickle, sys, time, csv
from pathlib import Path
import numpy as np, torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

root = Path(__file__).parent
D = (root / "data" / "void_dataset_v2").resolve()
PROMPTS = {"T-shirt/top": ["t-shirt", "shirt", "top"], "Trouser": ["trousers", "pants", "jeans"], "Pullover": ["sweater", "pullover", "sweatshirt"],
           "Dress": ["dress"], "Coat": ["coat", "jacket"], "Sandal": ["sandal", "shoe"], "Shirt": ["shirt", "blouse"],
           "Sneaker": ["sneaker", "shoe"], "Bag": ["bag", "handbag"], "Ankle boot": ["boot", "shoe"]}
MIN_SCORE, BATCH = 0.05, 16
dev = "cuda"
model = Sam3Model.from_pretrained("facebook/sam3").to(dev).eval()
proc = Sam3Processor.from_pretrained("facebook/sam3")
meta = list(csv.DictReader(open(D / "metadata.csv")))
for split in sys.argv[1:] or ["val", "test", "train"]:
    X = np.load(D / f"{split}.npz")["images"]
    rows = [r for r in meta if r["split"] == split]
    jobs = {}                                                    # prompt -> image indices
    for r in rows:
        for p in PROMPTS[r["part_name"]]:
            jobs.setdefault(p, []).append(int(r["id"].split("_")[1]))
    raw, t0, n = {}, time.time(), 0
    for p, ids in jobs.items():
        for b in range(0, len(ids), BATCH):
            bi = ids[b:b + BATCH]
            imgs = [Image.fromarray(X[i]).convert("RGB") for i in bi]
            inp = proc(images=imgs, text=[p] * len(bi), return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**inp)
            res = proc.post_process_instance_segmentation(out, threshold=MIN_SCORE, mask_threshold=0.5, target_sizes=[(224, 224)] * len(bi))
            for i, r in zip(bi, res):
                raw[(i, p)] = [(float(s), np.packbits(m.cpu().numpy().astype(bool))) for s, m in zip(r["scores"], r["masks"])]
            n += len(bi)
        print(split, p, f"{n} runs {time.time() - t0:.0f}s", flush=True)
    pickle.dump(raw, open(root / "out/sam3" / f"raw_{split}.pkl", "wb"))
    print(split, "done", len(raw), "runs", f"{time.time() - t0:.0f}s", flush=True)
