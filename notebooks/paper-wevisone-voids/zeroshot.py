"""Step 1, no fine-tuning: WeVisOne-VL-4B asked for voids with the paper's detection prompt.

    .venv/bin/python zeroshot.py dev     # grid of (query word, image scale) on the first 100 VAL images; writes out/zs_dev.json
    .venv/bin/python zeroshot.py test    # the config chosen on dev, on the first 200 TEST images (the series' held-out set), once

The choices (which word to ask for, how much to enlarge the image) are made on val only, the series' protocol.
Raw model responses are kept in out/zs_*_raw.jsonl so every number can be recomputed from the text the model wrote.
"""
import json
import sys
import time

from wevisone_vl import WeVisOneVL

import common as c

N_DEV, N_TEST, BATCH = 100, 200, 8
QUERIES = [["void"], ["air bubble"], ["bright spot"], ["defect"]]
SCALES = [1, 2, 4]


def load_model():
    return WeVisOneVL("tencent/WeVisOne-VL-4B", backend="transformers")


def run_config(model, split: str, n: int, query: list, scale: int, raw_file) -> dict:
    per_image, t0 = [], time.time()
    per_image_strict = []
    for start in range(0, n, BATCH):
        idx = list(range(start, min(n, start + BATCH)))
        imgs = [c.load_image(split, i, scale) for i in idx]
        res = model.predict_batch(imgs, task="detection", query=query, decode_matting=False, max_new_tokens=1024, temperature=0.0)
        for i, img, r in zip(idx, imgs, res):
            pred = c.parse_boxes(r.response, img.width, img.height, scale)
            gt = c.gt_boxes(split, i)
            per_image.append(c.score(gt, pred))
            per_image_strict.append(c.score(gt, pred, c.HIT_STRICT))
            raw_file.write(json.dumps({"split": split, "i": i, "query": query, "scale": scale, "response": r.response,
                                       "n_pred": len(pred), "n_gt": len(gt)}) + "\n")
    tot = [sum(x[k] for x in per_image) for k in range(3)]
    tot_s = [sum(x[k] for x in per_image_strict) for k in range(3)]
    return {"query": query, "scale": scale, "images": n, "iou0.2": c.prf(*tot), "iou0.5": c.prf(*tot_s),
            "boxes_per_image": round(sum(x[0] + x[1] for x in per_image) / n, 2), "seconds": round(time.time() - t0, 1)}


def main():
    mode = sys.argv[1]
    c.OUT.mkdir(exist_ok=True)
    model = load_model()
    if mode == "dev":
        rows = []
        with open(c.OUT / "zs_dev_raw.jsonl", "w") as raw:
            for scale in SCALES:
                for q in QUERIES:
                    row = run_config(model, "val", N_DEV, q, scale, raw)
                    rows.append(row)
                    print(f"scale {scale} {q[0]:12s} boxes/img {row['boxes_per_image']:5.2f}  IoU0.2 P {row['iou0.2']['precision']} R {row['iou0.2']['recall']} "
                          f"F1 {row['iou0.2']['f1']}   IoU0.5 F1 {row['iou0.5']['f1']}   {row['seconds']}s", flush=True)
        best = max(rows, key=lambda r: r["iou0.2"]["f1"])
        json.dump({"grid": rows, "chosen": {"query": best["query"], "scale": best["scale"]}, "protocol": "first 100 val images"},
                  open(c.OUT / "zs_dev.json", "w"), indent=1)
        print("chosen on dev:", best["query"], "scale", best["scale"])
    else:
        ch = json.load(open(c.OUT / "zs_dev.json"))["chosen"]
        with open(c.OUT / "zs_test_raw.jsonl", "w") as raw:
            row = run_config(model, "test", N_TEST, ch["query"], ch["scale"], raw)
        json.dump(row, open(c.OUT / "zs_test.json", "w"), indent=1)
        print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
