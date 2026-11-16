"""Step 3, the comparison. Scores the best-on-val adapter ONCE on the same 200 TEST images as the zero-shot run, then puts both side by side
with 95% intervals over images.

    .venv/bin/python eval_test.py --adapter out/lora_s2_epoch2 --scale 2
"""
import argparse
import json

from peft import PeftModel
from wevisone_vl import WeVisOneVL

import common as c

N_TEST, BATCH = 200, 8


def per_image_from_raw(path, split="test"):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r["split"] == split]
    out, out_s = [], []
    for r in rows:
        img = c.load_image(split, r["i"], r["scale"])
        pred, gt = c.parse_boxes(r["response"], img.width, img.height, r["scale"]), c.gt_boxes(split, r["i"])
        out.append(c.score(gt, pred)); out_s.append(c.score(gt, pred, c.HIT_STRICT))
    return out, out_s


def summarise(per, per_s):
    row = {}
    for name, p in (("iou0.2", per), ("iou0.5", per_s)):
        tot = [sum(x[k] for x in p) for k in range(3)]
        row[name] = c.prf(*tot)
        row[name]["ci"] = {k: c.bootstrap_ci(p, k) for k in ("precision", "recall", "f1")}
    row["boxes_per_image"] = round(sum(x[0] + x[1] for x in per) / len(per), 2)
    row["images_with_a_hit"] = sum(1 for x in per if x[0] > 0)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--scale", type=int, default=2)
    a = ap.parse_args()
    engine = WeVisOneVL("tencent/WeVisOne-VL-4B", backend="transformers")
    engine.model = PeftModel.from_pretrained(engine.model, a.adapter)
    engine.model.eval()
    per, per_s = [], []
    with open(c.OUT / "ft_test_raw.jsonl", "w") as raw:
        for start in range(0, N_TEST, BATCH):
            idx = list(range(start, min(N_TEST, start + BATCH)))
            imgs = [c.load_image("test", i, a.scale) for i in idx]
            res = engine.predict_batch(imgs, task="detection", query=[c.LABEL], decode_matting=False, max_new_tokens=1024, temperature=0.0)
            for i, img, r in zip(idx, imgs, res):
                pred, gt = c.parse_boxes(r.response, img.width, img.height, a.scale), c.gt_boxes("test", i)
                per.append(c.score(gt, pred)); per_s.append(c.score(gt, pred, c.HIT_STRICT))
                raw.write(json.dumps({"split": "test", "i": i, "query": [c.LABEL], "scale": a.scale, "response": r.response,
                                      "n_pred": len(pred), "n_gt": len(gt)}) + "\n")
    zs, zs_s = per_image_from_raw(c.OUT / "zs_test_raw.jsonl")
    res = {"zero_shot": summarise(zs, zs_s), "fine_tuned": summarise(per, per_s), "adapter": a.adapter}
    json.dump(res, open(c.OUT / "compare_test.json", "w"), indent=1)
    for k in ("zero_shot", "fine_tuned"):
        r = res[k]
        print(k, "boxes/img", r["boxes_per_image"], "IoU0.2", {m: (r["iou0.2"][m], r["iou0.2"]["ci"][m]) for m in ("precision", "recall", "f1")},
              "IoU0.5 F1", r["iou0.5"]["f1"], r["iou0.5"]["ci"]["f1"])


if __name__ == "__main__":
    main()
