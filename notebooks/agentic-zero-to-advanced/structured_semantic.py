"""Part 4, second half: is the valid JSON also *right*?

Reads out/structured_output_results.json (written by structured_output_demo.py) and checks, against the
exact void boxes, what the structure cannot tell you: count and box fidelity, the zero-candidate path,
whether `type` carries any information, and whether the model's `confidence` ranks real voids above
false alarms any better than the detector's own `strength` number.
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

import void_data as vd
from baseline_detector import match
from detector_tool import inspect_xray

OUT = Path(__file__).parent / "out"


def auc(pos, neg):
    """P(score of a random positive > score of a random negative), ties count half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def main():
    results = json.loads((OUT / "structured_output_results.json").read_text())
    n = len(results)
    status = Counter(r["status"] for r in results)

    count_ok = box_ok = zero_n = zero_ok = 0
    rows = []  # one row per candidate: label, type, severity, confidence, strength
    dark = 0
    for r in results:
        split, i = r["image"].split("_")[0], int(r["image"].split("_")[1])
        tool = json.loads(inspect_xray(r["image"]))
        boxes = [(c["x1"], c["y1"], c["x2"], c["y2"]) for c in tool["candidates"]]
        if r["status"] == "needs_review":
            continue
        defects = r["defects"]
        count_ok += len(defects) == len(boxes)
        box_ok += [tuple(d["bbox"]) for d in defects] == boxes
        if not boxes:
            zero_n += 1
            zero_ok += defects == []
        gt = [tuple(v["bbox"]) for v in vd.load_voids(split, i)]
        mg, mc = match(gt, boxes)
        if len(defects) == len(boxes):
            for ci, d in enumerate(defects):
                rows.append({"tp": ci in mc, "type": d["type"], "severity": d["severity"],
                             "confidence": d["confidence"], "strength": tool["candidates"][ci]["strength"]})
                dark += bool(re.search(r"\bdark", d["explanation"], re.I))

    tp_rows = [x for x in rows if x["tp"]]
    fp_rows = [x for x in rows if not x["tp"]]
    summary = {
        "n_images": n,
        "status": dict(status),
        "count_fidelity": f"{count_ok}/{n - status.get('needs_review', 0)}",
        "box_fidelity": f"{box_ok}/{n - status.get('needs_review', 0)}",
        "zero_candidate_images": zero_n, "zero_candidate_answered_empty": zero_ok,
        "candidates": len(rows), "true_voids_among_them": len(tp_rows), "false_alarms": len(fp_rows),
        "type_among_true_voids": dict(Counter(x["type"] for x in tp_rows)),
        "type_among_false_alarms": dict(Counter(x["type"] for x in fp_rows)),
        "severity_among_true_voids": dict(Counter(x["severity"] for x in tp_rows)),
        "severity_among_false_alarms": dict(Counter(x["severity"] for x in fp_rows)),
        "confidence_values": dict(Counter(x["confidence"] for x in rows).most_common(8)),
        "auc_model_confidence": auc([x["confidence"] for x in tp_rows], [x["confidence"] for x in fp_rows]),
        "auc_detector_strength": auc([x["strength"] for x in tp_rows], [x["strength"] for x in fp_rows]),
        "explanations_saying_dark": dark,
    }
    (OUT / "structured_semantic_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
