"""Part 8: build the two evaluation sets from the pipeline's own runs on the synthetic void dataset.

A. candidates.jsonl : every box the detector proposed on the first 200 dev (val) and first 200 held-out (test) images, the measurement
                      record the pipeline's classify node sees, and the exact label (the box hits a real void at IoU >= 0.2).
B. notes.jsonl      : every report note the model wrote in Part 7's six LLM runs (3 guided + 3 unguided). The answers come from
                      the structured record that produced the note (verdict, kept boxes, uncertain flags), not from labelling the prose.
"""
import json

from baseline_detector import match
from inspection_tools import N_PER_SPLIT, exact_voids, find_candidates, measure_candidate

OUT = __import__("pathlib").Path(__file__).parent / "out" / "typed"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for split, name in (("val", "dev"), ("test", "test")):
    for i in range(N_PER_SPLIT):
        im = f"{split}_{i:05d}"
        cands = find_candidates(im)["candidates"]
        _, hit = match(exact_voids(im), [tuple(c["box"]) for c in cands])
        for c in cands:
            m = measure_candidate(im, c["id"])
            rows.append({"split": name, "image": im, "id": c["id"], "y": int(c["id"] in hit),
                         "state": {"contrast": m["contrast"], "area_px": m["area_px"], "aspect_ratio": m["aspect_ratio"], "edge_distance_px": m["edge_distance_px"]}})
(OUT / "candidates.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
for s in ("dev", "test"):
    sel = [r for r in rows if r["split"] == s]
    print(f"candidates {s}: {len(sel)} boxes, {sum(r['y'] for r in sel)} real")

res = json.load(open(OUT.parent / "pipeline_results.json"))["variants"]
notes = []
for variant in ("llm", "llm_noguide"):
    for run_i, run in enumerate(res[variant]):
        for r in run["results"]:
            if r["summary_source"] != "llm":
                continue
            kept = r["report"]["kept"]
            notes.append({"variant": variant, "run": run_i, "image": r["image"], "note": r["report"]["summary"], "verdict": r["verdict"],
                          "y_escalate": int(r["verdict"] == "escalate"), "y_uncertain": int(any(k["uncertain"] for k in kept)),
                          "n_kept": len(kept), "kept": kept})
(OUT / "notes.jsonl").write_text("\n".join(json.dumps(n) for n in notes))
print(f"notes: {len(notes)} | escalate {sum(n['y_escalate'] for n in notes)} | uncertain {sum(n['y_uncertain'] for n in notes)} | "
      f"n_kept {dict(sorted({k: sum(n['n_kept'] == k for n in notes) for k in {n['n_kept'] for n in notes}}.items()))}")
