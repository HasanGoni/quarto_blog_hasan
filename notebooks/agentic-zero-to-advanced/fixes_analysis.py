"""Collects the numbers for the 'why agents fail' section. Needs agent_loop.py's
out/agent_loop_results.json and (for the fix rows) the fixes_*.json from agent_fixes.py."""
import json
from pathlib import Path

from agent_fixes import summarize, verify_report
from inspection_tools import find_candidates, score_kept

OUT = Path(__file__).parent / "out"
runs = json.loads((OUT / "agent_loop_results.json").read_text())


def plain_detector(rows):
    tp = fp = fn = 0
    for r in rows:
        find_candidates(r["image"])
        t, f, n = score_kept(r["image"], list(range(r["n_candidates"])))
        tp, fp, fn = tp + t, fp + f, fn + n
    return {"n_images": len(rows), "kept": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp / (tp + fp), 3), "recall": round(tp / (tp + fn), 3)}


dev, test = [r for r in runs if r["split"] == "val"], [r for r in runs if r["split"] == "test"]
out = {"plain_detector_dev": plain_detector(dev), "plain_detector_test": plain_detector(test),
       "original_agent_dev": summarize(dev), "original_agent_test": summarize(test)}

# How many of the original agent's reports would the verifier have bounced?
flagged, reasons = 0, {}
for r in runs:
    measured = {c["args"]["candidate_id"] for s in r["trace"] for c in s["calls"] if c["tool"] == "measure_candidate"}
    rep = next((c["args"] for s in r["trace"] for c in s["calls"] if c["tool"] == "submit_report"), None)
    problems = verify_report(rep, measured, r["n_candidates"]) if rep else ["no report"]
    flagged += bool(problems)
    for p in problems:
        key = p.split(" are ")[0].split(" were ")[0].split(" without")[0][:40]
        reasons[key] = reasons.get(key, 0) + 1
out["verifier_on_original_agent"] = {"reports": len(runs), "flagged": flagged}
for v in ("design", "verify", "both"):
    f = OUT / f"fixes_{v}.json"
    if f.exists():
        out[f"fix_{v}_test"] = json.loads(f.read_text())["summary"]
(OUT / "fixes_summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))


def passes_gate(agent: dict, baseline: dict, min_recall_ratio: float = 0.8) -> bool:
    """Task-level check, run on labeled frames before trusting an agent: it has to beat the plain
    detector's precision and keep most of its recall. Superficial checks (valid JSON, finished
    without error) would have passed every agent in this post."""
    return agent["precision"] >= baseline["precision"] and agent["recall"] >= min_recall_ratio * baseline["recall"]


gate = {"original_agent_test": passes_gate(out["original_agent_test"], out["plain_detector_test"])}
for v in ("design", "verify", "both"):
    if f"fix_{v}_test" in out:
        gate[f"fix_{v}_test"] = passes_gate(out[f"fix_{v}_test"], out["plain_detector_test"])
out["gate"] = gate
(OUT / "fixes_summary.json").write_text(json.dumps(out, indent=2))
print("gate:", gate)
