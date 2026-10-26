"""Part 9: what does a human step buy, and what does it cost? Measured with simulated reviewers on the real held-out images.

A reviewer is simulated from the exact ground truth (the synthetic dataset has it), with an error rate: each candidate box is judged
wrongly with that probability. "careful" looks at every candidate of the image; "rubber_stamp" approves whatever the pipeline kept.
This is a model of a person, not a person: the post says so, and the error rate is a parameter you change.

Two ways to run the same policy:
  run_graph(...)   the real LangGraph with interrupt() and a checkpointer (offline classifier "band")
  simulate(...)    the same decision rules applied to a SAVED pipeline run (out/pipeline_results.json), so the Part 7 model runs are reused
test_hitl_paths.py and the 9b notebook check that the two agree where both exist.
"""
import json
import random
from functools import lru_cache
from pathlib import Path

from baseline_detector import match
from inspection_tools import N_PER_SPLIT, _CANDIDATES, exact_voids, find_candidates, score_kept
from hitl_graph import POLICIES, build_hitl, hitl_initial_state

OUT = Path(__file__).parent / "out"
SECONDS_PER_IMAGE = 30      # ASSUMPTION, not a measurement: how long a person looks at one image. Change it and rerun the cost lines.
TEST = [f"test_{i:05d}" for i in range(N_PER_SPLIT)]


@lru_cache(maxsize=None)
def truth(image: str) -> dict[int, bool]:
    """candidate id -> is it a real void (matched one-to-one to an exact ground-truth box at the Part 2 IoU threshold)."""
    find_candidates(image)
    boxes = _CANDIDATES[image]
    _, mc = match(exact_voids(image), boxes)
    return {i: i in mc for i in range(len(boxes))}


class Reviewer:
    """error_rate: chance of judging one box wrongly. rubber_stamp: approves what the pipeline kept without looking.
    scope: "all" judges every candidate of the image (can rescue a box the pipeline rejected); "kept" judges only the boxes the pipeline kept."""

    def __init__(self, name: str = "ana", error_rate: float = 0.0, rubber_stamp: bool = False, seed: int = 0, scope: str = "all"):
        self.name, self.error_rate, self.rubber_stamp, self.seed, self.scope = name, error_rate, rubber_stamp, seed, scope

    def decide(self, payload: dict) -> dict:
        if self.rubber_stamp:
            return {"action": "approve", "reviewer": self.name, "note": "looks fine"}
        t = truth(payload["image"])
        rng = random.Random(f"{self.seed}-{payload['image']}")      # same image, same mistakes: graph and simulation agree
        kept = sorted(c["id"] for c in payload["candidates"] if c["pipeline"].startswith("kept"))
        # one draw per candidate in id order whatever the scope, so scopes differ only in what the reviewer is allowed to change
        seen = {c["id"]: (t[c["id"]] ^ (rng.random() < self.error_rate)) for c in payload["candidates"]}
        real = sorted(i for i, v in seen.items() if v and (self.scope == "all" or i in kept))
        if real == kept:
            return {"action": "approve", "reviewer": self.name}
        if not real:
            return {"action": "reject", "reviewer": self.name}
        return {"action": "edit", "reviewer": self.name, "keep_ids": real}


def payload_from_report(image: str, report: dict, candidates: list[dict], measurements: list[dict], uncertain_verdict: str) -> dict:
    """Rebuild what the reviewer would have been shown from a saved pipeline report (used by simulate())."""
    kept = {k["id"]: k for k in report["kept"]}
    cands = [{"id": c["id"], "pipeline": "kept" if c["id"] in kept else "rejected"} for c in candidates]
    return {"image": image, "pipeline_verdict": uncertain_verdict, "candidates": cands}


def _final_ids(policy: str, image: str, verdict: str, kept_ids: list[int], cand_ids: list[int], reviewer: Reviewer):
    """-> (final kept ids, sent_to_human). Mirrors hitl_graph: decide -> route -> human_review | finalize."""
    pol = POLICIES[policy]
    if not pol.needs_human({"verdict": verdict, "image": image}):
        return kept_ids, False
    payload = {"image": image, "candidates": [{"id": i, "pipeline": "kept" if i in kept_ids else "rejected"} for i in cand_ids]}
    d = reviewer.decide(payload)
    if d["action"] == "approve":
        return kept_ids, True
    if d["action"] == "reject":
        return [], True
    return sorted(d["keep_ids"]), True


def _score(rows: list[dict]) -> dict:
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
    sent = [r for r in rows if r["sent"]]
    gt = tp + fn
    real_cands = sum(sum(truth(r["image"]).values()) for r in rows)
    return {"images": len(rows), "sent_to_human": len(sent), "boxes_shown": sum(r["n_candidates"] for r in sent),
            "human_minutes": round(len(sent) * SECONDS_PER_IMAGE / 60, 1),
            "tp": tp, "fp": fp, "fn": fn, "precision": round(tp / (tp + fp), 3) if tp + fp else None, "recall": round(tp / gt, 3),
            "ceiling_recall": round(real_cands / gt, 3),    # best any reviewer could reach: the detector never boxed the rest
            "passes_audited": sum(1 for r in sent if r["pipeline_verdict"] == "pass"),
            "passes_rescued": sum(1 for r in sent if r["pipeline_verdict"] == "pass" and r["final"])}


def simulate(saved_results: list[dict], policy: str, reviewer: Reviewer) -> dict:
    """Apply a policy + reviewer to a saved pipeline run (a list of per-image dicts from pipeline_results.json)."""
    rows = []
    for r in saved_results:
        im = r["image"]
        cand_ids = sorted(truth(im))
        kept_ids = [k["id"] for k in r["report"]["kept"]]
        final, sent = _final_ids(policy, im, r["verdict"], kept_ids, cand_ids, reviewer)
        tp, fp, fn = score_kept(im, final)
        rows.append({"image": im, "n_candidates": r["n_candidates"], "sent": sent, "tp": tp, "fp": fp, "fn": fn,
                     "verdict": "pass" if not final else "fail", "final": final, "pipeline_verdict": r["verdict"]})
    return {"summary": _score(rows), "rows": rows}


def run_graph(policy: str, reviewer: Reviewer, images: list[str] = TEST, classifier: str = "band") -> dict:
    """The real thing: a LangGraph with interrupt() per image, answered by the simulated reviewer through resume."""
    from langgraph.checkpoint.memory import MemorySaver
    import hitl_graph as h
    g = build_hitl(classifier, policy, checkpointer=MemorySaver())
    rows = []
    for im in images:
        r = h.start(g, im, im)
        sent = r["status"] == "waiting"
        payload = r["payload"] if sent else None
        if sent:
            r = h.resume(g, im, reviewer.decide(payload))
        s = r["state"]
        final = [k["id"] for k in s["kept"]]
        tp, fp, fn = score_kept(im, final)
        rows.append({"image": im, "n_candidates": len(s["candidates"]), "sent": sent, "tp": tp, "fp": fp, "fn": fn,
                     "verdict": s["verdict"], "final": final, "decided_by": s["decided_by"],
                     "pipeline_verdict": payload["pipeline_verdict"] if sent else s["verdict"]})
    return {"summary": _score(rows), "rows": rows}


def load_saved(variant: str, run: int = 0) -> list[dict]:
    return json.loads((OUT / "pipeline_results.json").read_text())["variants"][variant][run]["results"]
