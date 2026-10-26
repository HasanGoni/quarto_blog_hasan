"""Part 9: the Part 7 inspection pipeline with a human in the loop.

    detect -> measure -> classify -> rules -> decide --(needs a human?)--> human_review -> report
                                                 \\--(no)------------------------------> report

Part 6 showed `interrupt_before`: stop before a node. Here the pause lives INSIDE a node. `interrupt(payload)` stops the
run and hands `payload` to whoever reads the queue; `Command(resume=answer)` continues the run and `interrupt(...)` returns
`answer`. The state is saved by a checkpointer, so the run can wait for hours and survive a restart of the process.

Everything that is not the human step is imported unchanged from pipeline_graph.py (Part 7).

Final verdicts: "pass" (nothing to act on) or "fail" (at least one defect confirmed). `decided_by` says who decided.
"""
import operator
from dataclasses import dataclass
from typing import Annotated, Callable, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ValidationError

from pipeline_graph import (CLASSIFIERS, PipelineState, after_classify, after_detect, decide_node, detect_node,
                            initial_state, measure_node, report_node, rules_node, template_summary)


class HitlState(PipelineState):
    decided_by: str                         # "pipeline" | "human" | "auto-fail"
    human: dict | None                      # the reviewer's decision, kept for the audit trail
    audit: Annotated[list, operator.add]    # one line per human interaction (including rejected answers)


# --- the question we ask and the answer we accept ------------------------------------------------------------------
class Decision(BaseModel):
    action: Literal["approve", "reject", "edit"]   # approve: the kept boxes are right. reject: all false alarms. edit: keep_ids
    reviewer: str
    keep_ids: list[int] | None = None              # only for "edit"
    note: str = ""


class DecisionError(ValueError):
    pass


def parse_decision(raw, valid_ids: set[int]) -> Decision:
    """The reviewer is a person using a form: never trust the shape of the answer. Raise DecisionError with a sentence they can act on."""
    try:
        d = Decision(**raw) if isinstance(raw, dict) else None
    except ValidationError as exc:
        raise DecisionError("; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())) from exc
    if d is None:
        raise DecisionError(f"expected a JSON object, got {type(raw).__name__}")
    if not d.reviewer.strip():
        raise DecisionError("reviewer must not be empty: every decision needs a name for the audit trail")
    if d.action == "edit":
        if d.keep_ids is None:
            raise DecisionError("action 'edit' needs keep_ids (use [] to clear everything, or choose 'reject')")
        bad = sorted(set(d.keep_ids) - valid_ids)
        if bad:
            raise DecisionError(f"keep_ids {bad} are not candidates of this image (valid: {sorted(valid_ids)})")
    elif d.keep_ids is not None:
        raise DecisionError(f"keep_ids is only allowed with action 'edit', not '{d.action}'")
    return d


def review_payload(state: HitlState) -> dict:
    """Exactly what the reviewer is shown. Plain JSON, so any UI (notebook, web form, chat message) can render it.
    Every candidate of the image is listed with what the pipeline did to it, so the reviewer can also rescue a box the pipeline threw away."""
    by_id = {m["candidate_id"]: m for m in state["measurements"]}
    kept = {k["id"]: k for k in state["kept"]}
    rejected = {r["id"]: r["by"] for r in state["rejected"]}
    cands = []
    for c in state["candidates"]:
        m = by_id[c["id"]]
        cands.append({"id": c["id"], "box": c["box"], "contrast": m["contrast"], "area_px": m["area_px"],
                      "edge_distance_px": m["edge_distance_px"],
                      "pipeline": "kept" + (" (uncertain)" if kept[c["id"]]["uncertain"] else "") if c["id"] in kept
                                  else f"rejected by {rejected.get(c['id'], 'classifier failure')}"})
    return {
        "image": state["image"],
        "pipeline_verdict": state["verdict"],
        "why_a_human": state["reason"],
        "candidates": cands,
        "all_candidate_ids": [c["id"] for c in cands],
        "question": "approve = the pipeline's kept boxes are exactly the real defects; reject = none is real; "
                    "edit = give the ids that are real (keep_ids), including boxes the pipeline rejected",
    }


# --- the human step -------------------------------------------------------------------------------------------------
REPLAYS: list[str] = []   # demo only: records every time the node body starts, to show that a resume re-runs it from the top


def human_review_node(state: HitlState) -> dict:
    REPLAYS.append(state["image"])           # anything here runs AGAIN on resume: keep code before interrupt() side-effect free
    payload = review_payload(state)
    valid = set(payload["all_candidate_ids"])
    audit = []
    while True:
        raw = interrupt(payload)             # first pass: pauses here. After Command(resume=x): returns x
        try:
            d = parse_decision(raw, valid)
            break
        except DecisionError as exc:         # bad answer: do not crash, do not guess, ask again with the reason
            audit.append({"image": state["image"], "event": "answer rejected", "why": str(exc), "answer": raw})
            payload = {**payload, "previous_answer_rejected": str(exc)}
    before = [k["id"] for k in state["kept"]]
    if d.action == "approve":
        after = before
    elif d.action == "reject":
        after = []
    else:
        after = sorted(d.keep_ids)
    kept = [k for k in state["kept"] if k["id"] in after]
    # a reviewer may rescue a box the pipeline rejected: give it the same record shape, flagged as human-added
    have = {k["id"] for k in kept}
    by_id = {m["candidate_id"]: m for m in state["measurements"]}
    for i in after:
        if i not in have:
            m = by_id[i]
            kept.append({"id": i, "severity": "major", "uncertain": False, "confidence": 1.0, "contrast": m["contrast"],
                         "area_px": m["area_px"], "added_by_human": True})
    verdict = "fail" if kept else "pass"
    reason = f"{d.reviewer} chose '{d.action}'" + (f": {d.note}" if d.note else "")
    human = {"reviewer": d.reviewer, "action": d.action, "kept_before": before, "kept_after": sorted(after), "note": d.note}
    return {"kept": kept, "verdict": verdict, "reason": reason, "decided_by": "human", "human": human,
            "audit": audit + [{"image": state["image"], "event": "decided", **human}],
            "trace": [{"node": "human_review", "action": d.action}]}


# --- who needs a human ------------------------------------------------------------------------------------------------
@dataclass
class Policy:
    """Which pipeline verdicts go to a person. A verdict that skips the human is final, so every skipped class is a risk you accept."""
    name: str
    needs_human: Callable[[dict], bool]
    auto_fail_confident: bool = False       # a "review" verdict that skips the human becomes a "fail" on its own


def _audit_pick(image: str, every: int) -> bool:
    return sum(map(ord, image)) % every == 0   # deterministic "random" sample: the same images every run


POLICIES = {
    "none": Policy("none: the pipeline decides alone", lambda s: False, auto_fail_confident=True),
    "escalate_only": Policy("humans see only 'escalate'", lambda s: s["verdict"] == "escalate", auto_fail_confident=True),
    "all_nonpass": Policy("humans see 'review' and 'escalate'", lambda s: s["verdict"] in ("review", "escalate")),
    "all_plus_audit": Policy("'review' + 'escalate' + 1 in 10 of the passes",
                             lambda s: s["verdict"] in ("review", "escalate") or (s["verdict"] == "pass" and _audit_pick(s["image"], 10))),
}


def finalize_node(state: HitlState) -> dict:
    """No human involved: the pipeline's verdict becomes final. 'review' and 'escalate' only exist as hand-off words."""
    v = state["verdict"]
    return {"verdict": "pass" if v == "pass" else "fail", "decided_by": "pipeline" if v == "pass" else "auto-fail",
            "trace": [{"node": "finalize", "from": v}]}


def template_report_node(state: HitlState) -> dict:
    """Same report record as Part 7, but the sentence comes from a template: no model call, so the whole graph runs offline."""
    report = {"image": state["image"], "verdict": state["verdict"], "reason": state["reason"],
              "kept": state.get("kept") or [], "rejected": state.get("rejected") or [],
              "summary": template_summary(state), "summary_source": "template", "decided_by": state["decided_by"]}
    return {"report": report, "trace": [{"node": "report", "source": "template"}]}


def build_hitl(classifier: str = "band", policy: str = "all_nonpass", prose: bool = False, **compile_kwargs):
    """classifier: "band" needs no model. policy: see POLICIES. prose=True lets the model write the note (Part 7); default is a template."""
    pol = POLICIES[policy]
    route = lambda s: "human_review" if pol.needs_human(s) else "finalize"
    g = StateGraph(HitlState)
    g.add_node("detect", detect_node)
    g.add_node("measure", measure_node)
    g.add_node("classify", CLASSIFIERS[classifier])
    g.add_node("rules", rules_node)
    g.add_node("decide", decide_node)
    g.add_node("human_review", human_review_node)
    g.add_node("finalize", finalize_node)
    g.add_node("report", report_node if prose else template_report_node)
    g.add_edge(START, "detect")
    g.add_conditional_edges("detect", after_detect, ["measure", "report"])
    g.add_edge("measure", "classify")
    g.add_conditional_edges("classify", after_classify, ["rules", "decide"])
    g.add_edge("rules", "decide")
    g.add_conditional_edges("decide", route, ["human_review", "finalize"])
    g.add_edge("human_review", "report")
    g.add_edge("finalize", "report")
    g.add_edge("report", END)
    return g.compile(**compile_kwargs)


def hitl_initial_state(image: str) -> HitlState:
    return {**initial_state(image), "decided_by": "pipeline", "human": None, "audit": []}


def start(graph, image: str, thread_id: str) -> dict:
    """Run until done or until the graph asks a person. Returns {"status": "done"|"waiting", ...}."""
    cfg = {"configurable": {"thread_id": thread_id}}
    out = graph.invoke(hitl_initial_state(image), cfg)
    return _status(graph, out, cfg)


def resume(graph, thread_id: str, answer) -> dict:
    cfg = {"configurable": {"thread_id": thread_id}}
    out = graph.invoke(Command(resume=answer), cfg)
    return _status(graph, out, cfg)


def _status(graph, out: dict, cfg: dict) -> dict:
    snap = graph.get_state(cfg)
    if snap.interrupts:
        return {"status": "waiting", "payload": snap.interrupts[0].value, "waiting_at": list(snap.next)}
    return {"status": "done", "state": out}
