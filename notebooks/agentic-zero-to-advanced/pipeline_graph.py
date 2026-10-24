"""Part 7: the inspection pipeline as a LangGraph graph. Code owns the order, the model owns two judgments.

    detect -> measure -> classify -> rules -> decide -> report
       |                     |                            ^
       +-- no candidates ----+----------------------------+   (and: classifier failed -> decide)

Compared with Part 5/6 there is no agent loop. The same two tools (find_candidates, measure_candidate)
are called by code, in a fixed order, once per image. The model is used in exactly two nodes:
`classify` (one batched structured call for all candidates) and `report` (prose, only when a human
will read it). Business rules and the verdict are plain Python.
"""
import json
import operator
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from agent_loop import MODEL, OUT_DIR, client
from inspection_tools import N_PER_SPLIT, find_candidates, measure_candidate, score_kept
from structured_output_demo import strip_fences

# --- policy: the numbers a QC owner would sign off, not the model ---------------------------------
MIN_AREA_PX = 20          # R2: smaller than this is not reportable (dev: 4% of 415 boxes under 20 px were real)
BAND_MIN_CONTRAST = 8.0   # `band` classifier floor, the best nonzero F1 on dev (out/band_dev_sweep.json)
CONF_MIN = 0.6            # below this a "defect" label is treated as uncertain
MAX_KEPT = 4              # more survivors than this on one image: the detector is confused, ask a human
SEVERITY_BANDS = [(60, "minor"), (250, "major")]  # placeholder pixel-area bands, else "critical"
MAX_CLASSIFY_ATTEMPTS = 2

GUIDANCE = """What the labeled dev data says about this detector:
- Contrast is the best signal. Boxes above 30 are real about 7 times in 10, 20 to 30 about 1 time in 3,
  15 to 20 about 1 time in 5, and below 15 about 1 time in 10 or fewer.
- Larger boxes are more often real than tiny ones."""

SYSTEM_CLASSIFY = """You classify candidate boxes from a noisy classical detector on X-ray images of
metal parts. You get one measurement record per candidate and no image. Most candidates are false alarms.
For every candidate choose a label: "defect", "false_alarm" or "uncertain", and a confidence from 0 to 1.
{guidance}
Return ONLY a JSON object, no markdown fences:
{{"labels": [{{"id": 0, "label": "defect", "confidence": 0.8, "reason": "one short sentence"}}]}}
Exactly one entry per candidate id, using the ids you were given."""

SYSTEM_REPORT = """You write the short note a QC engineer reads about one X-ray image. You get a JSON record of
what the inspection pipeline decided. Write 2 to 3 plain sentences. Use only facts in the record. Refer to
candidates only by the ids in the record. Do not add advice, do not change the verdict."""


class Label(BaseModel):
    id: int
    label: Literal["defect", "false_alarm", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class Labels(BaseModel):
    labels: list[Label]


class PipelineState(TypedDict):
    image: str
    candidates: list[dict]       # from detect
    measurements: list[dict]     # from measure
    labels: list[dict]           # from classify
    classify_status: str         # "ok" | "failed" | "skipped"
    kept: list[dict]             # survivors after rules, each with a severity
    rejected: list[dict]         # everything dropped, with the reason: the audit trail
    verdict: str                 # "pass" | "review" | "escalate"
    reason: str
    report: dict | None
    trace: Annotated[list, operator.add]   # reducer: one line per node that ran
    llm_calls: int
    tokens: int


# --- node 1-2: deterministic, the Part 5 tools called by code --------------------------------------
def detect_node(state: PipelineState) -> dict:
    cands = find_candidates(state["image"])["candidates"]
    out = {"candidates": cands, "trace": [{"node": "detect", "n": len(cands)}]}
    if not cands:
        out |= {"verdict": "pass", "reason": "the detector found no candidates"}
    return out


def measure_node(state: PipelineState) -> dict:
    ms = [measure_candidate(state["image"], c["id"]) for c in state["candidates"]]
    return {"measurements": ms, "trace": [{"node": "measure", "n": len(ms)}]}


# --- node 3: classify. Three interchangeable classifiers, same output shape -----------------------
def llm_labels(state: PipelineState, guided: bool) -> dict:
    """One batched call for all candidates. Validate, retry once with the exact error, else give up."""
    system = SYSTEM_CLASSIFY.format(guidance=GUIDANCE if guided else "Use your own judgment.")
    ids = {m["candidate_id"] for m in state["measurements"]}
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Candidates:\n" + json.dumps(state["measurements"])}]
    calls = tokens = 0
    error = "no attempt"
    for _ in range(MAX_CLASSIFY_ATTEMPTS):
        resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.2)
        calls, tokens = calls + 1, tokens + resp.usage.total_tokens
        raw = resp.choices[0].message.content or ""
        try:
            parsed = Labels(**json.loads(strip_fences(raw)))
            got = [lab.id for lab in parsed.labels]
            if sorted(got) != sorted(ids):  # well-formed is not enough: it must cover exactly our ids
                raise ValueError(f"expected exactly one label for each id in {sorted(ids)}, got {sorted(got)}")
            return {"labels": [lab.model_dump() for lab in parsed.labels], "classify_status": "ok",
                    "llm_calls": state["llm_calls"] + calls, "tokens": state["tokens"] + tokens,
                    "trace": [{"node": "classify", "attempts": calls}]}
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"That failed validation:\n{error}\nReturn ONLY the corrected JSON."}]
    return {"labels": [], "classify_status": "failed", "llm_calls": state["llm_calls"] + calls,
            "tokens": state["tokens"] + tokens, "trace": [{"node": "classify", "attempts": calls, "error": error[:200]}]}


def band_labels(state: PipelineState) -> dict:
    """No model. A contrast floor chosen on dev. Confidence is a flat 1.0: a rule has no doubt."""
    labs = [{"id": m["candidate_id"], "label": "defect" if m["contrast"] >= BAND_MIN_CONTRAST else "false_alarm",
             "confidence": 1.0, "reason": f"contrast at least {BAND_MIN_CONTRAST:g}"} for m in state["measurements"]]
    return {"labels": labs, "classify_status": "ok", "trace": [{"node": "classify", "attempts": 0}]}


CLASSIFIERS = {"llm": lambda s: llm_labels(s, guided=True),
               "llm_noguide": lambda s: llm_labels(s, guided=False),
               "band": band_labels}


# --- node 4-5: business rules and the decision, plain Python --------------------------------------
def severity(area_px: int) -> str:
    return next((name for limit, name in SEVERITY_BANDS if area_px < limit), "critical")


def apply_rules(measurements: list[dict], labels: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (kept, rejected). A candidate has to be labeled defect/uncertain AND pass every rule."""
    by_id = {m["candidate_id"]: m for m in measurements}
    kept, rejected = [], []
    for lab in labels:
        m = by_id[lab["id"]]
        if lab["label"] == "false_alarm":
            rejected.append({"id": lab["id"], "by": "classifier"})
        elif m["edge_distance_px"] == 0:  # R1: touches the image border, the ring is clipped, not measurable
            rejected.append({"id": lab["id"], "by": "R1 border"})
        elif m["area_px"] < MIN_AREA_PX:  # R2: too small to report
            rejected.append({"id": lab["id"], "by": "R2 min area"})
        else:
            uncertain = lab["label"] == "uncertain" or lab["confidence"] < CONF_MIN
            kept.append({"id": lab["id"], "severity": severity(m["area_px"]), "uncertain": uncertain,
                         "confidence": lab["confidence"], "contrast": m["contrast"], "area_px": m["area_px"]})
    return kept, rejected


def rules_node(state: PipelineState) -> dict:
    kept, rejected = apply_rules(state["measurements"], state["labels"])
    return {"kept": kept, "rejected": rejected,
            "trace": [{"node": "rules", "kept": len(kept), "rejected": len(rejected)}]}


def decide_node(state: PipelineState) -> dict:
    """Escalate whenever the pipeline itself is unsure. Never guess a pass or a fail."""
    kept = state.get("kept") or []
    if state["classify_status"] == "failed":
        verdict, reason = "escalate", "the classifier output was invalid after a retry"
    elif len(kept) > MAX_KEPT:
        verdict, reason = "escalate", f"{len(kept)} candidates survived (limit {MAX_KEPT}); the detector looks confused"
    elif any(k["uncertain"] for k in kept):
        verdict, reason = "escalate", "at least one surviving candidate is uncertain"
    elif kept:
        verdict, reason = "review", f"{len(kept)} confident defect(s) survived the rules"
    else:
        verdict, reason = "pass", "no candidate survived the classifier and the rules"
    return {"verdict": verdict, "reason": reason, "trace": [{"node": "decide", "verdict": verdict}]}


# --- node 6: report. Code owns the facts, the model only writes the prose --------------------------
def template_summary(state: PipelineState) -> str:
    n = len(state.get("candidates") or [])
    kept = state.get("kept") or []
    if state["verdict"] == "pass":
        return f"Pass. {n} candidate(s) examined, none kept. {state['reason']}."
    ids = ", ".join(f"#{k['id']} ({k['severity']})" for k in kept) or "none"
    return f"{state['verdict'].capitalize()}. {state['reason']}. Kept: {ids}."


def prose_problems(text: str, state: PipelineState) -> list[str]:
    """Cheap mechanical check: the prose may only cite candidate ids that exist in this image."""
    valid = {c["id"] for c in state.get("candidates") or []}
    cited = {int(m.group(1) or m.group(2))
             for m in re.finditer(r"(?:candidate|box|defect|id)s?\s*#?(\d+)|#(\d+)", text, flags=re.I)}
    bad = sorted(cited - valid)
    return [f"cites ids {bad} that do not exist (valid: {sorted(valid)})"] if bad else []


def report_node(state: PipelineState) -> dict:
    summary, source, calls, tokens = template_summary(state), "template", 0, 0
    if state["verdict"] != "pass":  # nothing to explain on a pass: do not pay a model to say "nothing"
        record = {"image": state["image"], "verdict": state["verdict"], "reason": state["reason"],
                  "kept": state["kept"], "rejected": state["rejected"]}
        resp = client.chat.completions.create(
            model=MODEL, temperature=0.2,
            messages=[{"role": "system", "content": SYSTEM_REPORT}, {"role": "user", "content": json.dumps(record)}])
        calls, tokens = 1, resp.usage.total_tokens
        text = (resp.choices[0].message.content or "").strip()
        if text and not prose_problems(text, state):
            summary, source = text, "llm"
        else:
            source = "template (llm prose rejected)"
    report = {"image": state["image"], "verdict": state["verdict"], "reason": state["reason"],
              "kept": state.get("kept") or [], "rejected": state.get("rejected") or [],
              "summary": summary, "summary_source": source}
    return {"report": report, "llm_calls": state["llm_calls"] + calls, "tokens": state["tokens"] + tokens,
            "trace": [{"node": "report", "source": source}]}


# --- edges: only two real branches in the whole pipeline -----------------------------------------
def after_detect(state: PipelineState) -> str:
    return "measure" if state["candidates"] else "report"


def after_classify(state: PipelineState) -> str:
    return "rules" if state["classify_status"] == "ok" else "decide"


def build_pipeline(classifier: str = "llm", **compile_kwargs):
    g = StateGraph(PipelineState)
    g.add_node("detect", detect_node)
    g.add_node("measure", measure_node)
    g.add_node("classify", CLASSIFIERS[classifier])
    g.add_node("rules", rules_node)
    g.add_node("decide", decide_node)
    g.add_node("report", report_node)
    g.add_edge(START, "detect")
    g.add_conditional_edges("detect", after_detect, ["measure", "report"])
    g.add_edge("measure", "classify")
    g.add_conditional_edges("classify", after_classify, ["rules", "decide"])
    g.add_edge("rules", "decide")
    g.add_edge("decide", "report")
    g.add_edge("report", END)
    return g.compile(**compile_kwargs)


def initial_state(image: str) -> PipelineState:
    return {"image": image, "candidates": [], "measurements": [], "labels": [], "classify_status": "skipped",
            "kept": [], "rejected": [], "verdict": "", "reason": "", "report": None, "trace": [],
            "llm_calls": 0, "tokens": 0}


def run_pipeline(image: str, graph) -> dict:
    t0 = time.time()
    final = graph.invoke(initial_state(image))
    kept_all = [k["id"] for k in final["kept"]]
    kept_confident = [k["id"] for k in final["kept"] if not k["uncertain"]]
    tp, fp, fn = score_kept(image, kept_all)              # headline: every box a human would be shown
    stp, sfp, sfn = score_kept(image, kept_confident)     # strict: only confident defects
    return {"image": image, "n_candidates": len(final["candidates"]), "kept": len(kept_all),
            "verdict": final["verdict"], "llm_calls": final["llm_calls"], "tokens": final["tokens"],
            "seconds": round(time.time() - t0, 1), "classify_status": final["classify_status"],
            "summary_source": final["report"]["summary_source"], "tp": tp, "fp": fp, "fn": fn,
            "strict": {"tp": stp, "fp": sfp, "fn": sfn}, "report": final["report"], "trace": final["trace"]}


def summarize(results: list[dict]) -> dict:
    n = len(results)
    tp, fp, fn = (sum(r[k] for r in results) for k in ("tp", "fp", "fn"))
    stp, sfp = (sum(r["strict"][k] for r in results) for k in ("tp", "fp"))
    ratio = lambda a, b: round(a / (a + b), 3) if a + b else None
    verdicts = {v: sum(r["verdict"] == v for r in results) for v in ("pass", "review", "escalate")}
    return {"n_images": n, "verdicts": verdicts, "kept": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "precision": ratio(tp, fp), "recall": ratio(tp, fn),
            "strict_precision": ratio(stp, sfp),
            "avg_llm_calls": round(sum(r["llm_calls"] for r in results) / n, 2),
            "avg_tokens": round(sum(r["tokens"] for r in results) / n),
            "avg_seconds": round(sum(r["seconds"] for r in results) / n, 1),
            "classify_failed": sum(r["classify_status"] == "failed" for r in results),
            "prose_rejected": sum("rejected" in r["summary_source"] for r in results),
            "prose_from_llm": sum(r["summary_source"] == "llm" for r in results)}


def run_variant(classifier: str, images: list[str]) -> list[dict]:
    graph = build_pipeline(classifier)
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(lambda im: run_pipeline(im, graph), images))


def main(repeats: int = 3):
    test = [f"test_{i:05d}" for i in range(N_PER_SPLIT)]  # the first N held-out images, as in Part 5
    out = {"model": MODEL, "n_test_images": len(test), "variants": {}}
    for classifier, n_runs in (("band", 1), ("llm", repeats), ("llm_noguide", repeats)):
        runs = []
        for i in range(n_runs):
            res = run_variant(classifier, test)
            runs.append({"summary": summarize(res), "results": res})
            s = runs[-1]["summary"]
            print(f"{classifier} run {i + 1}: kept {s['kept']} tp {s['tp']} fp {s['fp']} "
                  f"P {s['precision']} R {s['recall']} verdicts {s['verdicts']} tokens {s['avg_tokens']}", flush=True)
        out["variants"][classifier] = runs
    (OUT_DIR / "pipeline_results.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / "pipeline_summary.json").write_text(json.dumps(
        {"model": MODEL, "variants": {k: [r["summary"] for r in v] for k, v in out["variants"].items()}}, indent=2))


if __name__ == "__main__":
    main()
