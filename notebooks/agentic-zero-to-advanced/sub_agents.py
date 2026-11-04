"""Part 10: when one agent is not enough. Sub-agents, measured against one agent on the same 200 held-out images.

Two separate questions, because "use sub-agents" answers both and they are not the same thing:

  Q1  An agent runs the whole inspection and decides the order itself (Part 5/6).
      Does it help to split it into a supervisor plus two specialists (a measurer and a judge)?
          single       one agent, all tools                                (Part 5/6 prompt, plus the Part 7 guidance)
          supervisor   supervisor agent + `inspect_stage` + `judge_stage`  (each a sub-agent with its own context)

  Q2  Code owns the order (Part 7), the model only judges the boxes. How many model contexts should the judging use?
          batched      one call with all boxes of the image               (Part 7's `llm` classifier)
          fanout       one sub-agent per box, each sees ONLY that box, run in parallel, code collects the answers
          panel3       three independent sub-agents per box, majority vote (a second and third opinion)

Same model, same temperature, same guidance text, same business rules (R1 border, R2 minimum area) applied by code to
every arm, same scoring (`score_kept` against the exact void boxes). Only the number of contexts and who talks to whom changes.
"""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from pydantic import ValidationError

from inspection_tools import TOOL_FUNCS, TOOL_SPECS, find_candidates, measure_candidate, score_kept
from pipeline_graph import GUIDANCE, MIN_AREA_PX, Label, Labels
from structured_output_demo import load_dotenv, strip_fences

load_dotenv()
MODEL = os.environ.get("PART10_MODEL", "qwen3.8-27b")   # not AGENTIC_ZERO_MODEL: that variable still names the Part 5-7 model
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}   # same switch as Part 14
MAX_OUT = 2000
MAX_STEPS = 12
TEMPERATURE = 0.0

CLIENT = None   # tests replace this with a scripted fake


def client():
    global CLIENT
    if CLIENT is None:
        CLIENT = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
    return CLIENT


# --- counting what every context costs -------------------------------------------------------------------------------
class Meter:
    """One per image. Counts every model call from every agent, and the biggest single context any agent held."""

    def __init__(self):
        self.calls = self.prompt = self.completion = self.peak = 0
        self.by_role: dict[str, dict] = {}
        self._lock = threading.Lock()

    def add(self, usage, role: str):
        p, c = usage.prompt_tokens, usage.completion_tokens
        with self._lock:
            self.calls, self.prompt, self.completion = self.calls + 1, self.prompt + p, self.completion + c
            self.peak = max(self.peak, p)
            r = self.by_role.setdefault(role, {"calls": 0, "prompt": 0, "completion": 0, "peak": 0})
            r["calls"] += 1
            r["prompt"] += p
            r["completion"] += c
            r["peak"] = max(r["peak"], p)

    def as_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt, "completion_tokens": self.completion,
                "peak_prompt": self.peak, "by_role": self.by_role}


def chat(messages, meter: Meter, role: str, tools=None, temperature: float = TEMPERATURE):
    kw = dict(model=MODEL, messages=messages, temperature=temperature, max_tokens=MAX_OUT, extra_body=NO_THINK)
    if tools:
        kw["tools"] = tools
    resp = client().chat.completions.create(**kw)
    meter.add(resp.usage, role)
    return resp.choices[0].message


# --- the one loop every agent in this part uses ----------------------------------------------------------------------
def agent_loop(system: str, user: str, tools: list, funcs: dict, submit: str, meter: Meter, role: str,
               max_steps: int = MAX_STEPS) -> dict:
    """Ask, run the tool the model asked for, put the result back, ask again. Stops when `submit` is called.
    This is the Part 5 loop with the model call moved behind `chat` so every call is counted."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    submitted, calls_made = None, []
    for _ in range(max_steps):
        msg = chat(messages, meter, role, tools)
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            messages.append({"role": "user", "content": f"Use a tool, or call {submit} if you are done."})
            continue
        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args, result = None, {"error": "arguments were not valid JSON"}
            if args is not None:
                if name == submit:
                    submitted, result = args, {"ok": True}
                elif name in funcs:
                    try:
                        result = funcs[name](**args)
                    except Exception as exc:   # tool errors go back to the model (Part 5)
                        result = {"error": str(exc)}
                else:
                    result = {"error": f"unknown tool {name}"}
            calls_made.append(name)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
        if submitted is not None:
            break
    return {"submitted": submitted, "tool_calls": calls_made, "messages": messages}


# --- code-owned rules, the same for every arm --------------------------------------------------------------------------
def valid_id(i) -> bool:
    return isinstance(i, int) and not isinstance(i, bool)


def rules_filter(image: str, ids: list) -> list[int]:
    """R1 (touches the border) and R2 (smaller than MIN_AREA_PX) from Part 7, applied by code to whatever the agents kept.
    An id that is not an integer is dropped (never guessed): `finish` counts them. The first full run crashed on a string id."""
    out = []
    for i in dict.fromkeys(i for i in ids if valid_id(i)):
        m = measure_candidate(image, i)
        if "error" in m or m["edge_distance_px"] == 0 or m["area_px"] < MIN_AREA_PX:
            continue
        out.append(i)
    return out


def finish(image: str, arm: str, kept_ids: list, meter: Meter, t0: float, **extra) -> dict:
    kept = rules_filter(image, kept_ids)
    extra["bad_ids"] = [i for i in kept_ids if not valid_id(i)]
    tp, fp, fn = score_kept(image, kept)
    n_cand = len(find_candidates(image)["candidates"])   # for scoring only
    return {"arm": arm, "image": image, "n_candidates": n_cand, "kept": kept, "tp": tp, "fp": fp, "fn": fn,
            "seconds": round(time.time() - t0, 1), **meter.as_dict(), **extra}


# =====================================================================================================================
# Q1: an agent runs the inspection
# =====================================================================================================================
SYSTEM_SINGLE = f"""You inspect X-ray images of metal parts for voids (small defects).
Tools: find_candidates (noisy classical detector, most boxes are false alarms),
measure_candidate (look closer at one box), submit_report (finish).
Work like this: call find_candidates first. If there are no candidates, submit a pass right away.
Otherwise measure the candidates that look worth checking, decide which to keep, then submit_report.
Say in one short sentence what you are about to do before each tool call.

{GUIDANCE}"""

SUBMIT_REPORT = {"type": "function", "function": {
    "name": "submit_report", "description": "Finish the inspection. Call this exactly once, when you are done.",
    "parameters": {"type": "object", "properties": {
        "kept_ids": {"type": "array", "items": {"type": "integer"}, "description": "candidate ids you believe are real defects"},
        "reason": {"type": "string"}}, "required": ["kept_ids", "reason"]}}}


def arm_single(image: str) -> dict:
    t0, meter = time.time(), Meter()
    r = agent_loop(SYSTEM_SINGLE, f"Inspect image {image}.", TOOL_SPECS + [SUBMIT_REPORT], TOOL_FUNCS,
                   "submit_report", meter, "single")
    sub = r["submitted"]
    return finish(image, "single", sub["kept_ids"] if sub else [], meter, t0, finished=sub is not None,
                  tools=r["tool_calls"])


# --- supervisor + two specialists ------------------------------------------------------------------------------------
SYSTEM_MEASURER = """You are the measuring specialist for X-ray inspection. You get one image id.
Call find_candidates, then measure_candidate for EVERY candidate. Then call submit_findings with a short plain-text note
for your supervisor: what you found and the numbers you measured. You only measure; you do not decide what is a defect."""

SUBMIT_FINDINGS = {"type": "function", "function": {
    "name": "submit_findings", "description": "Hand your findings to the supervisor. Call this exactly once.",
    "parameters": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]}}}

SYSTEM_JUDGE = f"""You are the judging specialist for X-ray inspection. You get a note with measurements of candidate boxes
from a noisy classical detector. Most candidates are false alarms. {GUIDANCE}
Reply with one short line per candidate id: its id, "defect" or "false_alarm", and a confidence from 0 to 1.
Use only ids and numbers that appear in the note. Reply in plain text, no tools."""

SYSTEM_SUPERVISOR = """You supervise the inspection of one X-ray image of a metal part. You cannot measure or judge yourself.
You have two specialists, who each work in their own separate context and see ONLY what you pass them:
  inspect_stage(image)  measures every candidate box and returns a note
  judge_stage(brief)    reads a brief and says which boxes are defects. The judge cannot see the image or the measurer's note
                        unless you copy the numbers into the brief.
Call inspect_stage first. If it found no candidates, submit_report with an empty list. Otherwise call judge_stage with a brief that
contains every candidate id and every measurement, then submit_report with the ids the judge called defects."""

SUPERVISOR_TOOLS = [
    {"type": "function", "function": {"name": "inspect_stage", "description": "Measuring specialist: measures every candidate box of an image and returns a note.",
        "parameters": {"type": "object", "properties": {"image": {"type": "string"}}, "required": ["image"]}}},
    {"type": "function", "function": {"name": "judge_stage", "description": "Judging specialist: decides which boxes of a brief are defects.",
        "parameters": {"type": "object", "properties": {"brief": {"type": "string"}}, "required": ["brief"]}}},
    SUBMIT_REPORT,
]

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def fidelity(image: str, text: str) -> dict:
    """How much of the true measurement survived a hand-off in plain text? A box counts as 'carried' when its contrast
    value (within 0.06, so rounding to one decimal is forgiven but not to a whole number) and its id both appear."""
    cands = find_candidates(image)["candidates"]
    nums = [float(x) for x in NUM.findall(text)]
    carried = 0
    for c in cands:
        true = measure_candidate(image, c["id"])["contrast"]
        if any(abs(n - true) <= 0.06 for n in nums) and re.search(rf"(?<![\d.]){c['id']}(?![\d])", text):
            carried += 1
    return {"candidates": len(cands), "carried": carried}


def arm_supervisor(image: str) -> dict:
    t0, meter = time.time(), Meter()
    seen = {"notes": [], "briefs": [], "judge_out": []}

    def inspect_stage(image: str) -> dict:
        r = agent_loop(SYSTEM_MEASURER, f"Image {image}.", TOOL_SPECS + [SUBMIT_FINDINGS], TOOL_FUNCS,
                       "submit_findings", meter, "measurer")
        note = (r["submitted"] or {}).get("note", "(the measuring specialist did not finish)")
        seen["notes"].append(note)
        return {"note": note}

    def judge_stage(brief: str) -> dict:
        seen["briefs"].append(brief)
        msg = chat([{"role": "system", "content": SYSTEM_JUDGE}, {"role": "user", "content": brief}], meter, "judge")
        out = (msg.content or "").strip()
        seen["judge_out"].append(out)
        return {"judgement": out}

    r = agent_loop(SYSTEM_SUPERVISOR, f"Inspect image {image}.", SUPERVISOR_TOOLS,
                   {"inspect_stage": inspect_stage, "judge_stage": judge_stage}, "submit_report", meter, "supervisor")
    sub = r["submitted"]
    brief = seen["briefs"][-1] if seen["briefs"] else ""
    return finish(image, "supervisor", sub["kept_ids"] if sub else [], meter, t0, finished=sub is not None,
                  tools=r["tool_calls"], handoff=fidelity(image, brief) if brief else None,
                  brief=brief[:1500], note=(seen["notes"][-1] if seen["notes"] else "")[:1500],
                  judge_out=(seen["judge_out"][-1] if seen["judge_out"] else "")[:800])


# =====================================================================================================================
# Q2: code owns the order, how many contexts does the judging use?
# =====================================================================================================================
SYSTEM_BATCH = f"""You classify candidate boxes from a noisy classical detector on X-ray images of
metal parts. You get one measurement record per candidate and no image. Most candidates are false alarms.
For every candidate choose a label: "defect", "false_alarm" or "uncertain", and a confidence from 0 to 1.
{GUIDANCE}
Return ONLY a JSON object, no markdown fences:
{{"labels": [{{"id": 0, "label": "defect", "confidence": 0.8, "reason": "one short sentence"}}]}}
Exactly one entry per candidate id, using the ids you were given."""

SYSTEM_ONE = f"""You judge ONE candidate box from a noisy classical detector on an X-ray image of a metal part. You get its
measurement record and no image. Most candidates are false alarms.
Choose a label: "defect", "false_alarm" or "uncertain", and a confidence from 0 to 1.
{GUIDANCE}
Return ONLY a JSON object, no markdown fences:
{{"label": "defect", "confidence": 0.8, "reason": "one short sentence"}}"""


def measure_all(image: str) -> list[dict]:
    return [measure_candidate(image, c["id"]) for c in find_candidates(image)["candidates"]]


def judge_batch(ms: list[dict], meter: Meter) -> list[dict] | None:
    """Part 7's classifier: one call, validate, retry once with the exact error. None = gave up."""
    ids = sorted(m["candidate_id"] for m in ms)
    messages = [{"role": "system", "content": SYSTEM_BATCH}, {"role": "user", "content": "Candidates:\n" + json.dumps(ms)}]
    for _ in range(2):
        raw = chat(messages, meter, "judge").content or ""
        try:
            parsed = Labels(**json.loads(strip_fences(raw)))
            if sorted(x.id for x in parsed.labels) != ids:
                raise ValueError(f"expected exactly one label for each id in {ids}")
            return [x.model_dump() for x in parsed.labels]
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"That failed validation:\n{type(exc).__name__}: {exc}\nReturn ONLY the corrected JSON."}]
    return None


def judge_one(m: dict, meter: Meter, temperature: float = TEMPERATURE) -> dict | None:
    """A sub-agent whose whole world is one box. None = it failed twice (the other boxes are not affected)."""
    messages = [{"role": "system", "content": SYSTEM_ONE}, {"role": "user", "content": "Candidate:\n" + json.dumps(m)}]
    for _ in range(2):
        raw = chat(messages, meter, "judge", temperature=temperature).content or ""
        try:
            body = json.loads(strip_fences(raw))
            body["id"] = m["candidate_id"]          # the id is ours to set, not the model's
            return Label(**body).model_dump()
        except (json.JSONDecodeError, ValidationError, TypeError, AttributeError) as exc:
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"That failed validation:\n{type(exc).__name__}: {exc}\nReturn ONLY the corrected JSON."}]
    return None


def _keep(lab: dict) -> bool:
    return lab["label"] != "false_alarm"      # "uncertain" is kept: a human still looks at it (Part 7, Part 9)


def arm_batched(image: str) -> dict:
    t0, meter = time.time(), Meter()
    ms = measure_all(image)
    labs = judge_batch(ms, meter) if ms else []
    failed = labs is None
    return finish(image, "batched", [] if failed else [x["id"] for x in labs if _keep(x)], meter, t0,
                  judge_failed=int(failed), judge_units=1 if ms else 0)


def arm_fanout(image: str, workers: int = 8) -> dict:
    t0, meter = time.time(), Meter()
    ms = measure_all(image)
    with ThreadPoolExecutor(max(1, min(workers, len(ms)))) as pool:
        labs = list(pool.map(lambda m: judge_one(m, meter), ms))
    # a box whose sub-agent failed is kept as uncertain: a failure must never silently drop a defect
    kept = [m["candidate_id"] for m, lab in zip(ms, labs) if lab is None or _keep(lab)]
    return finish(image, "fanout", kept, meter, t0, judge_failed=sum(lab is None for lab in labs), judge_units=len(ms))


PANEL = 3
PANEL_TEMPERATURE = 0.7     # three copies at temperature 0 would say the same thing three times


def arm_panel3(image: str, workers: int = 8) -> dict:
    t0, meter = time.time(), Meter()
    ms = measure_all(image)
    jobs = [m for m in ms for _ in range(PANEL)]
    with ThreadPoolExecutor(max(1, min(workers, len(jobs)))) as pool:
        labs = list(pool.map(lambda m: judge_one(m, meter, PANEL_TEMPERATURE), jobs))
    kept, split = [], 0
    for i, m in enumerate(ms):
        votes = labs[i * PANEL:(i + 1) * PANEL]
        yes = sum(1 for v in votes if v is None or _keep(v))
        split += 0 < yes < PANEL
        if yes * 2 > PANEL:
            kept.append(m["candidate_id"])
    return finish(image, "panel3", kept, meter, t0, judge_failed=sum(lab is None for lab in labs),
                  judge_units=len(jobs), split_votes=split)


ARMS = {"single": arm_single, "supervisor": arm_supervisor, "batched": arm_batched, "fanout": arm_fanout, "panel3": arm_panel3}
