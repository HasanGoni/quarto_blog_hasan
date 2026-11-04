"""Part 10: offline checks for sub_agents.py. A scripted fake model replaces the server, so every path runs in seconds.

    uv run python test_sub_paths.py
"""
import json
import types

import sub_agents as sa
from inspection_tools import find_candidates, measure_candidate

IMG = "test_00002"          # 3 candidates
N = len(find_candidates(IMG)["candidates"])
ok = 0


def check(cond, name):
    global ok
    assert cond, f"FAILED: {name}"
    ok += 1
    print(f"ok  {name}")


class Msg:
    def __init__(self, content=None, calls=()):
        self.content = content
        self.tool_calls = [types.SimpleNamespace(id=f"c{i}", function=types.SimpleNamespace(name=n, arguments=a if isinstance(a, str) else json.dumps(a)))
                           for i, (n, a) in enumerate(calls)] or None

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in self.tool_calls]
        return {k: v for k, v in d.items() if v is not None}


class Fake:
    """script(messages, tools) -> Msg. Records every call, so a test can assert what a context was shown."""

    def __init__(self, script):
        self.script, self.seen = script, []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.seen.append(kw)
        msg = self.script(kw["messages"], kw.get("tools"), kw.get("temperature"))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                     usage=types.SimpleNamespace(prompt_tokens=100 + 10 * len(kw["messages"]), completion_tokens=7))


def use(script):
    f = Fake(script)
    sa.CLIENT = f
    return f


def judged_id(messages):
    return json.loads(messages[1]["content"].split("\n", 1)[1])["candidate_id"]


# ---- agent_loop -------------------------------------------------------------------------------------------------
def s_loop(messages, tools, t):
    n = sum(m["role"] == "assistant" for m in messages)
    return [Msg("thinking out loud"),                                   # no tool call: nudged
            Msg(calls=[("nope", {})]),                                  # unknown tool
            Msg(calls=[("find_candidates", "{not json")]),              # malformed arguments
            Msg(calls=[("measure_candidate", {"image": IMG, "candidate_id": 99})]),   # tool error comes back as a result
            Msg(calls=[("submit_report", {"kept_ids": [0], "reason": "x"})])][n]


m = sa.Meter()
f = use(s_loop)
r = sa.agent_loop("sys", "go", sa.TOOL_SPECS + [sa.SUBMIT_REPORT], sa.TOOL_FUNCS, "submit_report", m, "single")
results = [json.loads(x["content"]) for x in r["messages"] if x["role"] == "tool"]
check(r["submitted"]["kept_ids"] == [0], "loop: finishes on submit")
check(any("unknown tool" in str(x) for x in results) and any("not valid JSON" in str(x) for x in results) and any("out of range" in str(x) for x in results),
      "loop: unknown tool, bad JSON and tool error all go back to the model, none crash it")
check(any("Use a tool" in x.get("content", "") for x in r["messages"] if x["role"] == "user"), "loop: a talking model is nudged")
check(m.calls == 5 and m.by_role["single"]["calls"] == 5, "meter: counts every call, by role")

use(lambda msgs, tools, t: Msg("still talking"))
r = sa.agent_loop("sys", "go", [], {}, "submit_report", sa.Meter(), "x", max_steps=3)
check(r["submitted"] is None and len(r["messages"]) == 2 + 3 * 2, "loop: stops at the step limit with nothing submitted")

# ---- rules_filter ------------------------------------------------------------------------------------------------
tiny = next((i for i in range(N) if measure_candidate(IMG, i)["area_px"] < sa.MIN_AREA_PX), None)
check(sa.rules_filter(IMG, [0, 0, 1000]) in ([0], []), "rules: duplicates and out-of-range ids are dropped")

check(sa.rules_filter(IMG, ["0", True, 1.5, None, 0]) in ([0], []), "rules: ids that are not integers (strings, booleans, floats) are dropped, not guessed")
m = sa.Meter()
r = sa.finish(IMG, "x", ["0", 0], m, 0.0)
check(r["bad_ids"] == ["0"], "finish: the dropped ids are recorded")

# ---- batched -----------------------------------------------------------------------------------------------------
def labels(keep):
    return json.dumps({"labels": [{"id": i, "label": "defect" if i in keep else "false_alarm", "confidence": 0.9, "reason": "r"} for i in range(N)]})


use(lambda msgs, tools, t: Msg(labels({0})))
r = sa.arm_batched(IMG)
check(r["calls"] == 1 and r["judge_failed"] == 0 and r["kept"] == sa.rules_filter(IMG, [0]), "batched: one call, kept = defects after rules")

seq = iter(["```json\nnot json```", labels({0})])
use(lambda msgs, tools, t: Msg(next(seq)))
r = sa.arm_batched(IMG)
check(r["calls"] == 2 and r["judge_failed"] == 0, "batched: a malformed answer is retried with the error")

use(lambda msgs, tools, t: Msg(json.dumps({"labels": [{"id": 0, "label": "defect", "confidence": 0.9}]})))
r = sa.arm_batched(IMG)
check(r["judge_failed"] == 1 and r["kept"] == [], "batched: valid JSON that skips ids is refused, and one failure drops the WHOLE image")

# ---- fanout ------------------------------------------------------------------------------------------------------
def one(label):
    return json.dumps({"label": label, "confidence": 0.8, "reason": "r"})


f = use(lambda msgs, tools, t: Msg(one("defect")))
r = sa.arm_fanout(IMG)
check(r["calls"] == N and len(f.seen) == N, "fanout: one call per box")
check(all(len(c["messages"]) == 2 for c in f.seen), "fanout: every context is just system + one box")
ids_seen = sorted(judged_id(c["messages"]) for c in f.seen)
check(ids_seen == list(range(N)), "fanout: each sub-agent saw exactly its own box, once")
check(all(sum(str(i) in json.dumps(json.loads(c["messages"][1]["content"].split("\n", 1)[1])) for i in [judged_id(c["messages"])]) for c in f.seen), "fanout: the box is in its own brief")

use(lambda msgs, tools, t: Msg("garbage") if judged_id(msgs) == 1 else Msg(one("false_alarm")))
r = sa.arm_fanout(IMG)
check(r["judge_failed"] == 1 and ((1 in r["kept"]) == bool(sa.rules_filter(IMG, [1]))),
      "fanout: a failed sub-agent keeps its box (never silently drops it) and does not touch the other boxes")
check(r["calls"] == (N - 1) + 2, "fanout: the failed box was retried once")

# ---- panel3 ------------------------------------------------------------------------------------------------------
votes = {0: iter(["defect", "defect", "false_alarm"]), 1: iter(["false_alarm", "false_alarm", "defect"]), 2: iter(["false_alarm"] * 3)}
f = use(lambda msgs, tools, t: Msg(one(next(votes[judged_id(msgs)]))))
r = sa.arm_panel3(IMG)
check(r["calls"] == 3 * N and r["judge_units"] == 3 * N, "panel3: three sub-agents per box")
check(set(r["kept"]) <= {0}, "panel3: box 0 wins 2 of 3, box 1 loses 1 of 3")
check(r["split_votes"] == 2, "panel3: counts boxes where the panel disagreed")
check(all(c["temperature"] == sa.PANEL_TEMPERATURE for c in f.seen), "panel3: copies run warm (else they would be identical)")

# ---- single ------------------------------------------------------------------------------------------------------
def s_single(messages, tools, t):
    n = sum(m["role"] == "assistant" for m in messages)
    return [Msg(calls=[("find_candidates", {"image": IMG})]), Msg(calls=[("submit_report", {"kept_ids": [0, 0], "reason": "r"})])][n]


use(s_single)
r = sa.arm_single(IMG)
check(r["finished"] and r["calls"] == 2 and r["tools"] == ["find_candidates", "submit_report"], "single: one agent, tools then submit")

use(lambda msgs, tools, t: Msg("no"))
r = sa.arm_single(IMG)
check(not r["finished"] and r["kept"] == [] and r["calls"] == sa.MAX_STEPS, "single: never submitting counts as unfinished with nothing kept")

# ---- supervisor and the hand-off -----------------------------------------------------------------------------------
true = [measure_candidate(IMG, i) for i in range(N)]


def make_supervisor(brief_fmt):
    def script(messages, tools, t):
        sysm = messages[0]["content"]
        n = sum(m["role"] == "assistant" for m in messages)
        if sysm.startswith("You supervise"):
            return [Msg(calls=[("inspect_stage", {"image": IMG})]),
                    Msg(calls=[("judge_stage", {"brief": " ".join(brief_fmt(m) for m in true)})]),
                    Msg(calls=[("submit_report", {"kept_ids": [0], "reason": "r"})])][n]
        if sysm.startswith("You are the measuring"):
            return [Msg(calls=[("find_candidates", {"image": IMG})]), Msg(calls=[("submit_findings", {"note": "measured"})])][n]
        return Msg("0 defect 0.9")        # judge
    return script


f = use(make_supervisor(lambda m: f"id {m['candidate_id']}: contrast {m['contrast']}"))
r = sa.arm_supervisor(IMG)
check(r["handoff"] == {"candidates": N, "carried": N}, "supervisor: a faithful brief carries every measurement")
roles = r["by_role"]
check(set(roles) == {"supervisor", "measurer", "judge"} and roles["judge"]["calls"] == 1, "supervisor: tokens are counted per role")
judge_ctx = next(c for c in f.seen if c["messages"][0]["content"].startswith("You are the judging"))["messages"]
check(len(judge_ctx) == 2 and "image" not in judge_ctx[1]["content"].lower(), "supervisor: the judge sees only the brief, nothing else")

f = use(make_supervisor(lambda m: f"id {m['candidate_id']}: contrast {round(m['contrast'])}"))
r = sa.arm_supervisor(IMG)
check(r["handoff"]["carried"] < N or all(abs(m["contrast"] - round(m["contrast"])) <= 0.06 for m in true),
      "supervisor: rounding contrast to a whole number loses the measurement")

f = use(make_supervisor(lambda m: "something happened"))
check(sa.arm_supervisor(IMG)["handoff"]["carried"] == 0, "supervisor: a vague brief carries nothing")
check(sa.fidelity(IMG, "")["carried"] == 0, "fidelity: empty text carries nothing")

print(f"\nALL SUB-AGENT PATHS OK ({ok} checks)")
