"""Guard test: does Deep Agents recover if a repeated identical tool call is cut off?

Adds one middleware to the same agent as ctx_deepagents.py: when the model has asked for the SAME tool call(s) three times in
the run (not necessarily in a row: the first version of this guard required three in a row, never fired because the loop
alternated limit=1 / limit=2, and its runs are kept as out/old_ctx_da_guard_consecutive_rule.json), the run ends and the last tool result is returned unanswered (no hint about the answer is given). Only the T1 runs that
looped are re-run (3 runs: 2 `da`, 1 `da+chunk`), because those were the clean "answer already in context" loops.

    uv run python ctx_da_guard.py        # writes out/ctx_da_guard_runs.json
"""
import json
import time

from langchain.agents.middleware import before_model
from langchain_core.messages import AIMessage

import ctx_deepagents as d
from ctx_eval import build_tasks, score
from ctx_harness import parse_answer

OUT = d.OUT.with_name("ctx_da_guard_runs.json")
REPEAT = 3


def _sig(m) -> str:
    return json.dumps([(c["name"], c["args"]) for c in (m.tool_calls or [])], sort_keys=True)


@before_model(can_jump_to=["end"])
def stop_repeats(state, runtime):
    ai = [m for m in state["messages"] if m.type == "ai" and m.tool_calls]
    sigs = [_sig(m) for m in ai]
    if sigs and sigs.count(sigs[-1]) >= REPEAT:
        return {"messages": [AIMessage(content="STOPPED: same tool call repeated")], "jump_to": "end"}
    return None


_orig = d.build_agent


def build_with_guard(arm, root):
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from langchain.agents.middleware import ModelCallLimitMiddleware
    return create_deep_agent(
        model=d.make_llm(), tools=[d.search_reports, d.similar_cases], system_prompt=d.SYSTEM,
        subagents=[d.CHUNK_ANALYST] if arm == "da+chunk" else None,
        backend=FilesystemBackend(root_dir=root, virtual_mode=True),
        middleware=[ModelCallLimitMiddleware(run_limit=d.CALL_LIMIT, exit_behavior="end"), stop_repeats])


d.build_agent = build_with_guard

if __name__ == "__main__":
    tasks = {t["q"]: t for t in build_tasks()}
    old = json.loads(d.OUT.read_text())
    todo = [r for r in old if r["task"] == "T1" and "Recursion" in r["status"]]
    runs = []
    for r in todo:
        t = tasks[r["question"]]
        m = d.run(t["q"], r["strategy"])
        m["task"], m["parsed"] = "T1", parse_answer(m["answer"] if isinstance(m["answer"], str) else None)
        m["correct"], m["gold"] = score(t, m["parsed"]), t.get("gold", t.get("gold_ids"))
        runs.append(m)
        OUT.write_text(json.dumps(runs, indent=1))
        print(r["strategy"], t["q"][60:100], "| before: loop", r["model_calls_all"], "calls |", "after:", m["status"],
              "correct=", m["correct"], "calls=", m["model_calls_all"], "tokens=", m["total_prompt_tokens"],
              "answer=", str(m["answer"])[:90], flush=True)
