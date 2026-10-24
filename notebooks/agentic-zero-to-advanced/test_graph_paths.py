"""Part 6: prove every edge of the graph works, with a scripted fake model (no vLLM needed).

The real run never took the `nudge` edge (Qwen3-Coder always called a tool), so a real run proves nothing about it.
Here the 'model' is a script: each call returns the next scripted message.
"""
import json
from types import SimpleNamespace

import graph_agent as ga


def msg(content=None, calls=()):
    tool_calls = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                  for i, (n, a) in enumerate(calls)]
    d = {"role": "assistant", "content": content, **({"tool_calls": tool_calls} if tool_calls else {})}
    return SimpleNamespace(model_dump=lambda exclude_none=True: d)


def run(script, image="test_00020"):
    it = iter(script)

    def fake_create(**kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=next(it))], usage=SimpleNamespace(total_tokens=10))

    ga.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    final = ga.GRAPH.invoke(ga.initial_state(image), {"recursion_limit": 4 * ga.MAX_STEPS})
    path = [("nudge" if not s["calls"] else s["calls"][0]["tool"]) for s in final["trace"]]
    return final, path


img = "test_00020"
# 1. happy path: find, submit -> END via the tools node
final, path = run([msg(calls=[("find_candidates", {"image": img})]),
                   msg(calls=[("submit_report", {"kept_ids": [], "verdict": "pass", "reason": "nothing"})])])
print("happy path:      ", path, "| steps", final["step"], "| report:", bool(final["report"]))

# 2. the model talks once, gets nudged, then submits
final, path = run([msg("I think it is fine."),
                   msg(calls=[("submit_report", {"kept_ids": [], "verdict": "pass", "reason": "ok"})])])
print("talk then submit:", path, "| steps", final["step"], "| report:", bool(final["report"]),
      "| nudge text in messages:", any(m.get("content", "").startswith("Use a tool") for m in final["messages"]))

# 3. the model never submits: the step limit must stop the graph
final, path = run([msg(calls=[("find_candidates", {"image": img})])] * ga.MAX_STEPS)
print("never submits:   ", len(path), "steps | stopped at step", final["step"], "| report:", final["report"])

# 4. the model only ever talks: nudge edge must also respect the step limit
final, path = run([msg("hmm")] * ga.MAX_STEPS)
print("only talks:      ", path.count("nudge"), "nudges | stopped at step", final["step"], "| report:", final["report"])
