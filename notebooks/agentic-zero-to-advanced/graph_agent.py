"""Part 6: the Part 5 agent loop, rebuilt as a LangGraph graph.

Same model, same tools, same system prompt, same MAX_STEPS, same scoring as agent_loop.py. Only the
control flow changes: the `for` loop becomes three nodes (agent, tools, nudge) and the `if`s
become edges. Nothing here changes what the agent can do, only how the flow is written down.
"""
import json
import operator
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_loop import MAX_STEPS, MODEL, OUT_DIR, SUBMIT_SPEC, SYSTEM, client
from inspection_tools import N_PER_SPLIT, TOOL_FUNCS, TOOL_SPECS, find_candidates, score_kept


class AgentState(TypedDict):
    """Everything the graph remembers. Each key is one box on the whiteboard."""
    messages: Annotated[list, operator.add]  # reducer: new messages are appended, never replaced
    trace: Annotated[list, operator.add]
    step: int
    tokens: int
    report: dict | None


def agent_node(state: AgentState) -> dict:
    """Ask the model what to do next. Same call as the first line inside Part 5's loop."""
    resp = client.chat.completions.create(
        model=MODEL, messages=state["messages"], tools=TOOL_SPECS + [SUBMIT_SPEC], temperature=0.2)
    msg = resp.choices[0].message
    return {"messages": [msg.model_dump(exclude_none=True)], "step": state["step"] + 1,
            "tokens": state["tokens"] + resp.usage.total_tokens}


def tools_node(state: AgentState) -> dict:
    """Run every tool the model asked for and put the results back as messages."""
    last, results, calls, report = state["messages"][-1], [], [], None
    for call in last["tool_calls"]:
        args = json.loads(call["function"]["arguments"] or "{}")
        name = call["function"]["name"]
        if name == "submit_report":
            report, result = args, {"ok": True}
        elif name in TOOL_FUNCS:
            try:
                result = TOOL_FUNCS[name](**args)
            except Exception as exc:  # tool errors go back to the model, they don't crash the graph
                result = {"error": str(exc)}
        else:
            result = {"error": f"unknown tool {name}"}
        calls.append({"tool": name, "args": args, "result": result})
        results.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
    return {"messages": results, "report": report,
            "trace": [{"step": state["step"], "says": last.get("content"), "calls": calls}]}


def nudge_node(state: AgentState) -> dict:
    """The model talked instead of acting. Remind it, once per step, same wording as Part 5."""
    return {"messages": [{"role": "user", "content": "Use a tool, or call submit_report if you are done."}],
            "trace": [{"step": state["step"], "says": state["messages"][-1].get("content"), "calls": []}]}


def after_agent(state: AgentState) -> str:
    """Edge 1: did the model ask for a tool? Yes -> tools. No -> nudge (or stop at the step limit)."""
    if state["messages"][-1].get("tool_calls"):
        return "tools"
    return "nudge" if state["step"] < MAX_STEPS else END


def after_tools(state: AgentState) -> str:
    """Edge 2: submitted the report, or out of steps? Stop. Otherwise go round again."""
    return END if state["report"] or state["step"] >= MAX_STEPS else "agent"


def build_graph(**compile_kwargs):
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("nudge", nudge_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, ["tools", "nudge", END])
    g.add_conditional_edges("tools", after_tools, ["agent", END])
    g.add_conditional_edges("nudge", lambda s: END if s["step"] >= MAX_STEPS else "agent", ["agent", END])
    return g.compile(**compile_kwargs)


GRAPH = build_graph()


def initial_state(image: str) -> AgentState:
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Inspect image {image}."}],
            "trace": [], "step": 0, "tokens": 0, "report": None}


def run_graph_agent(image: str) -> dict:
    t0 = time.time()
    final = GRAPH.invoke(initial_state(image), {"recursion_limit": 4 * MAX_STEPS})
    report = final["report"]
    n_cand = len(find_candidates(image)["candidates"])  # for scoring only, not shown to the agent
    kept_ids = report["kept_ids"] if report else []
    tp, fp, fn = score_kept(image, kept_ids)
    return {"image": image, "split": image.split("_")[0], "finished": report is not None, "steps": len(final["trace"]),
            "tool_calls": sum(len(s["calls"]) for s in final["trace"]), "tokens": final["tokens"],
            "seconds": round(time.time() - t0, 1), "n_candidates": n_cand, "kept": len(kept_ids),
            "verdict": report["verdict"] if report else None, "tp": tp, "fp": fp, "fn": fn,
            "trace": final["trace"]}


def summarize(results: list[dict]) -> dict:
    n = len(results)
    tp, fp, fn = (sum(r[k] for r in results) for k in ("tp", "fp", "fn"))
    prec = lambda t, f: round(t / (t + f), 3) if t + f else None
    return {"n_images": n, "finished": sum(r["finished"] for r in results),
            "avg_steps": round(sum(r["steps"] for r in results) / n, 2),
            "avg_tool_calls": round(sum(r["tool_calls"] for r in results) / n, 2),
            "avg_tokens": round(sum(r["tokens"] for r in results) / n),
            "avg_seconds": round(sum(r["seconds"] for r in results) / n, 1),
            "tp": tp, "fp": fp, "fn": fn, "precision": prec(tp, fp), "recall": round(tp / (tp + fn), 3)}


def main():
    # the same 400 images as Part 5's loop: first N of val (dev) and of test (held-out)
    images = [f"val_{i:05d}" for i in range(N_PER_SPLIT)] + [f"test_{i:05d}" for i in range(N_PER_SPLIT)]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(run_graph_agent, images))
    (OUT_DIR / "graph_agent_results.json").write_text(json.dumps(results, indent=2))
    loop_results = json.loads((OUT_DIR / "agent_loop_results.json").read_text())
    summary = {"model": MODEL}
    for split in ("val", "test"):
        summary[split] = {"graph": summarize([r for r in results if r["split"] == split]),
                          "part5_loop": summarize([r for r in loop_results if r["split"] == split])}
    (OUT_DIR / "graph_agent_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
