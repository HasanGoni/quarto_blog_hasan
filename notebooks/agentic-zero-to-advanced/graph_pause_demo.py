"""Part 6: pause the graph before a tool runs, look at the state, then resume.

This is the thing a `for` loop cannot do for free. With a checkpointer, LangGraph saves the state after
every node, so `interrupt_before=["tools"]` stops the run right before the tool call and lets us inspect
(or even edit) what the model is about to do.
"""
from langgraph.checkpoint.memory import MemorySaver

from graph_agent import build_graph, initial_state

graph = build_graph(checkpointer=MemorySaver(), interrupt_before=["tools"])
cfg = {"configurable": {"thread_id": "demo"}}

graph.invoke(initial_state("test_00020"), cfg)
snap = graph.get_state(cfg)
print("paused before node:", snap.next)
want = snap.values["messages"][-1]["tool_calls"][0]["function"]
print("model wants to call:", want["name"], want["arguments"])
print("messages so far:", len(snap.values["messages"]), "| step:", snap.values["step"])

graph.invoke(None, cfg)  # resume: None means "continue from the saved state"
snap = graph.get_state(cfg)
print("paused again before:", snap.next, "| step:", snap.values["step"], "| tool results so far:",
      sum(m["role"] == "tool" for m in snap.values["messages"]))
