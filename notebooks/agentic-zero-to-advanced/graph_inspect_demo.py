"""Part 6: look at the graph as data (nodes, edges), then run it on one real image and print the trace."""
from graph_agent import GRAPH, run_graph_agent

g = GRAPH.get_graph()
print("nodes:", [n for n in g.nodes])
for e in g.edges:
    print(f"  {e.source:>9} -> {e.target:<9} {'(conditional)' if e.conditional else ''}")

r = run_graph_agent("test_00020")
print()
print({k: v for k, v in r.items() if k != "trace"})
for s in r["trace"]:
    calls = ", ".join(f"{c['tool']}({', '.join(f'{k}={v}' for k, v in c['args'].items())})" for c in s["calls"])
    print(f"step {s['step']}: {calls or '(just talked)'}")
