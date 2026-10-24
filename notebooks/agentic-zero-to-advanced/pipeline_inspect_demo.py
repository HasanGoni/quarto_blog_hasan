"""Part 7: the pipeline graph as data, then one real image run through it with the trace printed."""
import json

from pipeline_graph import build_pipeline, run_pipeline

graph = build_pipeline("llm")
g = graph.get_graph()
print("nodes:", list(g.nodes))
for e in g.edges:
    print(f"  {e.source:>9} -> {e.target:<9} {'(conditional)' if e.conditional else ''}")

r = run_pipeline("test_00020", graph)
print()
print({k: v for k, v in r.items() if k not in ("report", "trace", "strict")})
for t in r["trace"]:
    print(" ", t)
print()
print(json.dumps({k: v for k, v in r["report"].items() if k != "rejected"}, indent=2))
print("rejected:", [(x["id"], x["by"]) for x in r["report"]["rejected"]])
