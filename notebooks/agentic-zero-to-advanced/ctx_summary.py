"""Context-engineering post: every number in the post comes from this script (reads the saved run files)."""
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).parent / "out"
runs = json.loads((OUT / "ctx_eval_runs.json").read_text())
t6 = json.loads((OUT / "ctx_subagent_runs.json").read_text())

TASK_NAMES = {"T1": "needle in a 4,000-row CSV", "T2": "count across 200 reports", "T3": "stale vs current policy",
              "T4": "find 4 reports by meaning", "T5": "20 visual neighbours (DINOv3 + LanceDB)"}
ORDER = ["stuff", "payload", "harness", "harness+delegate", "search-paths", "search-paths+delegate",
         "v-payload", "v-harness", "v-harness+delegate"]


def table() -> list[dict]:
    g = defaultdict(list)
    for r in runs:
        g[(r["task"], r["strategy"])].append(r)
    rows = []
    for (task, strat), rs in sorted(g.items(), key=lambda kv: (kv[0][0], ORDER.index(kv[0][1]))):
        n = len(rs)
        rows.append({
            "task": task, "strategy": strat, "n": n, "correct": sum(r["correct"] for r in rs),
            "overflow": sum(r["status"] == "overflow" for r in rs),
            "max_steps": sum(r["status"] == "max_steps" for r in rs),
            "peak_tokens": max(r["peak_prompt_tokens"] for r in rs),
            "mean_total_tokens": round(sum(r["total_prompt_tokens"] + r["sub_tokens"] for r in rs) / n),
            "mean_seconds": round(sum(r["seconds"] for r in rs) / n, 1)})
    return rows


def t6_table() -> list[dict]:
    g = defaultdict(list)
    for r in t6:
        g[r["arm"]].append(r)
    return [{"arm": a, "n": len(rs), "correct": sum(r["correct"] for r in rs),
             "prompt_tokens": round(sum(r["prompt_tokens"] for r in rs) / len(rs))} for a, rs in g.items()]


if __name__ == "__main__":
    print(f"{'task':4} {'strategy':22} {'ok':>5} {'ovfl':>4} {'cap':>4} {'peak tok':>9} {'mean total tok':>15} {'s':>6}")
    last = None
    for r in table():
        if last and r["task"] != last:
            print()
        last = r["task"]
        print(f"{r['task']:4} {r['strategy']:22} {r['correct']}/{r['n']:<3} {r['overflow']:4d} {r['max_steps']:4d} "
              f"{r['peak_tokens']:9d} {r['mean_total_tokens']:15d} {r['mean_seconds']:6.1f}")
    print("\nT6: sub-agent brief size (12 target reports, exact verdict + confidence)")
    for r in t6_table():
        print(f"{r['arm']:14} {r['correct']}/{r['n']}  mean prompt tokens {r['prompt_tokens']}")
    json.dump({"table": table(), "t6": t6_table()}, open(OUT / "ctx_summary.json", "w"), indent=1)
