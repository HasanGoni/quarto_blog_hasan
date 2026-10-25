"""Context-engineering post: every number in the framework-comparison section comes from this script (reads the saved run files).

    uv run python ctx_frameworks.py        # prints the table, writes out/ctx_frameworks.json
Runs compared: my hand-written harness (ctx_eval_runs.json), Deep Agents (ctx_da_runs.json), Google ADK with my tools
(ctx_adk_runs.json) and ADK's own EnvironmentToolset in a Docker sandbox (ctx_adk_env_runs.json), plus the guard test.
"""
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).parent / "out"
QN = {"T1": 6, "T2": 3, "T3": 2, "T4": 3, "T5": 3}
# the best hand-harness setup per task is named here, not hidden: it flatters the harness, the post says so
HARNESS_BEST = {"T1": "harness", "T2": "harness", "T3": "harness", "T4": "search-paths", "T5": "v-harness"}
SETS = {"harness (best set-up per task)": None, "Deep Agents": ("ctx_da_runs.json", "da"),
        "Deep Agents + chunk sub-agent": ("ctx_da_runs.json", "da+chunk"), "ADK, my tools": ("ctx_adk_runs.json", "adk"),
        "ADK, my tools + chunk sub-agent": ("ctx_adk_runs.json", "adk+chunk"),
        "ADK, its own toolset (sandboxed)": ("ctx_adk_env_runs.json", "adk-env")}


def load(f):
    return json.loads((OUT / f).read_text())


def kind(r):
    if r["correct"]:
        return "ok"
    s = r["status"]
    if "Limit" in s or "Recursion" in s or s == "max_steps":
        return "cap"
    if "ContextWindow" in s or s == "overflow":
        return "overflow"
    return "wrong"


def collect() -> dict:
    ev = load("ctx_eval_runs.json")
    res = {}
    for name, src in SETS.items():
        per = {}
        for t in QN:
            if src is None:
                rs = [r for r in ev if r["task"] == t and r["strategy"] == HARNESS_BEST[t]]
            else:
                rs = [r for r in load(src[0]) if r["task"] == t and r["strategy"] == src[1]]
            ok = [r for r in rs if r["correct"]]
            tok = lambda r: r["total_prompt_tokens"] + r.get("sub_tokens", 0)
            per[t] = {"n": len(rs), "correct": len(ok), "cap": sum(kind(r) == "cap" for r in rs),
                      "overflow": sum(kind(r) == "overflow" for r in rs), "wrong": sum(kind(r) == "wrong" for r in rs),
                      "tokens_per_correct": round(sum(tok(r) for r in ok) / len(ok)) if ok else None,
                      "tokens_per_run": round(sum(tok(r) for r in rs) / len(rs)) if rs else None}
        res[name] = {"per_task": per, "correct": sum(p["correct"] for p in per.values()),
                     "n": sum(p["n"] for p in per.values()), "cap": sum(p["cap"] for p in per.values()),
                     "overflow": sum(p["overflow"] for p in per.values())}
    stuff = {t: [r for r in ev if r["task"] == t and r["strategy"] == "stuff"] for t in QN}
    res["stuff"] = {t: {"n": len(v), "correct": sum(r["correct"] for r in v)} for t, v in stuff.items() if v}
    g = load("ctx_da_guard_runs.json")
    res["guard"] = [{"strategy": r["strategy"], "correct": r["correct"], "calls": r["model_calls_all"],
                     "tokens": r["total_prompt_tokens"], "status": r["status"][:30]} for r in g]
    return res


if __name__ == "__main__":
    res = collect()
    for name in SETS:
        r = res[name]
        print(f"{name:36s} {r['correct']:2d}/{r['n']}  cap={r['cap']:2d} overflow={r['overflow']}  " +
              "  ".join(f"{t}:{p['correct']}/{p['n']}" for t, p in r["per_task"].items()))
        print(" " * 36, "tokens per correct run:", {t: p["tokens_per_correct"] for t, p in r["per_task"].items()})
    print("stuff", res["stuff"]); print("guard", res["guard"])
    (OUT / "ctx_frameworks.json").write_text(json.dumps(res, indent=1))
