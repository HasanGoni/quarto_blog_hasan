"""Context-engineering post: four retrieval tasks x several strategies, scored against exact ground truth.

    T1 needle    "what are n_voids and void_area of test_0xxxx?"   (4,000-row CSV)           6 questions
    T2 aggregate "how many reports are <verdict> with a <severity> candidate?" (200 reports)  3 questions
    T3 stale     "what is the minimum reportable area?"            (current + superseded doc)  2 questions
    T4 semantic  "give me 4 reports about <idea in other words>"   (200 reports, by meaning)   3 questions

    uv run python ctx_eval.py                 # all strategies, resumes from out/ctx_eval_runs.json
    uv run python ctx_eval.py T4 harness      # only some tasks / strategies
"""
import csv
import json
import random
import sys
import time
from pathlib import Path

from ctx_corpus import ARCHIVE
from ctx_harness import Config, parse_answer, run

OUT = Path(__file__).parent / "out" / "ctx_eval_runs.json"

STRATEGIES = {
    # everything the question is about, pasted into the prompt. Best case for stuffing: it gets the right folder.
    "stuff": Config("stuff"),
    # tools that hand back whole things: whole files, whole documents from search
    "payload": Config("payload"),
    # the article's checklist: grep, paged reads, offload large results, search returns paths
    "harness": Config("harness", grep=True, paged=True, offload=True, search_returns_paths=True),
    "harness+delegate": Config("harness+delegate", grep=True, paged=True, offload=True, search_returns_paths=True,
                               delegate=True),
    # search-only arms (no grep/glob), to isolate what the semantic path costs
    "search-paths": Config("search-paths", grep=True, grep_tools=False, paged=True, offload=True,
                           search_returns_paths=True),
    "search-paths+delegate": Config("search-paths+delegate", grep=True, grep_tools=False, paged=True, offload=True,
                                    search_returns_paths=True, delegate=True),
}
# T5 (visual retrieval) arms: the same checklist, with DINOv3 + LanceDB `similar_cases` added
V = dict(grep=True, paged=True, offload=True, visual=True)
STRATEGIES |= {
    "v-payload": Config("v-payload", visual=True),
    "v-harness": Config("v-harness", search_returns_paths=True, **V),
    "v-harness+delegate": Config("v-harness+delegate", search_returns_paths=True, delegate=True, **V),
}
ALL = [s for s in STRATEGIES if not s.startswith("v-")]
CORE = ["stuff", "payload", "harness", "harness+delegate"]
TASK_STRATEGIES = {"T1": CORE, "T2": CORE, "T3": CORE, "T4": ALL,
                   "T5": ["v-payload", "v-harness", "v-harness+delegate"]}


# ---------------------------------------------------------------------------------------------------
# ground truth, computed from the same files the agent sees
# ---------------------------------------------------------------------------------------------------
def load_reports() -> dict[str, dict]:
    out = {}
    for f in sorted((ARCHIVE / "reports").glob("*.md")):
        txt = f.read_text()
        kept = []
        for line in txt.splitlines():
            if line.startswith("- kept candidate"):
                kept.append({"severity": line.split("severity ")[1].split(",")[0],
                             "uncertain": "uncertain true" in line})
        out[f.stem] = {"verdict": next(l for l in txt.splitlines() if l.startswith("verdict:")).split(": ")[1],
                       "n_examined": int(next(l for l in txt.splitlines()
                                              if l.startswith("candidates_examined")).split(": ")[1]),
                       "kept": kept}
    return out


def build_tasks() -> list[dict]:
    rng = random.Random(7)
    rows = list(csv.DictReader(open(ARCHIVE / "exports" / "inspection_log.csv")))
    by_id = {r["id"]: r for r in rows}
    tasks = []
    for tid in rng.sample([f"test_{i:05d}" for i in range(600)], 6):
        r = by_id[tid]
        tasks.append({"task": "T1", "q": f"In exports/inspection_log.csv, what are n_voids and void_area for the "
                      f"row with id {tid}? Answer: ANSWER: {{\"n_voids\": int, \"void_area\": int}}",
                      "gold": {"n_voids": int(r["n_voids"]), "void_area": int(r["void_area"])}})
    reps = load_reports()
    for verdict, sev in [("escalate", "critical"), ("escalate", "major"), ("review", "minor")]:
        n = sum(1 for r in reps.values() if r["verdict"] == verdict and any(k["severity"] == sev for k in r["kept"]))
        tasks.append({"task": "T2", "q": f"How many reports in reports/ have verdict {verdict} AND at least one kept "
                      f"candidate with severity {sev}? Answer: ANSWER: {{\"count\": int}}", "gold": {"count": n}})
    for q in ["What is the smallest void area, in pixels, that the pipeline reports? Name the policy file you used. "
              "Answer: ANSWER: {\"min_area_px\": int, \"source\": \"<file name>\"}",
              "Below how many pixels of area is a candidate NOT reported? Name the policy file you used. "
              "Answer: ANSWER: {\"min_area_px\": int, \"source\": \"<file name>\"}"]:
        tasks.append({"task": "T3", "q": q, "gold": {"min_area_px": 20, "source": "policy-2026-10-17.md"}})
    sem = [("Find 4 reports where a human should double-check a defect that the model was unsure about.",
            lambda r: any(k["uncertain"] for k in r["kept"])),
           ("Find 4 reports about large, serious voids.", lambda r: any(k["severity"] == "critical" for k in r["kept"])),
           ("Find 4 reports where candidates were examined but the part simply passed.",
            lambda r: r["verdict"] == "pass" and r["n_examined"] > 0)]
    for q, pred in sem:
        tasks.append({"task": "T4", "q": q + " Answer: ANSWER: {\"ids\": [\"test_00000\", ...4 report ids]}",
                      "gold_ids": sorted(i for i, r in reps.items() if pred(r))})
    from ctx_visual import neighbours
    for tid, k, field, thr in [("test_00010", 20, "n_voids", 3), ("test_00123", 20, "void_area", 400),
                               ("test_00345", 20, "n_voids", 0)]:
        nb = neighbours(tid, k)
        n = sum(int(by_id[i][field]) >= thr if thr else int(by_id[i][field]) == 0 for i in nb)
        what = f"{field} of at least {thr}" if thr else "no voids at all (n_voids 0)"
        tasks.append({"task": "T5", "q": f"cases/ holds one record per test image. Use similar_cases to get the {k} "
                      f"test images that look most like {tid} (not counting {tid} itself). How many of them have "
                      f"{what}? For void_area, the record's void_area_total_px is the field. "
                      f"Answer: ANSWER: {{\"count\": int}}", "gold": {"count": n}})
    return tasks


def stuffed_context(task: str) -> str:
    parts = {"T1": ["exports/inspection_log.csv"], "T2": ["reports/*.md"], "T3": ["policy/*.md"],
             "T4": ["reports/*.md"]}[task]
    chunks = []
    for pat in parts:
        for f in sorted(ARCHIVE.glob(pat)):
            chunks.append(f"### FILE {f.relative_to(ARCHIVE)}\n{f.read_text()}")
    return "\n\n".join(chunks)


def score(t: dict, ans: dict | None) -> bool:
    if not ans:
        return False
    try:
        if t["task"] == "T1":
            return int(ans["n_voids"]) == t["gold"]["n_voids"] and int(ans["void_area"]) == t["gold"]["void_area"]
        if t["task"] in ("T2", "T5"):
            return int(ans["count"]) == t["gold"]["count"]
        if t["task"] == "T3":
            return int(ans["min_area_px"]) == 20 and t["gold"]["source"] in str(ans.get("source", ""))
        ids = list(dict.fromkeys(ans["ids"]))
        return len(ids) == 4 and all(i in t["gold_ids"] for i in ids)
    except (KeyError, TypeError, ValueError):
        return False


def main():
    want_t = [a for a in sys.argv[1:] if a in TASK_STRATEGIES]
    want_s = [a for a in sys.argv[1:] if a in STRATEGIES]
    runs = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r["question"], r["strategy"]) for r in runs}
    for t in build_tasks():
        if want_t and t["task"] not in want_t:
            continue
        for s in TASK_STRATEGIES[t["task"]]:
            if (want_s and s not in want_s) or (t["q"], s) in done:
                continue
            t0 = time.time()
            m = run(t["q"], STRATEGIES[s], stuffed=stuffed_context(t["task"]) if s == "stuff" else None)
            m["task"], m["seconds"] = t["task"], round(time.time() - t0, 1)
            m["parsed"] = parse_answer(m["answer"])
            m["correct"] = score(t, m["parsed"])
            m["gold"] = t.get("gold", t.get("gold_ids"))
            runs.append(m)
            OUT.write_text(json.dumps(runs, indent=1))
            print(f"{t['task']} {s:22s} {m['status']:9s} correct={m['correct']!s:5} steps={m['steps']:2d} "
                  f"peak={m['peak_prompt_tokens']:6d} total={m['total_prompt_tokens']:7d} sub={m['sub_tokens']:5d} "
                  f"{m['seconds']:5.1f}s", flush=True)


if __name__ == "__main__":
    main()
