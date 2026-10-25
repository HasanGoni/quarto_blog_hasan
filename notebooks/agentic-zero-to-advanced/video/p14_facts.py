"""Part 14 videos: every number the narration says, read from the saved runs and asserted. Writes mc/src/facts_p14.json.
    python3 p14_facts.py"""
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "out"
S = json.loads((OUT / "ctx_summary.json").read_text())
row = {(r["task"], r["strategy"]): r for r in S["table"]}
t6 = {r["arm"]: r for r in S["t6"]}
old = json.loads((OUT / "old_ctx_eval_runs_maxtok1200.json").read_text())
quality = re.search(r"([\d.]+) \(random neighbours: ([\d.]+);", (OUT / "ctx_visual_quality_output.txt").read_text())
csv_tokens = (Path(__file__).resolve().parents[1] / ".." / "agentic-factory-segmentation" / "data" / "void_dataset_v1" / "metadata.csv").resolve().stat().st_size // 4

f = {
    "window": 65536,
    "csv_tokens": csv_tokens,
    "t1": {k: {"correct": row[("T1", k)]["correct"], "n": row[("T1", k)]["n"], "overflow": row[("T1", k)]["overflow"],
               "tokens": row[("T1", k)]["mean_total_tokens"]} for k in ("stuff", "payload", "harness", "harness+delegate")},
    "t2": {k: {"correct": row[("T2", k)]["correct"], "n": row[("T2", k)]["n"], "tokens": row[("T2", k)]["mean_total_tokens"]}
           for k in ("stuff", "payload", "harness", "harness+delegate")},
    "t2_old_stuff_correct": sum(r["correct"] for r in old if r["task"] == "T2" and r["strategy"] == "stuff"),
    "t2_old_stuff_completion_tokens": max(r["completion_tokens"] for r in old if r["task"] == "T2" and r["strategy"] == "stuff"),
    "t6": {"narrow": t6["narrow"]["prompt_tokens"], "pad100": t6["pad100-middle"]["prompt_tokens"],
           "runs": sum(r["n"] for r in S["t6"]), "correct": sum(r["correct"] for r in S["t6"])},
    "dino": {"near": float(quality.group(1)), "random": float(quality.group(2))},
    "t5": {"payload": row[("T5", "v-payload")]["mean_total_tokens"], "harness": row[("T5", "v-harness")]["mean_total_tokens"]},
}
# the narration says these; refuse to build if the data disagrees
assert 100_000 < f["csv_tokens"] < 120_000
assert (f["t1"]["stuff"]["overflow"], f["t1"]["payload"]["overflow"]) == (6, 6)
assert (f["t1"]["harness"]["correct"], f["t1"]["harness"]["n"]) == (6, 6) and 2500 < f["t1"]["harness"]["tokens"] < 3000
assert (f["t1"]["harness+delegate"]["correct"], f["t1"]["harness+delegate"]["n"]) == (3, 6)
assert (f["t2"]["stuff"]["correct"], f["t2"]["stuff"]["n"]) == (3, 3) and 25000 < f["t2"]["stuff"]["tokens"] < 30000
assert all(f["t2"][k]["correct"] == 0 for k in ("payload", "harness", "harness+delegate"))
assert f["t2_old_stuff_correct"] == 0 and f["t2_old_stuff_completion_tokens"] == 1200
assert (f["t6"]["narrow"], f["t6"]["pad100"], f["t6"]["runs"], f["t6"]["correct"]) == (364, 13924, 84, 84)
assert round(f["t6"]["pad100"] / f["t6"]["narrow"]) == 38
assert (round(f["dino"]["near"] * 100), round(f["dino"]["random"] * 100)) == (73, 10)
assert f["t5"]["payload"] < f["t5"]["harness"]
# --- chapter 6: the frameworks (out/ctx_frameworks.json, written by ctx_frameworks.py)
FW = json.loads((OUT / "ctx_frameworks.json").read_text())
H, DA, DAC, AD, ADC, ENV = (FW[k] for k in ("harness (best set-up per task)", "Deep Agents", "Deep Agents + chunk sub-agent",
                                            "ADK, my tools", "ADK, my tools + chunk sub-agent", "ADK, its own toolset (sandboxed)"))
f["fw"] = {"totals": {"harness": H["correct"], "da": DA["correct"], "da_chunk": DAC["correct"], "adk": AD["correct"],
                       "adk_chunk": ADC["correct"], "adk_env": ENV["correct"]},
           "n": H["n"], "t1_tokens": {"harness": H["per_task"]["T1"]["tokens_per_correct"], "adk": AD["per_task"]["T1"]["tokens_per_correct"],
                                      "adk_env": ENV["per_task"]["T1"]["tokens_per_correct"], "da": DA["per_task"]["T1"]["tokens_per_correct"]},
           "da_cap": DA["cap"] + DAC["cap"], "da_runs": sum(p["n"] for x in (DA, DAC) for p in x["per_task"].values()),
           "guard": [("answered" if g["correct"] else "stopped" if g["status"] == "no_answer" else "looped") for g in FW["guard"]]}
fw = f["fw"]
assert fw["n"] == 17 and fw["totals"] == {"harness": 12, "da": 6, "da_chunk": 10, "adk": 9, "adk_chunk": 8, "adk_env": 12}
assert [round(fw["t1_tokens"][k], -2) for k in ("harness", "adk", "adk_env", "da")] == [2800, 5100, 8100, 13100]
assert (fw["da_cap"], fw["da_runs"]) == (18, 34)
assert sorted(fw["guard"]) == ["answered", "looped", "stopped"]
(Path(__file__).parent / "mc" / "src" / "facts_p14.json").write_text(json.dumps(f, indent=1))
print("facts ok:", json.dumps({k: f[k] for k in ("csv_tokens", "t6", "dino", "t5", "fw")}))
