"""Part 8: what each KIND of tool costs per call, measured warm and sequential on this machine (GB10 GPU shared with two vLLM servers).
rule: the contrast floor + min-area + border rule as one Python expression.  typed: Laya on GPU (from typed_eval) and on CPU.  LLM: the one-number
judgments (from typed_eval) and writing one of Part 7's real notes (the report node's prompt, on the real kept boxes of 20 real notes)."""
import json
import statistics
import time
from pathlib import Path

OUT = Path(__file__).parent / "out" / "typed"
rows = [json.loads(l) for l in (OUT / "candidates.jsonl").read_text().splitlines()]
notes = [json.loads(l) for l in (OUT / "notes.jsonl").read_text().splitlines()]
costs = {}

# 1. the deterministic rule, 1e6 calls
states = [r["state"] for r in rows] * 3000
t = time.perf_counter()
n = sum(1 for s in states if s["contrast"] >= 8 and s["area_px"] >= 20 and s["edge_distance_px"] > 0)
costs["rule"] = {"ns_per_call": (time.perf_counter() - t) / len(states) * 1e9, "calls": len(states), "tokens": 0}
print(f"rule: {costs['rule']['ns_per_call']:.0f} ns per call over {len(states)} calls")

# 2. Laya on CPU, same question the GPU run answered
from typed_eval import GUIDANCE_A, Q_A
from typed_models import LayaJudge
cpu = LayaJudge(None, device="cpu", name="laya-english-cpu")
fn = lambda r: cpu.noul(r["state"], f"{Q_A} {GUIDANCE_A}")
for r in rows[:3]:
    fn(r)
ms = sorted(fn(r)["ms"] for r in rows[3:33])
costs["laya_cpu"] = {"p50_ms": statistics.median(ms), "p90_ms": ms[int(.9 * (len(ms) - 1))], "device": cpu.device, "threads": __import__("torch").get_num_threads()}
print("laya on", cpu.device, f"p50 {costs['laya_cpu']['p50_ms']:.0f} ms")
del cpu

# 3. the LLM writing a note: Part 7's report prompt on real records (kept boxes + verdict + reason of 20 real notes)
from pipeline_graph import MODEL, SYSTEM_REPORT, client
sel = [n for n in notes if n["variant"] == "llm"][:20]
ms, toks, outs = [], [], []
for n in sel:
    rec = {"image": n["image"], "verdict": n["verdict"], "kept": n["kept"]}
    t = time.perf_counter()
    r = client.chat.completions.create(model=MODEL, temperature=0.2, messages=[{"role": "system", "content": SYSTEM_REPORT}, {"role": "user", "content": json.dumps(rec)}])
    ms.append((time.perf_counter() - t) * 1000); toks.append(r.usage.total_tokens); outs.append(r.usage.completion_tokens)
ms.sort()
costs["llm_note"] = {"p50_ms": statistics.median(ms), "p90_ms": ms[int(.9 * (len(ms) - 1))], "avg_tokens": statistics.mean(toks), "avg_completion_tokens": statistics.mean(outs), "n": len(ms)}
print(f"llm note: p50 {costs['llm_note']['p50_ms']:.0f} ms, {costs['llm_note']['avg_tokens']:.0f} tokens ({costs['llm_note']['avg_completion_tokens']:.0f} generated)")
(OUT / "costs.json").write_text(json.dumps(costs, indent=1))
