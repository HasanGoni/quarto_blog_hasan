"""Part 5: the agent loop, written by hand. No framework.

Same detector as Parts 2-4, split into tools (inspection_tools.py). The loop is the whole
"agent": ask the model, run whatever tool it asks for, put the result back in the message
list, ask again. Stops when the model calls submit_report or hits MAX_STEPS.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from detector_tool import DATA_DIR
from inspection_tools import TOOL_FUNCS, TOOL_SPECS, find_candidates, score_kept
from structured_output_demo import load_dotenv

load_dotenv()
MODEL = os.environ.get("AGENTIC_ZERO_MODEL", "openai:qwen3-coder").split(":", 1)[-1]
client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
OUT_DIR = Path(__file__).parent / "out"
MAX_STEPS = 12

SUBMIT_SPEC = {"type": "function", "function": {
    "name": "submit_report",
    "description": "Finish the inspection. Call this exactly once, when you are done.",
    "parameters": {"type": "object", "properties": {
        "kept_ids": {"type": "array", "items": {"type": "integer"},
                     "description": "candidate ids you believe are real defects"},
        "verdict": {"type": "string", "enum": ["pass", "review"]},
        "reason": {"type": "string"}}, "required": ["kept_ids", "verdict", "reason"]}}}

SYSTEM = """You inspect aluminum casting X-ray frames for defects.
Tools: find_candidates (noisy classical detector, most boxes are false alarms),
measure_candidate (look closer at one box), submit_report (finish).
Work like this: call find_candidates first. If there are no candidates, submit a pass right away.
Otherwise measure the candidates that look worth checking, decide which to keep, then submit_report.
Say in one short sentence what you are about to do before each tool call."""


def run_agent(image: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Inspect frame {image}."}]
    trace, kept_ids, report, t0, tokens = [], [], None, time.time(), 0
    for step in range(1, MAX_STEPS + 1):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SPECS + [SUBMIT_SPEC], temperature=0.2)
        tokens += resp.usage.total_tokens
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:  # model talked instead of acting: nudge it once per step
            trace.append({"step": step, "says": msg.content, "calls": []})
            messages.append({"role": "user", "content": "Use a tool, or call submit_report if you are done."})
            continue
        step_calls = []
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            if call.function.name == "submit_report":
                report, result = args, {"ok": True}
            elif call.function.name in TOOL_FUNCS:
                try:
                    result = TOOL_FUNCS[call.function.name](**args)
                except Exception as exc:  # tool errors go back to the model, they don't crash the loop
                    result = {"error": str(exc)}
            else:
                result = {"error": f"unknown tool {call.function.name}"}
            step_calls.append({"tool": call.function.name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
        trace.append({"step": step, "says": msg.content, "calls": step_calls})
        if report:
            break
    n_cand = len(find_candidates(image)["candidates"])  # for scoring only, not shown to the agent
    kept_ids = report["kept_ids"] if report else []
    tp, fp, fn = score_kept(image, kept_ids)
    return {"image": image, "finished": report is not None, "steps": len(trace),
            "tool_calls": sum(len(s["calls"]) for s in trace), "tokens": tokens,
            "seconds": round(time.time() - t0, 1), "n_candidates": n_cand,
            "kept": len(kept_ids), "verdict": report["verdict"] if report else None,
            "tp": tp, "fp": fp, "fn": fn, "trace": trace}


def main():
    images = sorted(p.name for p in DATA_DIR.glob("C0001_*.png"))
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run_agent, images))
    for r in results:
        print(f"{r['image']}: {r['n_candidates']:>3} cands -> kept {r['kept']:>3}, {r['steps']} steps, "
              f"{r['tool_calls']} tool calls, {r['seconds']}s, tp={r['tp']} fp={r['fp']}")
    (OUT_DIR / "agent_loop_results.json").write_text(json.dumps(results, indent=2))

    # baseline = keep every candidate (what Part 2 scored)
    base_tp = base_fp = base_fn = 0
    for r in results:
        t, f, n = score_kept(r["image"], list(range(r["n_candidates"])))
        base_tp, base_fp, base_fn = base_tp + t, base_fp + f, base_fn + n
    tp, fp, fn = (sum(r[k] for r in results) for k in ("tp", "fp", "fn"))
    prec = lambda t, f: round(t / (t + f), 3) if t + f else None
    summary = {
        "model": MODEL, "n_images": len(results),
        "finished": sum(r["finished"] for r in results),
        "avg_steps": round(sum(r["steps"] for r in results) / len(results), 2),
        "avg_tool_calls": round(sum(r["tool_calls"] for r in results) / len(results), 2),
        "avg_tokens": round(sum(r["tokens"] for r in results) / len(results)),
        "avg_seconds": round(sum(r["seconds"] for r in results) / len(results), 1),
        "baseline": {"tp": base_tp, "fp": base_fp, "fn": base_fn, "precision": prec(base_tp, base_fp),
                     "recall": round(base_tp / (base_tp + base_fn), 3)},
        "agent": {"tp": tp, "fp": fp, "fn": fn, "precision": prec(tp, fp),
                  "recall": round(tp / (tp + fn), 3)},
    }
    (OUT_DIR / "agent_loop_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
