"""Part 5, second half: the same agent with fixes for two of the three failure buckets.

Split: val = dev (used to write the guidance), test = held-out (never looked at while
writing it). Variants:
  design   -- better role spec: what real defects look like in THIS data + a stop rule
  verify   -- code checks the report before accepting it; problems go back to the model
  both     -- design + verify
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from agent_loop import MODEL, OUT_DIR, SUBMIT_SPEC, SYSTEM, client
from inspection_tools import N_PER_SPLIT, TOOL_FUNCS, TOOL_SPECS, find_candidates, score_kept

MAX_STEPS, MAX_REJECTIONS = 14, 2

# Bucket 1 (design): the original prompt never said what a real void looks like or how this
# detector fails. This does, using only the DEV split (val, all 600 images, 1,736 candidates):
#   edge_distance_px == 0 : 1 real of 525        contrast >= 30 : 71% real (121)
#   contrast 20-30 : 36% real (196)              contrast < 15 : about 10% real or less
#   area < 20 px   : 4% real (415)
SYSTEM_DESIGN = SYSTEM + """
What the labeled dev data says about this detector:
- A box touching the image border (edge_distance_px = 0) is almost never real: 1 in 525.
- Contrast is the best signal. Above 30 about 7 in 10 boxes are real, 20 to 30 about 1 in 3,
  below 15 about 1 in 10 or fewer.
- Boxes smaller than 20 pixels in area are almost never real: about 1 in 25.
- Measure every candidate before you keep or drop it. Never keep an id you did not measure.
- Finish with verdict "review" if you kept anything, "pass" if you kept nothing."""


# Bucket 3 (verification): the checks the paper calls superficial are the ones that only ask
# "is it valid JSON". These test the report against what actually happened in the loop.
def verify_report(report: dict, measured: set[int], n_candidates: int) -> list[str]:
    problems = []
    kept = report.get("kept_ids", [])
    bad = [i for i in kept if not 0 <= i < n_candidates]
    if bad:
        problems.append(f"kept_ids {bad} are not valid candidate ids (valid: 0..{n_candidates - 1})")
    unmeasured = [i for i in kept if i not in measured and 0 <= i < n_candidates]
    if unmeasured:
        problems.append(f"you kept ids {unmeasured} without measuring them. Measure them first.")
    unchecked = [i for i in range(n_candidates) if i not in measured]
    if unchecked:
        problems.append(f"candidates {unchecked} were never measured. Measure them or say why not.")
    if report.get("verdict") == "pass" and kept:
        problems.append("verdict is 'pass' but kept_ids is not empty")
    if report.get("verdict") == "review" and not kept:
        problems.append("verdict is 'review' but kept_ids is empty")
    return problems


def run_agent_fixed(image: str, design: bool, verify: bool) -> dict:
    messages = [{"role": "system", "content": SYSTEM_DESIGN if design else SYSTEM},
                {"role": "user", "content": f"Inspect image {image}."}]
    report, rejections, t0, tokens, steps, calls_total = None, 0, time.time(), 0, 0, 0
    measured: set[int] = set()
    n_cand = 0
    for _ in range(MAX_STEPS):
        steps += 1
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SPECS + [SUBMIT_SPEC], temperature=0.2)
        tokens += resp.usage.total_tokens
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            messages.append({"role": "user", "content": "Use a tool, or call submit_report if you are done."})
            continue
        done = False
        for call in msg.tool_calls:
            calls_total += 1
            args = json.loads(call.function.arguments or "{}")
            name = call.function.name
            if name == "submit_report":
                problems = verify_report(args, measured, n_cand) if verify else []
                if problems and rejections < MAX_REJECTIONS:
                    rejections += 1
                    result = {"ok": False, "problems": problems, "fix": "Fix these and call submit_report again."}
                else:
                    report, result, done = args, {"ok": True}, True
            elif name in TOOL_FUNCS:
                try:
                    result = TOOL_FUNCS[name](**args)
                except Exception as exc:
                    result = {"error": str(exc)}
                if name == "find_candidates":
                    n_cand = len(result.get("candidates", []))
                if name == "measure_candidate" and "error" not in result:
                    measured.add(args["candidate_id"])
            else:
                result = {"error": f"unknown tool {name}"}
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
        if done:
            break
    kept_ids = report["kept_ids"] if report else []
    n_cand = len(find_candidates(image)["candidates"])
    tp, fp, fn = score_kept(image, kept_ids)
    return {"image": image, "finished": report is not None, "steps": steps, "tool_calls": calls_total,
            "rejections": rejections, "tokens": tokens, "seconds": round(time.time() - t0, 1),
            "n_candidates": n_cand, "kept": len(kept_ids), "tp": tp, "fp": fp, "fn": fn}


def summarize(rows: list[dict]) -> dict:
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    return {"n_images": len(rows), "finished": sum(r["finished"] for r in rows),
            "kept": sum(r["kept"] for r in rows), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None, "recall": round(tp / (tp + fn), 3),
            "avg_tool_calls": round(sum(r["tool_calls"] for r in rows) / len(rows), 2),
            "avg_tokens": round(sum(r["tokens"] for r in rows) / len(rows)),
            "avg_rejections": round(sum(r.get("rejections", 0) for r in rows) / len(rows), 2)}


if __name__ == "__main__":
    variant = sys.argv[1]
    design, verify = variant in ("design", "both"), variant in ("verify", "both")
    test = [f"test_{i:05d}" for i in range(N_PER_SPLIT)]
    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(lambda im: run_agent_fixed(im, design, verify), test))
    (OUT_DIR / f"fixes_{variant}.json").write_text(json.dumps({"summary": summarize(rows), "rows": rows}, indent=2))
    print(variant, json.dumps(summarize(rows)))
