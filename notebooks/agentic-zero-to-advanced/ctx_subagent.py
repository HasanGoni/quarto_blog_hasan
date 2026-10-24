"""Context-engineering post, T6: how much should a sub-agent be given?

The same chunk analyst (ctx_harness.ANALYST_SYSTEM), the same question about ONE target report, three briefs:

    narrow   the target report only
    pad20    the target + 19 other reports      (~5k tokens)
    pad100   the target + 99 other reports      (~14k tokens)

For the padded briefs the target sits at the start, the middle or the end, to test "lost in the middle".
The question needs two exact facts from the target (its verdict and the confidence of its first kept candidate),
so a sub-agent that drifts onto a neighbouring report gets it wrong.

    uv run python ctx_subagent.py            # ~100 short calls, resumes from out/ctx_subagent_runs.json
"""
import json
import random
import re
import time
from pathlib import Path

from ctx_corpus import ARCHIVE
from ctx_harness import ANALYST_SYSTEM, NO_THINK, client, MODEL, est_tokens
from ctx_eval import load_reports

OUT = Path(__file__).parent / "out" / "ctx_subagent_runs.json"
N_TARGETS = 12
FORMAT = ('Reply with exactly one line: ANSWER: {"verdict": "<pass|review|escalate>", "confidence": <number>} '
          "for the report named in the question, and nothing else.")


def brief(files: list[Path]) -> str:
    return "\n\n".join(f"### FILE reports/{f.name}\n{f.read_text()}" for f in files)


def target_ids() -> list[str]:
    reps = load_reports()
    ok = [i for i, r in reps.items() if r["kept"]]
    return random.Random(3).sample(ok, N_TARGETS)


def gold(tid: str) -> dict:
    txt = (ARCHIVE / "reports" / f"{tid}.md").read_text()
    verdict = re.search(r"^verdict: (\w+)", txt, re.M).group(1)
    conf = float(re.search(r"kept candidate \d+: .*?confidence ([\d.]+)", txt).group(1))
    return {"verdict": verdict, "confidence": conf}


def arms(tid: str) -> list[tuple[str, list[Path]]]:
    rng = random.Random(tid)
    others = sorted(p for p in (ARCHIVE / "reports").glob("*.md") if p.stem != tid)
    target = ARCHIVE / "reports" / f"{tid}.md"
    out = [("narrow", [target])]
    for size, label in [(20, "pad20"), (100, "pad100")]:
        pad = rng.sample(others, size - 1)
        for pos_name, pos in [("start", 0), ("middle", size // 2), ("end", size - 1)]:
            files = pad[:pos] + [target] + pad[pos:]
            out.append((f"{label}-{pos_name}", files))
    return out


def main():
    runs = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r["target"], r["arm"]) for r in runs}
    for tid in target_ids():
        g = gold(tid)
        q = f"Question: for report {tid}, what is the verdict and the confidence of the FIRST kept candidate listed?"
        for arm, files in arms(tid):
            if (tid, arm) in done:
                continue
            body = brief(files)
            t0 = time.time()
            r = client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=200, extra_body=NO_THINK,
                messages=[{"role": "system", "content": ANALYST_SYSTEM + " " + FORMAT},
                          {"role": "user", "content": f"{q}\n\n{body}"}])
            txt = r.choices[0].message.content or ""
            m = re.search(r"ANSWER:\s*(\{.*\})", txt, re.S)
            try:
                a = json.loads(m.group(1)) if m else None
                ok = bool(a) and a["verdict"] == g["verdict"] and abs(float(a["confidence"]) - g["confidence"]) < 1e-6
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                a, ok = None, False
            runs.append({"target": tid, "arm": arm, "n_files": len(files), "prompt_tokens": r.usage.prompt_tokens,
                         "est_body_tokens": est_tokens(body), "answer": txt, "gold": g, "correct": ok,
                         "seconds": round(time.time() - t0, 1)})
            OUT.write_text(json.dumps(runs, indent=1))
            print(f"{tid} {arm:14s} correct={ok!s:5} prompt_tokens={r.usage.prompt_tokens:6d} {runs[-1]['seconds']}s",
                  flush=True)


if __name__ == "__main__":
    main()
