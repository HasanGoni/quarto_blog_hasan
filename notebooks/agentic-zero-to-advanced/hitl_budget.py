"""Part 9: how many boxes must a person look at? Deferral curves from the Part 8 judges (saved probabilities, no model calls).

The judge decides every box on its own at the threshold chosen on dev. A person then re-checks the k boxes the judge was least sure about
and gets them right. "Least sure" = closest to the threshold in rank order (the raw probabilities of some judges are badly scaled, ranks are not).
Reference lines: random choice of boxes, and the best possible choice (re-check the judge's actual mistakes first).
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "out" / "typed"
JUDGES = [("band", "guided"), ("laya-english", "guided"), ("llm-verbalized", "guided"), ("llm-logprob", "plain"), ("llm-logprob", "guided")]


def load():
    rows = [json.loads(l) for l in (OUT / "candidates.jsonl").read_text().splitlines()]
    split = [r["split"] for r in rows]
    y = [r["y"] for r in rows]
    return rows, split, y


def judge_scores(name: str, tag: str):
    d = json.loads((OUT / f"results_{name}.json").read_text())["A"]
    return d["p"][tag], d["scores"][tag]["dev_threshold"]


def curve(name: str, tag: str):
    rows, split, y = load()
    p, thr = judge_scores(name, tag)
    idx = [i for i, s in enumerate(split) if s == "test"]
    pt, yt = [p[i] for i in idx], [y[i] for i in idx]
    n = len(pt)
    pred = [int(v >= thr) for v in pt]
    wrong = [int(a != b) for a, b in zip(pred, yt)]
    order_p = sorted(range(n), key=lambda i: pt[i])
    rank = [0.0] * n
    for r, i in enumerate(order_p):
        rank[i] = r / (n - 1)
    thr_rank = sum(v < thr for v in pt) / (n - 1)
    uncertain_first = sorted(range(n), key=lambda i: abs(rank[i] - thr_rank))
    base = sum(wrong)
    pts = []
    for k in range(0, n + 1, 5):
        fixed = sum(wrong[i] for i in uncertain_first[:k])
        pts.append({"k": k, "errors_left": base - fixed, "random_left": base * (1 - k / n), "best_left": max(0, base - k)})
    return {"judge": f"{name}/{tag}", "n": n, "positives": sum(yt), "threshold": thr, "errors_at_zero": base,
            "flagged": sum(pred), "tp": sum(a and b for a, b in zip(pred, yt)), "curve": pts}


def summary_row(c: dict, fracs=(0.1, 0.25, 0.5)) -> dict:
    n = c["n"]
    out = {"judge": c["judge"], "errors": c["errors_at_zero"]}
    for f in fracs:
        pt = min(c["curve"], key=lambda q: abs(q["k"] - f * n))
        out[f"fixed@{int(f*100)}%"] = round(c["errors_at_zero"] - pt["errors_left"], 1)
        out[f"random@{int(f*100)}%"] = round(c["errors_at_zero"] - pt["random_left"], 1)
    return out
