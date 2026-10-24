"""Part 8: head-to-head of judges on two narrow semantic calls that already exist in the Part 7 pipeline.

A. candidate: "is this detector box a real defect?"  (1,059 boxes, 203 real; fit on dev = first 200 val images, score on test = first 200 test images)
B. note:      three questions about the report notes the model wrote in Part 7 (the notes Part 7's six model runs wrote), answers taken from the record.

Metrics: accuracy, AUC, Brier, ECE (10 equal-width bins), latency (sequential, warm), tokens. Probabilities are kept per item in
out/typed/results_*.json so the plots and the pipeline integration use exactly what was measured here.
Usage:  python typed_eval.py <judge,judge,...> [A|B|AB]      judges: band, laya-english, laya-typed-decisions, laya-multilingual, llm-verbalized, llm-logprob, jev (needs an API key)
"""
import json
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).parent / "out" / "typed"
GUIDANCE_A = ("Background from labeled data: brighter boxes are real more often. Above contrast 30 about 7 in 10 boxes are real, "
              "20 to 30 about 1 in 3, below 15 about 1 in 10 or fewer. A box touching the image border is almost never real.")
Q_A = "Is this detector box a real defect rather than a false alarm?"
Q_B_ESC = ("Does this note say the image was escalated, meaning the pipeline was unsure and a human should look at it, "
           "as opposed to a normal review of confident defects?")
Q_B_UNC = "Does the note say that at least one kept candidate is uncertain or has low confidence?"
Q_B_N = "How many candidates does the note say were kept?"
OPTS_N = {"1": "exactly one candidate was kept", "2": "exactly two candidates were kept", "3": "exactly three candidates were kept",
          "4_or_more": "four or more candidates were kept"}


# ---------------------------------------------------------------------------------------------- metrics
def auc(p, y):
    pos = [a for a, b in zip(p, y) if b]
    neg = [a for a, b in zip(p, y) if not b]
    if not pos or not neg:
        return None
    s = sum((x > z) + 0.5 * (x == z) for x in pos for z in neg)
    return s / (len(pos) * len(neg))


def brier(p, y):
    return sum((a - b) ** 2 for a, b in zip(p, y)) / len(y)


def ece(conf, correct, bins=10):
    """Expected calibration error: gap between stated confidence and how often it was right, averaged over confidence bins."""
    n, tot = len(conf), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if idx:
            tot += len(idx) / n * abs(sum(correct[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return tot


def reliability(conf, correct, bins=10):
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if idx:
            out.append({"lo": lo, "hi": hi, "n": len(idx), "mean_conf": sum(conf[i] for i in idx) / len(idx),
                        "acc": sum(correct[i] for i in idx) / len(idx)})
    return out


def logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def temper(p, T):
    return 1 / (1 + math.exp(-logit(p) / T))


def fit_temperature(p, y):
    """Dev-set recalibration: the single temperature that minimises log-loss. One number, so it cannot memorise a few hundred boxes."""
    def nll(T):
        return -sum(math.log(max(1e-9, temper(a, T) if b else 1 - temper(a, T))) for a, b in zip(p, y)) / len(y)
    return min((0.25 * k for k in range(1, 81)), key=nll)


def best_f1_threshold(p, y):
    """Threshold that maximises F1 on the dev boxes (ties go to the higher threshold). Chosen on dev, applied to test."""
    best, best_t = -1.0, 0.5
    for t in sorted(set(p)):
        tp = sum(1 for a, b in zip(p, y) if a >= t and b)
        fp = sum(1 for a, b in zip(p, y) if a >= t and not b)
        fn = sum(y) - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 >= best:
            best, best_t = f1, t
    return best_t


def binary_report(p, y, thr=0.5, base_rate=None):
    pred = [int(a >= thr) for a in p]
    conf = [a if q else 1 - a for a, q in zip(p, pred)]
    correct = [int(q == b) for q, b in zip(pred, y)]
    tp = sum(1 for q, b in zip(pred, y) if q and b)
    fp = sum(1 for q, b in zip(pred, y) if q and not b)
    br = brier(p, y)
    br0 = brier([base_rate if base_rate is not None else sum(y) / len(y)] * len(y), y)  # predict the base rate for everything
    return {"n": len(y), "pos": sum(y), "thr": thr, "acc": sum(correct) / len(y), "auc": auc(p, y), "brier": br, "brier_skill": 1 - br / br0,
            "ece": ece(conf, correct),
            "tp": tp, "fp": fp, "fn": sum(y) - tp, "flagged": tp + fp,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / sum(y) if sum(y) else None,
            "reliability": reliability(conf, correct)}


# ---------------------------------------------------------------------------------------------- judges
def make_judge(name):
    from typed_models import LayaJudge, LLMJudge
    if name == "band":
        return None
    if name.startswith("laya-"):
        sub = name[5:]
        return LayaJudge(None if sub == "english" else sub, name=name)
    if name.startswith("llm-"):
        return LLMJudge(name[4:])
    if name == "jev":  # needs TYPESAFE_BASE_URL + TYPESAFE_API_KEY; never run in this series (no key)
        from typed_models import JevJudge
        return JevJudge()
    raise SystemExit(f"unknown judge {name}")


def latency(fn, items, warm=3, n=30):
    for it in items[:warm]:
        fn(it)
    ms = sorted(fn(it)["ms"] for it in items[warm:warm + n])
    return {"p50_ms": statistics.median(ms), "p90_ms": ms[int(0.9 * (len(ms) - 1))], "n": len(ms)}


def run_A(name, judge, rows):
    if judge is None:  # the deterministic tool: the contrast floor from Part 7, as a 0/1 "probability"
        t = time.perf_counter()
        ps = [float(r["state"]["contrast"] >= 8) for r in rows]
        us = (time.perf_counter() - t) / len(rows) * 1e6
        return {"guided": {"p": ps, "latency": {"p50_ms": us / 1000, "p90_ms": us / 1000, "n": len(rows)}, "tokens": 0}}
    out = {}
    for tag, instr in (("plain", Q_A), ("guided", f"{Q_A} {GUIDANCE_A}")):
        fn = lambda r: judge.noul(r["state"], instr)
        lat = latency(fn, rows)
        workers = 1 if judge.device != "vLLM" else 4  # laya is one GPU model; the LLM server can take parallel requests
        with ThreadPoolExecutor(workers) as pool:
            res = list(pool.map(fn, rows))
        out[tag] = {"p": [r["p"] for r in res], "latency": lat, "tokens": sum(r["tokens"] for r in res) / len(res)}
        print(f"  {name} A {tag}: p50 {lat['p50_ms']:.0f} ms", flush=True)
    return out


def run_B(name, judge, notes):
    out = {}
    qs = {"escalate": lambda n: judge.noul(n["note"], Q_B_ESC), "uncertain": lambda n: judge.noul(n["note"], Q_B_UNC),
          "n_kept": lambda n: judge.choice(n["note"], Q_B_N, OPTS_N)}
    for key, fn in qs.items():
        lat = latency(fn, notes)
        workers = 1 if judge.device != "vLLM" else 4
        with ThreadPoolExecutor(workers) as pool:
            res = list(pool.map(fn, notes))
        out[key] = {"p": [r["p"] for r in res], "latency": lat, "tokens": sum(r["tokens"] for r in res) / len(res)}
        print(f"  {name} B {key}: p50 {lat['p50_ms']:.0f} ms", flush=True)
    return out


def score_A(raw, rows):
    res = {}
    for tag, d in raw.items():
        dev = [(p, r["y"]) for p, r in zip(d["p"], rows) if r["split"] == "dev"]
        test = [(p, r["y"]) for p, r in zip(d["p"], rows) if r["split"] == "test"]
        T = fit_temperature([a for a, _ in dev], [b for _, b in dev]) if len(set(d["p"])) > 2 else 1.0
        base = sum(b for _, b in dev) / len(dev)
        thr = best_f1_threshold([a for a, _ in dev], [b for _, b in dev])
        res[tag] = {"test": binary_report([a for a, _ in test], [b for _, b in test], base_rate=base),
                    "test_at_dev_threshold": binary_report([a for a, _ in test], [b for _, b in test], thr=thr, base_rate=base),
                    "test_recalibrated": binary_report([temper(a, T) for a, _ in test], [b for _, b in test], base_rate=base), "temperature": T,
                    "dev_threshold": thr,
                    "all": binary_report(d["p"], [r["y"] for r in rows]), "latency": d["latency"], "tokens": d["tokens"]}
    return res


def score_B(raw, notes):
    res = {}
    for key in ("escalate", "uncertain"):
        y = [n["y_escalate" if key == "escalate" else "y_uncertain"] for n in notes]
        res[key] = {**binary_report(raw[key]["p"], y), "latency": raw[key]["latency"], "tokens": raw[key]["tokens"]}
    labels = list(OPTS_N)
    truth = [str(n["n_kept"]) if n["n_kept"] < 4 else "4_or_more" for n in notes]
    top = [max(p, key=p.get) for p in raw["n_kept"]["p"]]
    conf = [max(p.values()) for p in raw["n_kept"]["p"]]
    corr = [int(a == b) for a, b in zip(top, truth)]
    res["n_kept"] = {"n": len(notes), "acc": sum(corr) / len(corr), "ece": ece(conf, corr), "reliability": reliability(conf, corr),
                     "brier": sum(sum((p[k] - (k == t)) ** 2 for k in labels) for p, t in zip(raw["n_kept"]["p"], truth)) / len(truth),
                     "latency": raw["n_kept"]["latency"], "tokens": raw["n_kept"]["tokens"]}
    return res


def main():
    names = sys.argv[1].split(",")
    which = sys.argv[2] if len(sys.argv) > 2 else "AB"
    rows = [json.loads(l) for l in (OUT / "candidates.jsonl").read_text().splitlines()]
    notes = [json.loads(l) for l in (OUT / "notes.jsonl").read_text().splitlines()]
    for name in names:
        print(f"== {name}", flush=True)
        judge = make_judge(name)
        res = {"judge": name, "device": getattr(judge, "device", "python")}
        if "A" in which:
            raw = run_A(name, judge, rows)
            res["A"] = {"scores": score_A(raw, rows), "p": {k: v["p"] for k, v in raw.items()}}
            for tag, sc in res["A"]["scores"].items():
                s, d = sc["test"], sc["test_at_dev_threshold"]
                print(f"  A {tag} test: AUC {s['auc']:.3f} | at 0.5: acc {s['acc']:.3f} flagged {s['flagged']} | Brier skill {s['brier_skill']:+.2f} ECE {s['ece']:.3f} "
                      f"| at dev thr {d['thr']:.2f}: flagged {d['flagged']} tp {d['tp']} P {d['precision'] and round(d['precision'], 3)} R {d['recall']}", flush=True)
        if "B" in which and judge is not None:
            raw = run_B(name, judge, notes)
            res["B"] = {"scores": score_B(raw, notes), "p": {k: v["p"] for k, v in raw.items()}}
            print("  B:", {k: round(v["acc"], 3) for k, v in res["B"]["scores"].items() if "acc" in v}, flush=True)
        (OUT / f"results_{name}.json").write_text(json.dumps(res))
        del judge


if __name__ == "__main__":
    main()
