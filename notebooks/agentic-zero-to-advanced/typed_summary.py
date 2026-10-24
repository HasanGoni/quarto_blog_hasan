"""Part 8: one table from every results_*.json (what the post and the videos read, so no number is typed by hand)."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "out" / "typed"
ORDER = ["band", "laya-english", "laya-typed-decisions", "laya-multilingual", "llm-verbalized", "llm-logprob"]
res = {n: json.load(open(OUT / f"results_{n}.json")) for n in ORDER if (OUT / f"results_{n}.json").exists()}
rnd = lambda v, k=3: None if v is None else round(v, k)
summary = {"A": {}, "B": {}}
for n, r in res.items():
    a = r["A"]["scores"]
    summary["A"][n] = {tag: {"auc": rnd(s["test"]["auc"]), "brier_skill": rnd(s["test"]["brier_skill"], 2), "ece": rnd(s["test"]["ece"]),
                             "brier_skill_recal": rnd(s["test_recalibrated"]["brier_skill"], 2), "ece_recal": rnd(s["test_recalibrated"]["ece"]),
                             "flagged_at_0.5": s["test"]["flagged"], "dev_thr": rnd(s["dev_threshold"], 3),
                             "flagged_at_dev_thr": s["test_at_dev_threshold"]["flagged"], "tp_at_dev_thr": s["test_at_dev_threshold"]["tp"],
                             "precision_at_dev_thr": rnd(s["test_at_dev_threshold"]["precision"]), "recall_at_dev_thr": rnd(s["test_at_dev_threshold"]["recall"]),
                             "p50_ms": rnd(s["latency"]["p50_ms"], 3), "tokens": rnd(s["tokens"], 0)} for tag, s in a.items()}
    if "B" in r:
        b = r["B"]["scores"]
        summary["B"][n] = {k: {"acc": rnd(v["acc"]), "ece": rnd(v["ece"]), "p50_ms": rnd(v["latency"]["p50_ms"], 1), "tokens": rnd(v["tokens"], 0)} for k, v in b.items()}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print("A (held-out boxes, 513, 81 real)   guided: AUC | Brier skill raw->recal | ECE raw->recal | dev-thr flagged/tp/P/R | p50 ms | tokens")
for n, v in summary["A"].items():
    for tag in ("plain", "guided"):
        s = v.get(tag)
        if s: print(f"  {n:22s} {tag:7s} AUC {s['auc']:.3f} | {s['brier_skill']:+.2f}->{s['brier_skill_recal']:+.2f} | {s['ece']:.3f}->{s['ece_recal']:.3f} | thr {s['dev_thr']} {s['flagged_at_dev_thr']}/{s['tp_at_dev_thr']}/{s['precision_at_dev_thr']}/{s['recall_at_dev_thr']} | {s['p50_ms']} ms | {s['tokens']}")
print("B (520 real notes)  accuracy [ECE]  escalate | uncertain | n_kept   p50 ms")
for n, v in summary["B"].items():
    print(f"  {n:22s} " + " | ".join(f"{v[k]['acc']:.3f} [{v[k]['ece']:.3f}]" for k in ("escalate", "uncertain", "n_kept")) + "   " + " / ".join(str(v[k]["p50_ms"]) for k in ("escalate", "uncertain", "n_kept")))
