"""Part 7: prove every branch of the pipeline with a scripted fake model (no vLLM, no GPU).

The real run may never produce invalid JSON or a made-up candidate id, so a real run proves nothing about
those paths. Here the 'model' is a script: each call returns the next prepared string. The detector and
the measuring tool are the real ones, on a real image (test_00200, 7 candidates, ids 0..6; ids 1, 3, 4, 5 pass the rules).
"""
import json
from types import SimpleNamespace

import pipeline_graph as pl

IMG = "test_00200"
N = 7


def fake(script):
    it, calls = iter(script), []

    def create(**kw):
        calls.append(kw["messages"][0]["content"][:20])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(it)))],
                               usage=SimpleNamespace(total_tokens=10))

    pl.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return calls


def labels(default="defect", conf=0.9, over=None):
    over = over or {}
    return json.dumps({"labels": [{"id": i, "label": over.get(i, default), "confidence": conf, "reason": "x"}
                                  for i in range(N)]})


def run(script, classifier="llm", image=IMG):
    calls = fake(script)
    final = pl.build_pipeline(classifier).invoke(pl.initial_state(image))
    return final, [t["node"] for t in final["trace"]], calls


def check(name, cond, detail):
    assert cond, f"FAILED {name}: {detail}"
    print(f"{name:<34} {detail}")


# 1. nothing detected: measure/classify/rules/decide are skipped, no model call at all
real_find = pl.find_candidates
pl.find_candidates = lambda image: {"candidates": []}
final, path, calls = run([])
pl.find_candidates = real_find
check("no candidates", path == ["detect", "report"] and not calls and final["verdict"] == "pass",
      f"{path} | model calls {len(calls)} | verdict {final['verdict']}")

# 2. everything is a false alarm: full path, verdict pass, and the report does NOT call the model
final, path, calls = run([labels("false_alarm")])
check("all false alarms", final["verdict"] == "pass" and len(calls) == 1 and final["report"]["summary_source"] == "template",
      f"verdict {final['verdict']} | model calls {len(calls)} (classify only) | prose from {final['report']['summary_source']}")

# 3. one confident defect: review, and the report calls the model once
final, path, calls = run([labels("false_alarm", over={5: "defect"}), "Candidate 5 was kept for review."])
check("one confident defect", final["verdict"] == "review" and [k["id"] for k in final["kept"]] == [5] and len(calls) == 2,
      f"verdict {final['verdict']} | kept {[k['id'] for k in final['kept']]} | model calls {len(calls)}")

# 4. invalid JSON once, valid on the retry: still ends normally, and the retry is in the trace
final, path, calls = run([labels()[:40], labels("false_alarm")])
check("bad JSON, retry works", final["classify_status"] == "ok" and next(t for t in final["trace"] if t["node"] == "classify")["attempts"] == 2,
      f"status {final['classify_status']} | attempts 2 | verdict {final['verdict']}")

# 5. well-formed JSON that skips an id is NOT accepted: validating the shape is not enough
final, path, calls = run([json.dumps({"labels": json.loads(labels())["labels"][:-1]}), labels("false_alarm")])
check("missing id is rejected", final["classify_status"] == "ok" and len(calls) == 2,
      f"first answer covered {N - 1}/{N} ids -> retried -> status {final['classify_status']}")

# 6. invalid twice: the classify->decide edge skips the rules and the pipeline escalates, it does not guess
final, path, calls = run(["not json", "still not json", "A note."])
check("classifier fails twice", path == ["detect", "measure", "classify", "decide", "report"] and final["verdict"] == "escalate",
      f"{path} | verdict {final['verdict']}")

# 7. one uncertain survivor: escalate
final, path, calls = run([labels("false_alarm", over={5: "uncertain"}), "Candidate 5 is uncertain."])
check("uncertain survivor", final["verdict"] == "escalate", f"verdict {final['verdict']} | {final['reason']}")

# 8. a 'defect' label with low confidence counts as uncertain too
final, path, calls = run([labels("false_alarm", conf=0.4, over={5: "defect"}), "Candidate 5 is a weak call."])
check("low confidence", final["verdict"] == "escalate", f"verdict {final['verdict']} | {final['reason']}")

# 9. too many survivors on one image: escalate (limit lowered to 1 for the test)
pl.MAX_KEPT = 1
final, path, calls = run([labels(), "Several candidates survived."])
pl.MAX_KEPT = 4
check("too many survivors", final["verdict"] == "escalate" and len(final["kept"]) > 1,
      f"kept {len(final['kept'])} > limit 1 -> {final['verdict']}")

# 10. business rules beat the classifier: border-touching and tiny boxes labeled 'defect' are still dropped
final, path, calls = run([labels(), "Some candidates survived."])
by = {r["id"]: r["by"] for r in final["rejected"]}
check("rules beat the classifier", "R1 border" in by.values() or "R2 min area" in by.values(),
      f"rejected by rule: { {i: b for i, b in by.items() if b != 'classifier'} }")

# 11. the model's prose cites an id that does not exist: it is thrown away and the template is used
final, path, calls = run([labels("false_alarm", over={5: "defect"}), "Candidate 99 looks severe."])
check("invented id in prose", final["report"]["summary_source"].startswith("template") and "99" not in final["report"]["summary"],
      f"prose source: {final['report']['summary_source']}")

# 12. deterministic classifier: the same graph runs with no model in classify
final, path, calls = run(["Candidate 5 was kept for review."], classifier="band")
check("band classifier", final["classify_status"] == "ok" and len(calls) <= 1,
      f"model calls {len(calls)} (report only) | verdict {final['verdict']}")

# 13. rules on synthetic measurements: every rule and every severity band, no detector involved
ms = [{"candidate_id": i, "contrast": 12.0, "area_px": a, "aspect_ratio": 1.0, "edge_distance_px": e}
      for i, (a, e) in enumerate([(10, 50), (30, 0), (30, 50), (100, 50), (300, 50)])]
labs = [{"id": i, "label": "defect", "confidence": 0.9, "reason": ""} for i in range(5)]
kept, rejected = pl.apply_rules(ms, labs)
check("each rule and severity band", {r["id"]: r["by"] for r in rejected} == {0: "R2 min area", 1: "R1 border"}
      and [(k["id"], k["severity"]) for k in kept] == [(2, "minor"), (3, "major"), (4, "critical")],
      f"rejected { {r['id']: r['by'] for r in rejected} } | kept {[(k['id'], k['severity']) for k in kept]}")
