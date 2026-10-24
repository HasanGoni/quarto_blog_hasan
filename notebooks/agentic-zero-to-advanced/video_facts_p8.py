"""Part 8 videos: collect every number the Motion Canvas chapters show into one facts.json, and assert the claims the narration makes.
Reads out/typed/{summary,costs,guard_audit,notes,results_*}.json. Writes video/mc/src/facts.json."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "out" / "typed"
S = json.load(open(OUT / "summary.json"))
C = json.load(open(OUT / "costs.json"))
AUD = json.load(open(OUT / "guard_audit.json"))
notes = [json.loads(l) for l in (OUT / "notes.jsonl").read_text().splitlines()]
res = json.load(open(OUT / "results_laya-english.json"))["B"]["p"]

flagged = sorted(i for i, n in enumerate(notes) if ((res["escalate"][i] >= .5) != bool(n["y_escalate"])) or ((res["uncertain"][i] >= .5) != bool(n["y_uncertain"])))
cats = {k: AUD[k]["notes"] for k in ("laya_called_a_fine_note_uncertain", "laya_misread_an_explicit_verdict", "ambiguous", "real_note_omissions")}
assert sorted(sum(cats.values(), [])) == flagged, "the audit file does not cover exactly the notes Laya disagreed on"
wrong = len(cats["laya_called_a_fine_note_uncertain"]) + len(cats["laya_misread_an_explicit_verdict"])
assert (len(notes), len(flagged), wrong, len(cats["ambiguous"]), len(cats["real_note_omissions"])) == (520, 72, 70, 2, 0)

ex_ok, ex_bad, ex_amb = notes[6], notes[2], notes[cats["ambiguous"][0]]  # an escalation Laya reads right, a fine note Laya calls uncertain, an arguable one
assert ex_ok["y_escalate"] and ex_ok["y_uncertain"] and res["escalate"][6] > .9 and res["uncertain"][6] > .9
assert not ex_bad["y_uncertain"] and res["uncertain"][2] >= .5 and "uncertain" not in ex_bad["note"].lower() and "escalat" in ex_amb["note"].lower()
assert 30 < C["rule"]["ns_per_call"] < 50 and 15 < S["A"]["laya-english"]["guided"]["p50_ms"] < 25 and 450 < C["laya_cpu"]["p50_ms"] < 600
assert 2.0 < C["llm_note"]["p50_ms"] / 1000 < 3.0 and C["llm_note"]["avg_tokens"] > 200
facts = {
    "costs": {"rule_ns": round(C["rule"]["ns_per_call"]), "laya_gpu_ms": round(S["A"]["laya-english"]["guided"]["p50_ms"]), "laya_cpu_ms": round(C["laya_cpu"]["p50_ms"]),
              "llm_judge_ms": round(S["A"]["llm-logprob"]["guided"]["p50_ms"]), "llm_judge_number_ms": round(S["A"]["llm-verbalized"]["guided"]["p50_ms"]),
              "llm_note_ms": round(C["llm_note"]["p50_ms"]), "llm_note_tokens": round(C["llm_note"]["avg_tokens"]), "llm_judge_tokens": round(S["A"]["llm-verbalized"]["guided"]["tokens"])},
    "auc": {j: {t: S["A"][j][t]["auc"] for t in S["A"][j]} for j in S["A"] if j in ("band", "laya-english", "llm-verbalized", "llm-logprob")},
    "notes_acc": {j: {q: S["B"][j][q]["acc"] for q in ("escalate", "uncertain", "n_kept")} for j in ("laya-english", "llm-verbalized", "llm-logprob")},
    "laya_guided": {"ece": S["A"]["laya-english"]["guided"]["ece"], "flagged_at_0.5": S["A"]["laya-english"]["guided"]["flagged_at_0.5"]},
    "guard": {"n": len(notes), "disagree": len(flagged), "agree": len(notes) - len(flagged), "laya_wrong": wrong, "ambiguous": len(cats["ambiguous"]), "uncertain_false": len(cats["laya_called_a_fine_note_uncertain"]),
              "misread_verdict": len(cats["laya_misread_an_explicit_verdict"]), "real_omissions": len(cats["real_note_omissions"]),
              "flag_idx": cats, "all_flagged": flagged},
    "example_ok": {"i": 6, "image": ex_ok["image"], "note": ex_ok["note"], "verdict": ex_ok["verdict"], "kept": ex_ok["kept"], "p_escalate": round(res["escalate"][6], 3), "p_uncertain": round(res["uncertain"][6], 3)},
    "example_ambiguous": {"i": cats["ambiguous"][0], "image": ex_amb["image"], "note": ex_amb["note"], "verdict": ex_amb["verdict"]},
    "example_bad": {"i": 2, "image": ex_bad["image"], "note": ex_bad["note"], "verdict": ex_bad["verdict"], "kept": ex_bad["kept"], "p_uncertain": round(res["uncertain"][2], 3)},
}
dst = Path(__file__).parent / "video" / "mc" / "src" / "facts.json"
dst.write_text(json.dumps(facts, indent=1))
print("facts.json written:", json.dumps(facts["costs"]), "| guard", {k: v for k, v in facts["guard"].items() if not isinstance(v, (list, dict))})
