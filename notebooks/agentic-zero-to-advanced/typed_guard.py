"""Part 8: a typed small model as a GUARD on the report note. The LLM writes the prose; Laya reads the finished note and answers two
typed questions (was it escalated? is a kept candidate uncertain?). If either answer disagrees with the structured record that the
note was written from, the note is replaced by the template. This reads Laya's saved answers from typed_eval.py, so it adds no new model run.
Usage: python typed_guard.py [judge]     writes out/typed/guard_<judge>.json and prints every disagreement for a manual audit."""
import json
import sys
from pathlib import Path

OUT = Path(__file__).parent / "out" / "typed"
judge = sys.argv[1] if len(sys.argv) > 1 else "laya-english"
notes = [json.loads(l) for l in (OUT / "notes.jsonl").read_text().splitlines()]
res = json.load(open(OUT / f"results_{judge}.json"))["B"]["p"]
rows = []
for i, n in enumerate(notes):
    pe, pu = res["escalate"][i], res["uncertain"][i]
    bad = []
    if (pe >= 0.5) != bool(n["y_escalate"]):
        bad.append(f"escalate: record {bool(n['y_escalate'])}, reads {pe:.2f}")
    if (pu >= 0.5) != bool(n["y_uncertain"]):
        bad.append(f"uncertain: record {bool(n['y_uncertain'])}, reads {pu:.2f}")
    rows.append({"i": i, "variant": n["variant"], "run": n["run"], "image": n["image"], "disagree": bad, "note": n["note"], "kept": n["kept"]})
flag = [r for r in rows if r["disagree"]]
print(f"{judge}: {len(flag)} of {len(rows)} notes disagree with their record ({100 * len(flag) / len(rows):.1f}%)")
(OUT / f"guard_{judge}.json").write_text(json.dumps(rows, indent=1))
for r in flag:
    print(f"\n[{r['i']}] {r['variant']} run{r['run']} {r['image']} | {'; '.join(r['disagree'])}")
    print("   kept:", [(k["id"], k["severity"], k["confidence"], "UNCERTAIN" if k["uncertain"] else "ok") for k in r["kept"]])
    print("   note:", r["note"])
