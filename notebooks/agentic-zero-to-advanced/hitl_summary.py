"""Part 9: every number in the post comes from this script (reads the saved Part 7/8 runs, applies the human step, writes out/hitl_summary.json)."""
import json
import statistics as st

import hitl_budget as hb
import hitl_eval as ev
from hitl_graph import POLICIES

ERRORS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
SEEDS = range(5)


def mean(xs):
    return round(st.mean(xs), 3)


def main():
    out = {"seconds_per_image": ev.SECONDS_PER_IMAGE, "n_images": len(ev.TEST)}
    saved = {"band": [ev.load_saved("band")], "llm": [ev.load_saved("llm", i) for i in range(3)]}
    # verdict mix and what is really inside each verdict (careful reviewer with no mistakes only needs the truth)
    mix = {}
    for v, runs in saved.items():
        m = {k: 0 for k in ("pass", "review", "escalate")}
        for r in runs[0]:
            m[r["verdict"]] += 1
        mix[v] = m
    out["verdict_mix"] = mix
    out["policies"] = {}
    for v, runs in saved.items():
        for pol in POLICIES:
            S = [ev.simulate(r, pol, ev.Reviewer())["summary"] for r in runs]
            out["policies"][f"{v}/{pol}"] = {k: mean(s[k] for s in S) for k in
                                           ("sent_to_human", "human_minutes", "boxes_shown", "tp", "fp", "fn", "precision", "recall", "ceiling_recall",
                                            "passes_audited", "passes_rescued")}
    # reviewer quality sweep, llm run 0..2 averaged over seeds
    sweep = {}
    for scope in ("all", "kept"):
        for e in ERRORS:
            S = [ev.simulate(r, "all_nonpass", ev.Reviewer(error_rate=e, seed=s, scope=scope))["summary"] for r in saved["llm"] for s in SEEDS]
            sweep[f"{scope}/{e}"] = {"precision": mean(s["precision"] for s in S), "recall": mean(s["recall"] for s in S)}
    out["noise_sweep"] = sweep
    stamp = ev.simulate(saved["llm"][0], "all_nonpass", ev.Reviewer(rubber_stamp=True))["summary"]
    out["rubber_stamp"] = {k: stamp[k] for k in ("precision", "recall", "sent_to_human", "human_minutes")}
    # deferral curves
    out["deferral"] = [hb.summary_row(hb.curve(n, t)) for n, t in hb.JUDGES]
    (ev.OUT / "hitl_summary.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    o = main()
    print("verdict mix", o["verdict_mix"])
    print(f"{'run/policy':24s} {'sent':>6s} {'min':>6s} {'P':>6s} {'R':>6s} {'ceil':>6s} {'fp':>4s} {'audited':>8s} {'rescued':>8s}")
    for k, v in o["policies"].items():
        print(f"{k:24s} {v['sent_to_human']:6.1f} {v['human_minutes']:6.1f} {v['precision']:6.3f} {v['recall']:6.3f} {v['ceiling_recall']:6.3f} {v['fp']:4.0f} {v['passes_audited']:8.1f} {v['passes_rescued']:8.1f}")
    print("\nreviewer error sweep (all_nonpass, llm runs)       scope=all: P / R      scope=kept: P / R")
    for e in ERRORS:
        a, k = o["noise_sweep"][f"all/{e}"], o["noise_sweep"][f"kept/{e}"]
        print(f"  error {e:4.2f}                                       {a['precision']:.3f} / {a['recall']:.3f}        {k['precision']:.3f} / {k['recall']:.3f}")
    print("rubber stamp:", o["rubber_stamp"])
    print("\ndeferral (errors fixed after the person re-checks the k% least sure boxes, vs random)")
    for r in o["deferral"]:
        print(" ", r)
