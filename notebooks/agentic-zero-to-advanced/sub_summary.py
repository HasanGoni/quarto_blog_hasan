"""Part 10: every number in the post, computed from out/sub_runs.jsonl and written to out/sub_summary.json.

    uv run python sub_summary.py
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).parent / "out"
ARMS = ["single", "supervisor", "batched", "fanout", "panel3"]
BUCKETS = [("0", 0, 0), ("1-3", 1, 3), ("4-7", 4, 7), ("8+", 8, 99)]


def load() -> dict:
    """{arm: {run: {image: record}}}, ok records only."""
    d: dict = defaultdict(lambda: defaultdict(dict))
    lines = [ln for p in sorted(OUT.glob("sub_runs*.jsonl")) for ln in p.read_text().splitlines()]
    for r in map(json.loads, lines):
        if r.get("status") == "ok":
            d[r["arm"]][r["run"]][r["image"]] = r
    return d


def pr(tp, fp, fn):
    return (round(tp / (tp + fp), 3) if tp + fp else None, round(tp / (tp + fn), 3) if tp + fn else None)


def run_row(recs: list[dict]) -> dict:
    n = len(recs)
    tp, fp, fn = (sum(r[k] for r in recs) for k in ("tp", "fp", "fn"))
    p, rc = pr(tp, fp, fn)
    secs = sorted(r["seconds"] for r in recs)
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": rc,
            "avg_calls": round(sum(r["calls"] for r in recs) / n, 2),
            "avg_prompt_tokens": round(sum(r["prompt_tokens"] for r in recs) / n),
            "avg_completion_tokens": round(sum(r["completion_tokens"] for r in recs) / n),
            "avg_total_tokens": round(sum(r["prompt_tokens"] + r["completion_tokens"] for r in recs) / n),
            "avg_peak_prompt": round(sum(r["peak_prompt"] for r in recs) / n),
            "max_peak_prompt": max(r["peak_prompt"] for r in recs),
            "avg_seconds": round(sum(secs) / n, 1), "p95_seconds": secs[int(0.95 * (n - 1))],
            "finished": sum(r.get("finished", True) for r in recs),
            "judge_failed": sum(r.get("judge_failed", 0) for r in recs)}


def boot_diff(a: dict, b: dict, metric: str, reps: int = 2000, seed: int = 0) -> list[float]:
    """95% interval for metric(a) - metric(b), resampling IMAGES. a, b = {run: {image: rec}}; counts are averaged over runs."""
    images = sorted(set(next(iter(a.values()))) & set(next(iter(b.values()))))

    def per_image(x):
        return {im: tuple(sum(run[im][k] for run in x.values() if im in run) / sum(im in run for run in x.values()) for k in ("tp", "fp", "fn")) for im in images}

    pa, pb = per_image(a), per_image(b)
    rng, diffs = random.Random(seed), []
    for _ in range(reps):
        s = [images[rng.randrange(len(images))] for _ in images]
        va = pr(*[sum(pa[i][k] for i in s) for k in range(3)])
        vb = pr(*[sum(pb[i][k] for i in s) for k in range(3)])
        j = 0 if metric == "precision" else 1
        if va[j] is not None and vb[j] is not None:
            diffs.append(va[j] - vb[j])
    diffs.sort()
    return [round(diffs[int(0.025 * len(diffs))], 3), round(diffs[int(0.975 * len(diffs))], 3)]


def judge_defects(text: str) -> set[int]:
    """Ids the judge called defect, from its plain-text lines ('Cand 0: defect 0.7', 'C 2 defect 0.65', '1 false_alarm 0.9')."""
    return {int(m.group(1)) for ln in text.splitlines() if (m := re.search(r"(\d+)\D{0,14}?(defect|false_alarm)", ln, re.I)) and m.group(2).lower() == "defect"}


def boxes_carried(r: dict) -> tuple[int, int]:
    """(candidates, boxes whose four true coordinates appear in order in the brief). Only briefs the file kept whole (<1500 chars)."""
    from inspection_tools import find_candidates
    brief = r["brief"]
    cands = find_candidates(r["image"])["candidates"]
    pat = lambda b: r"\D+".join(str(v) for v in b)
    return len(cands), sum(bool(re.search(pat(c["box"]), brief)) for c in cands)


def summarize(d=None) -> dict:
    d = d or load()
    out = {"arms": {}, "buckets": {}, "diff_vs_batched": {}, "handoff": None, "agreement": {}}
    for arm in ARMS:
        if arm not in d:
            continue
        runs = {str(k): run_row(list(v.values())) for k, v in sorted(d[arm].items())}
        allrecs = [r for v in d[arm].values() for r in v.values()]
        pooled = run_row(allrecs)
        ps = [r["precision"] for r in runs.values() if r["precision"] is not None]
        rs = [r["recall"] for r in runs.values() if r["recall"] is not None]
        out["arms"][arm] = {"runs": runs, "pooled": pooled, "n_runs": len(runs),
                            "precision_range": [min(ps), max(ps)], "recall_range": [min(rs), max(rs)]}
        out["buckets"][arm] = {name: run_row([r for r in allrecs if lo <= r["n_candidates"] <= hi]) for name, lo, hi in BUCKETS
                               if any(lo <= r["n_candidates"] <= hi for r in allrecs)}
        out["arms"][arm]["by_role"] = {role: {k: round(sum(r["by_role"].get(role, {}).get(k, 0) for r in allrecs) / len(allrecs), 1)
                                              for k in ("calls", "prompt", "completion")}
                                       for role in sorted({ro for r in allrecs for ro in r["by_role"]})}
    if "batched" in d:
        # a yardstick for "stricter": if the batched judge's kept boxes were thinned at random down to the same number,
        # how many real boxes would remain? An arm that beats this is ranking boxes better, not only keeping fewer.
        b = out["arms"]["batched"]["pooled"]
        bn = out["arms"]["batched"]["n_runs"]
        b_tp, b_kept = b["tp"] / bn, (b["tp"] + b["fp"]) / bn
        for arm, a in out["arms"].items():
            p_, n_ = a["pooled"], a["n_runs"]
            kept = (p_["tp"] + p_["fp"]) / n_
            a["per_run"] = {"tp": round(p_["tp"] / n_, 1), "fp": round(p_["fp"] / n_, 1), "kept": round(kept, 1),
                            "tp_if_random_thinning_of_batched": round(b_tp * kept / b_kept, 1)}
    for arm in ARMS:
        if arm in d and "batched" in d and arm != "batched":
            out["diff_vs_batched"][arm] = {m: boot_diff(d[arm], d["batched"], m) for m in ("precision", "recall")}
    if "single" in d and "supervisor" in d:
        out["diff_vs_single"] = {m: boot_diff(d["supervisor"], d["single"], m) for m in ("precision", "recall")}
    if "supervisor" in d:
        recs = [r for r in d["supervisor"][0].values() if r.get("handoff")]
        c = sum(r["handoff"]["candidates"] for r in recs)
        k = sum(r["handoff"]["carried"] for r in recs)
        out["handoff"] = {"images_with_brief": len(recs), "candidates": c, "carried": k,
                          "images_complete": sum(r["handoff"]["carried"] == r["handoff"]["candidates"] for r in recs),
                          "images_no_judge_call": sum(1 for r in d["supervisor"][0].values() if not r.get("handoff")),
                          "kept_but_no_brief": sum(1 for r in d["supervisor"][0].values() if not r.get("handoff") and r["kept"])}
        sup = [r for r in d["supervisor"][0].values() if r.get("brief") and len(r["brief"]) < 1500]
        bc = [boxes_carried(r) for r in sup]
        out["handoff"]["box_coordinates"] = {"images": len(sup), "boxes": sum(c for c, _ in bc), "carried": sum(k for _, k in bc)}
        ju = [r for r in d["supervisor"][0].values() if r.get("judge_out")]
        out["handoff"]["followed_judge"] = {"images": len(ju), "kept_subset_of_judge_defects": sum(set(r["kept"]) <= judge_defects(r["judge_out"]) for r in ju),
                                            "judge_called_defect_but_supervisor_dropped": sum(bool(judge_defects(r["judge_out"]) - set(r["kept"])) for r in ju)}
        out["handoff"]["bad_ids_runs"] = sum(1 for r in d["supervisor"][0].values() if r.get("bad_ids"))
    # how often do two runs of the SAME arm give a different kept set on the same image? that is the noise floor
    for arm in ("batched", "fanout", "panel3"):
        if arm in d and len(d[arm]) > 1:
            r0, r1 = d[arm][0], d[arm][1]
            both = set(r0) & set(r1)
            out["agreement"][arm] = {"images": len(both), "differ": sum(sorted(r0[i]["kept"]) != sorted(r1[i]["kept"]) for i in both)}
    if "batched" in d:
        for arm in ARMS:
            if arm in d and arm != "batched":
                both = set(d[arm][0]) & set(d["batched"][0])
                out["agreement"][f"{arm}_vs_batched"] = {"images": len(both), "differ": sum(sorted(d[arm][0][i]["kept"]) != sorted(d["batched"][0][i]["kept"]) for i in both)}
    if "panel3" in d:
        recs = [r for v in d["panel3"].values() for r in v.values()]
        out["panel_split_boxes"] = sum(r.get("split_votes", 0) for r in recs)
        out["panel_boxes"] = sum(r["judge_units"] for r in recs) // 3
    return out


def main():
    s = summarize()
    (OUT / "sub_summary.json").write_text(json.dumps(s, indent=1))
    f = "{:<11}{:>5}{:>8}{:>8}{:>7}{:>9}{:>9}{:>8}{:>8}{:>8}{:>9}"
    print(f.format("arm", "runs", "prec", "recall", "calls", "tokens", "peak ctx", "sec/img", "p95 sec", "failed", "done"))
    for arm, a in s["arms"].items():
        p = a["pooled"]
        print(f.format(arm, a["n_runs"], p["precision"], p["recall"], p["avg_calls"], p["avg_total_tokens"], p["avg_peak_prompt"],
                       p["avg_seconds"], p["p95_seconds"], p["judge_failed"], f"{p['finished']}/{p['n']}"))
    print("\nprecision range over runs:", {a: v["precision_range"] for a, v in s["arms"].items()})
    print("recall range over runs:   ", {a: v["recall_range"] for a, v in s["arms"].items()})
    print("\n95% interval, difference vs batched (image bootstrap):", json.dumps(s["diff_vs_batched"]))
    print("supervisor minus single:", s.get("diff_vs_single"))
    print("hand-off:", s["handoff"])
    print("agreement:", json.dumps(s["agreement"]))
    for arm, b in s["buckets"].items():
        print(arm, {k: (v["n"], v["precision"], v["recall"], v["avg_peak_prompt"]) for k, v in b.items()})


if __name__ == "__main__":
    main()
