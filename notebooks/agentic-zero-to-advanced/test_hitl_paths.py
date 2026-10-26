"""Part 9: every path of the human-in-the-loop graph and its queue, offline (classifier "band", template reports, no model call).

    uv run python test_hitl_paths.py
"""
import subprocess
import sys
import tempfile

from langgraph.checkpoint.memory import MemorySaver

import hitl_eval as ev
import hitl_graph as h
from hitl_queue import ReviewQueue

OK = []


def check(name, cond, detail=""):
    assert cond, f"FAILED: {name} {detail}"
    OK.append(name)
    print(f"ok  {name}")


def first_waiting(g, policy_images):
    for im in policy_images:
        r = h.start(g, im, im)
        if r["status"] == "waiting":
            return im, r
    raise AssertionError("no image paused")


g = h.build_hitl("band", "all_nonpass", checkpointer=MemorySaver())
imgs = ev.TEST[:40]

# 1. approve / reject / edit
im, r = first_waiting(g, imgs)
kept_before = [c["id"] for c in r["payload"]["candidates"] if c["pipeline"].startswith("kept")]
d = h.resume(g, im, {"action": "approve", "reviewer": "ana"})["state"]
check("approve keeps the pipeline's boxes, verdict fail, decided_by human",
      d["verdict"] == "fail" and d["decided_by"] == "human" and [k["id"] for k in d["kept"]] == kept_before)
im2, r2 = first_waiting(g, [i for i in imgs if i != im])
d = h.resume(g, im2, {"action": "reject", "reviewer": "bo", "note": "dust"})["state"]
check("reject clears everything, verdict pass, the note is in the reason", d["verdict"] == "pass" and d["kept"] == [] and "dust" in d["reason"])
im3, r3 = first_waiting(g, [i for i in imgs if i not in (im, im2)])
ids = r3["payload"]["all_candidate_ids"]
d = h.resume(g, im3, {"action": "edit", "reviewer": "cy", "keep_ids": ids[:1]})["state"]
check("edit keeps exactly the given ids", [k["id"] for k in d["kept"]] == ids[:1])

# 2. bad answers are re-asked, never guessed, and logged
im4, r4 = first_waiting(g, [i for i in imgs if i not in (im, im2, im3)])
bad = ["approve", {"action": "maybe", "reviewer": "x"}, {"action": "approve"}, {"action": "approve", "reviewer": "  "},
       {"action": "edit", "reviewer": "x"}, {"action": "edit", "reviewer": "x", "keep_ids": [999]},
       {"action": "approve", "reviewer": "x", "keep_ids": [0]}]
for b in bad:
    r = h.resume(g, im4, b)
    assert r["status"] == "waiting" and "previous_answer_rejected" in r["payload"], b
check(f"{len(bad)} malformed answers are each refused and the question is asked again", True)
r = h.resume(g, im4, {"action": "reject", "reviewer": "dee"})
check("the audit trail keeps every refused answer and the final decision",
      [a["event"] for a in r["state"]["audit"]] == ["answer rejected"] * len(bad) + ["decided"])

# 3. a reviewer can rescue a box the pipeline threw away
rescued = None
for im5 in imgs:
    st = h.start(g, im5, "x" + im5)
    if st["status"] == "waiting":
        rej = [c["id"] for c in st["payload"]["candidates"] if c["pipeline"].startswith("rejected")]
        if rej:
            d = h.resume(g, "x" + im5, {"action": "edit", "reviewer": "ed", "keep_ids": rej[:1]})["state"]
            rescued = d
            break
check("edit can add a rejected box; it is flagged added_by_human", rescued and rescued["kept"][0].get("added_by_human") is True)

# 4. who pauses
n_wait = {}
for pol in h.POLICIES:
    gp = h.build_hitl("band", pol, checkpointer=MemorySaver())
    n_wait[pol] = sum(h.start(gp, i, i)["status"] == "waiting" for i in ev.TEST[:60])
check("policy none never pauses", n_wait["none"] == 0, n_wait)
check("escalate_only pauses less than all_nonpass", n_wait["escalate_only"] < n_wait["all_nonpass"], n_wait)
check("the audit policy pauses on some images that the pipeline passed", n_wait["all_plus_audit"] > n_wait["all_nonpass"], n_wait)
gn = h.build_hitl("band", "none", checkpointer=MemorySaver())
s = h.start(gn, ev.TEST[2], "n2")["state"]
check("without a human a 'review' verdict becomes fail by itself (decided_by auto-fail)", s["verdict"] == "fail" and s["decided_by"] == "auto-fail", s["decided_by"])

# 5. resume re-runs the node body from the top
h.REPLAYS.clear()
gr = h.build_hitl("band", "all_nonpass", checkpointer=MemorySaver())
imr, _ = first_waiting(gr, imgs)
h.REPLAYS.clear()
h.resume(gr, imr, {"action": "approve"})            # refused (no reviewer) -> asks again
h.resume(gr, imr, {"action": "approve", "reviewer": "ana"})
check("each resume starts the review node again from its first line", len(h.REPLAYS) == 2, h.REPLAYS)

# 6. durable: another process resumes what this one started
db = tempfile.mktemp(suffix=".db")
q = ReviewQueue(db)
for i in imgs[:12]:
    q.submit(i)
tid = q.pending()[0]["thread_id"]
code = (f"import sys; sys.path.insert(0, '.'); from hitl_queue import ReviewQueue; q = ReviewQueue({db!r}); "
        f"r = q.answer({tid!r}, {{'action': 'approve', 'reviewer': 'ana'}}); print(r['status'], r['state']['decided_by'])")
out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.split()
check("a brand-new process finishes a review started in this one", out == ["done", "human"], out)

# 7. queue discipline
try:
    q.answer(tid, {"action": "reject", "reviewer": "bo"})
    check("a second answer to a closed item is refused", False)
except RuntimeError:
    check("a second answer to a closed item is refused", True)
try:
    q.answer("insp-nope", {"action": "approve", "reviewer": "x"})
    check("an unknown item raises KeyError", False)
except KeyError:
    check("an unknown item raises KeyError", True)
check("submitting the same image twice does not run it twice", "already submitted" in q.submit(imgs[0]).get("note", ""))
n_wait_q = len(q.pending())
import time
q.clock = lambda: time.time() + 7200
held = q.expire(3600)
check("expire marks unanswered items 'held', never 'pass'", len(held) == n_wait_q and q.counts().get("held") == n_wait_q and "pass" not in
      [v for (v,) in q.conn.execute("SELECT verdict FROM review_queue WHERE status='held'")])

# 8. the real graph and the simulation on a saved run agree
band = ev.load_saved("band")
for rev in (ev.Reviewer(), ev.Reviewer(error_rate=0.1, seed=3), ev.Reviewer(scope="kept", error_rate=0.1, seed=3), ev.Reviewer(rubber_stamp=True)):
    a = ev.run_graph("all_nonpass", rev)["rows"]
    b = ev.simulate(band, "all_nonpass", rev)["rows"]
    same = [sorted(x["final"]) for x in a] == [sorted(x["final"]) for x in b]
    check(f"graph == simulation (reviewer error {rev.error_rate}, scope {rev.scope}, rubber_stamp {rev.rubber_stamp})", same)

# 9. a rubber-stamp reviewer is worth exactly nothing
llm = ev.load_saved("llm")
none = ev.simulate(llm, "none", ev.Reviewer())["summary"]
stamp = ev.simulate(llm, "all_nonpass", ev.Reviewer(rubber_stamp=True))["summary"]
check("rubber stamping gives the same tp/fp/fn as no human at all", (none["tp"], none["fp"], none["fn"]) == (stamp["tp"], stamp["fp"], stamp["fn"]))
print(f"\nALL HITL PATHS OK ({len(OK)} checks)")
