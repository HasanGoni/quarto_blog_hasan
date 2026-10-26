"""Part 9: a review queue on top of the human-in-the-loop graph. Two SQLite tables, one file:

    checkpoints/writes   LangGraph's own: the saved graph state, so a paused run survives a restart of the process
    review_queue         ours: who is waiting, since when, what they were shown, how it ended

The queue is the part a reviewer's tool talks to: `pending()` to list, `answer()` to decide, `expire()` for the ones nobody answered.
A pause that nobody answers must never turn into a pass: `expire()` marks the item "held" and leaves the part blocked.
"""
import json
import sqlite3
import time
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

import hitl_graph as h

SCHEMA = """CREATE TABLE IF NOT EXISTS review_queue (
    thread_id TEXT PRIMARY KEY, image TEXT, status TEXT, created REAL, closed REAL,
    payload TEXT, verdict TEXT, decided_by TEXT, reviewer TEXT)"""


class ReviewQueue:
    def __init__(self, db: str | Path, classifier: str = "band", policy: str = "all_nonpass", clock=time.time):
        self.db, self.clock = str(db), clock
        self.conn = sqlite3.connect(self.db, check_same_thread=False, isolation_level=None)
        self.conn.execute(SCHEMA)
        self.graph = h.build_hitl(classifier, policy, checkpointer=SqliteSaver(self.conn))

    def submit(self, image: str) -> dict:
        """Run one image. Finished images are logged; waiting ones join the queue. Submitting the same image twice reuses its thread."""
        tid = f"insp-{image}"
        row = self.conn.execute("SELECT status FROM review_queue WHERE thread_id=?", (tid,)).fetchone()
        if row:
            return {"status": row[0], "thread_id": tid, "note": "already submitted: not run again"}
        r = h.start(self.graph, image, tid)
        if r["status"] == "waiting":
            self.conn.execute("INSERT INTO review_queue VALUES (?,?,?,?,?,?,?,?,?)",
                              (tid, image, "waiting", self.clock(), None, json.dumps(r["payload"]), None, None, None))
        else:
            s = r["state"]
            self.conn.execute("INSERT INTO review_queue VALUES (?,?,?,?,?,?,?,?,?)",
                              (tid, image, "done", self.clock(), self.clock(), None, s["verdict"], s["decided_by"], None))
        return {**r, "thread_id": tid}

    def pending(self) -> list[dict]:
        rows = self.conn.execute("SELECT thread_id, image, created, payload FROM review_queue WHERE status='waiting' ORDER BY created").fetchall()
        return [{"thread_id": t, "image": i, "waiting_s": round(self.clock() - c, 1), "payload": json.loads(p)} for t, i, c, p in rows]

    def answer(self, thread_id: str, decision) -> dict:
        row = self.conn.execute("SELECT status FROM review_queue WHERE thread_id=?", (thread_id,)).fetchone()
        if row is None:
            raise KeyError(f"no such review item: {thread_id}")
        if row[0] != "waiting":                      # closed, held or expired: a second answer must not rewrite history
            raise RuntimeError(f"{thread_id} is '{row[0]}', not waiting: answer refused")
        r = h.resume(self.graph, thread_id, decision)
        if r["status"] == "waiting":                 # the answer was rejected by the graph and the question was asked again
            self.conn.execute("UPDATE review_queue SET payload=? WHERE thread_id=?", (json.dumps(r["payload"]), thread_id))
            return r
        s = r["state"]
        self.conn.execute("UPDATE review_queue SET status='done', closed=?, verdict=?, decided_by=?, reviewer=? WHERE thread_id=?",
                          (self.clock(), s["verdict"], s["decided_by"], (s.get("human") or {}).get("reviewer"), thread_id))
        return r

    def expire(self, older_than_s: float) -> list[str]:
        """Items nobody answered in time. They are HELD, never passed: the part stays blocked and someone is told."""
        cutoff = self.clock() - older_than_s
        ids = [t for (t,) in self.conn.execute("SELECT thread_id FROM review_queue WHERE status='waiting' AND created<?", (cutoff,))]
        for t in ids:
            self.conn.execute("UPDATE review_queue SET status='held', closed=?, verdict='hold', decided_by='nobody' WHERE thread_id=?", (self.clock(), t))
        return ids

    def counts(self) -> dict:
        return dict(self.conn.execute("SELECT status, COUNT(*) FROM review_queue GROUP BY status").fetchall())
