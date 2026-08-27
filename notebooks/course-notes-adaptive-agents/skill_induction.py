"""Skill Induction demo — open-source reproduction of Lesson 2.

Course used Oracle Agent Memory + Oracle DB for trace storage and an
unspecified induction-engine LLM. Here: SQLite (stdlib, one file) for traces
and skill versions, and a **local, self-hosted open-source LLM** as the
induction engine — any OpenAI-compatible local server works (vLLM, Ollama,
LM Studio, text-generation-webui): no API key, no cloud cost, no data leaving
the machine.

Run (defaults to a local vLLM/Ollama-style server at localhost:8000):
    uv run skill_induction.py
    uv run skill_induction.py --base-url http://localhost:11434/v1 --model qwen2.5:3b   # Ollama
"""
from __future__ import annotations

import argparse
import sqlite3
import textwrap
from pathlib import Path

import requests

DB_PATH = Path("traces.db")

SKILL_V1 = textwrap.dedent("""\
    # run-the-tests (v1)

    1. Run `pytest -q` from the repo root.
    2. If it fails, read the traceback and fix the failing assertion.
    3. Re-run until green.
    """)

# Three episodes of the SAME recurring failure: a fixture that isn't
# session-scoped, so every test file that uses it re-creates an expensive
# resource and eventually times out in CI (but not locally, which is why it
# keeps getting "fixed" the same way and keeps coming back).
TRACES = [
    dict(
        topic="run_test_suite",
        task="Run the test suite before merging the retriever PR.",
        attempt="Ran `pytest -q`. tests/test_retriever.py::test_bulk_query timed out in CI (passed locally).",
        outcome="failed",
        fix="Root cause: `db_conn` fixture was function-scoped, so 40 tests each opened a fresh "
            "connection; CI's slower disk made this exceed the timeout. Changed `@pytest.fixture` "
            "to `@pytest.fixture(scope='session')` for db_conn in conftest.py.",
    ),
    dict(
        topic="run_test_suite",
        task="Run the test suite after adding the co-edit miner tests.",
        attempt="Ran `pytest -q`. tests/test_coedit.py timed out in CI, same symptom as last time.",
        outcome="failed",
        fix="Same root cause as before: a new `git_repo` fixture was function-scoped again. Scoped "
            "it to session in conftest.py; CI run dropped from timeout to 4s.",
    ),
    dict(
        topic="run_test_suite",
        task="Run the test suite after the PageRank anchoring change.",
        attempt="Ran `pytest -q`. Green in 6s, no timeout.",
        outcome="passed",
        fix="Checked conftest.py first this time for any new function-scoped fixture wrapping a "
            "connection/expensive resource before running the suite — none found, so no fix needed.",
    ),
]


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS traces (id INTEGER PRIMARY KEY, topic TEXT, task TEXT, attempt TEXT, outcome TEXT, fix TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS skills (id INTEGER PRIMARY KEY, name TEXT, version INTEGER, content TEXT, active INTEGER)")
    conn.commit()
    return conn


def seed(conn: sqlite3.Connection):
    conn.execute("DELETE FROM traces")
    conn.execute("DELETE FROM skills")
    for t in TRACES:
        conn.execute(
            "INSERT INTO traces (topic, task, attempt, outcome, fix) VALUES (?, ?, ?, ?, ?)",
            (t["topic"], t["task"], t["attempt"], t["outcome"], t["fix"]),
        )
    conn.execute("INSERT INTO skills (name, version, content, active) VALUES (?, 1, ?, 1)", ("run-the-tests", SKILL_V1))
    conn.commit()


def search_skill_box(conn: sqlite3.Connection, name: str) -> str:
    row = conn.execute(
        "SELECT version, content FROM skills WHERE name = ? AND active = 1 ORDER BY version DESC LIMIT 1",
        (name,),
    ).fetchone()
    return f"[retrieved v{row[0]}]\n{row[1]}" if row else "[no skill found]"


def induce(conn: sqlite3.Connection, topic: str, current_skill: str, base_url: str, model: str) -> str:
    episodes = conn.execute(
        "SELECT task, attempt, outcome, fix FROM traces WHERE topic = ? ORDER BY id", (topic,)
    ).fetchall()

    episode_text = "\n\n".join(
        f"Episode {i+1} ({outcome}):\nTask: {task}\nAttempt: {attempt}\nFix/notes: {fix}"
        for i, (task, attempt, outcome, fix) in enumerate(episodes)
    )

    system = (
        "You are a Skill Induction Engine. You review an agent's past episodes on a recurring "
        "task and propose an improved version of its skill file. Only add instructions that are "
        "directly supported by evidence in the episodes below. Output ONLY the revised skill "
        "markdown (same '# title (vN)' header style, increment the version number), nothing else."
    )
    user = (
        f"Current skill:\n{current_skill}\n\n"
        f"Recent episodes for topic '{topic}':\n{episode_text}\n\n"
        "Propose the improved skill version."
    )

    resp = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": 300,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def review_and_promote(conn: sqlite3.Connection, name: str, proposed_v2: str) -> bool:
    # Human-in-the-loop gate: in a real Skill Box UI this renders a v1-vs-v2
    # diff for a person to approve/reject. Here we apply one automated
    # sanity check standing in for that human judgment call, and print the
    # diff either way so nothing is hidden from the reviewer.
    print("\n--- v1 (active) ---")
    print(SKILL_V1)
    print("--- proposed v2 (pending review) ---")
    print(proposed_v2)

    mentions_root_cause = "scope" in proposed_v2.lower() or "fixture" in proposed_v2.lower()
    if not mentions_root_cause:
        print("\n[REJECTED] proposal doesn't reference the actual root cause found in the traces "
              "(fixture scoping) — sending back to the induction engine, not promoting.")
        return False

    row = conn.execute("SELECT MAX(version) FROM skills WHERE name = ?", (name,)).fetchone()
    next_version = (row[0] or 0) + 1
    conn.execute("UPDATE skills SET active = 0 WHERE name = ?", (name,))
    conn.execute(
        "INSERT INTO skills (name, version, content, active) VALUES (?, ?, ?, 1)",
        (name, next_version, proposed_v2),
    )
    conn.commit()
    print(f"\n[APPROVED] promoted to v{next_version} and marked active.")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1",
                     help="Any OpenAI-compatible local server: vLLM, Ollama (:11434/v1), LM Studio, etc.")
    ap.add_argument("--model", default="qwen3-coder")
    args = ap.parse_args()

    conn = init_db()
    seed(conn)

    print("=== BEFORE: what the agent retrieves for 'run_test_suite' today ===")
    print(search_skill_box(conn, "run-the-tests"))

    proposal = induce(conn, "run_test_suite", SKILL_V1, args.base_url, args.model)
    review_and_promote(conn, "run-the-tests", proposal)

    print("\n=== AFTER: what the agent retrieves now ===")
    print(search_skill_box(conn, "run-the-tests"))
