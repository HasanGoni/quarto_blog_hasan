"""Context-engineering post: the same tasks through the real Deep Agents library, to see whether anything changes.

    uv run python ctx_deepagents.py            # T1-T5 x 2 arms, resumes from out/ctx_da_runs.json
    uv run python ctx_deepagents.py T1 da      # only some tasks / arms

Same model (local Qwen3.8-27B, temperature 0, thinking off), same questions, same exact-answer scoring as ctx_eval.py.
What is different from my hand-written harness is everything the library brings: its built-in ls/read_file/glob/grep tools,
its planning tool, its offloading (20,000 tokens, its default), its summarisation middleware, and its `task` sub-agent tool.

Arms
    da        create_deep_agent(...) with my search tools (paths only) and nothing else changed. The library's default
              general-purpose sub-agent is still available through `task`.
    da+chunk  the same plus the article's `chunk_analyst` sub-agent (reads ONE file, returns key facts + source).

Each run gets a throwaway copy of the archive as its filesystem root, because offloading writes files.
The step cap of my own harness (14 model calls) is NOT applied here; the number of model calls is recorded instead, so the
analysis can report both "correct" and "correct within 14 calls".
"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import BadRequestError

from ctx_corpus import ARCHIVE
from ctx_eval import build_tasks, score
from ctx_harness import ANALYST_SYSTEM, MAX_OUT_TOKENS, MODEL, NO_THINK, SYSTEM, index, parse_answer

OUT = Path(__file__).parent / "out" / "ctx_da_runs.json"
CALL_LIMIT = 40          # a safety net against loops, far above the 14 of my own harness
WINDOW = 65536


class Usage(BaseCallbackHandler):
    """Counts every model call in the run, including the ones inside sub-agents."""

    def __init__(self):
        self.calls = self.total_in = self.total_out = self.peak_in = 0

    def on_llm_end(self, response, **kw):
        for gens in response.generations:
            for g in gens:
                um = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.calls += 1
                self.total_in += um.get("input_tokens", 0)
                self.total_out += um.get("output_tokens", 0)
                self.peak_in = max(self.peak_in, um.get("input_tokens", 0))


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL, base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"],
                      temperature=0, max_tokens=MAX_OUT_TOKENS, extra_body=NO_THINK,
                      profile={"max_input_tokens": WINDOW})   # tell the library how big the window really is


@tool
def search_reports(query: str) -> str:
    """Semantic (by-meaning) search over the inspection reports. Returns the 4 best matches as file paths."""
    return "\n".join("/reports/" + h.name for h in index().search(query, k=4))


@tool
def similar_cases(image_id: str, k: int = 4) -> str:
    """Visual search: the k test images that look most like the given one (DINOv3 embeddings).
    Returns file paths under /cases/, one record per image."""
    from ctx_visual import neighbours
    return "\n".join(f"/cases/{i}.md" for i in neighbours(image_id, int(k)))


CHUNK_ANALYST = {
    "name": "chunk_analyst",
    "description": "Reads ONE file and returns only the facts that bear on the question, plus its source path. "
                   "Pass exactly one file path per call; several calls in one turn run in parallel.",
    "system_prompt": ANALYST_SYSTEM,
}


def build_agent(arm: str, root: Path):
    return create_deep_agent(
        model=make_llm(), tools=[search_reports, similar_cases], system_prompt=SYSTEM,
        subagents=[CHUNK_ANALYST] if arm == "da+chunk" else None,
        backend=FilesystemBackend(root_dir=root, virtual_mode=True),
        middleware=[ModelCallLimitMiddleware(run_limit=CALL_LIMIT, exit_behavior="end")])


def run(question: str, arm: str) -> dict:
    m = {"strategy": arm, "question": question, "status": "ok", "answer": None}
    usage = Usage()
    root = Path(tempfile.mkdtemp(prefix="ctx_da_"))
    shutil.copytree(ARCHIVE, root / "a", ignore=shutil.ignore_patterns("large_tool_results", "_*"))
    t0 = time.time()
    try:
        agent = build_agent(arm, root / "a")
        res = agent.invoke({"messages": [{"role": "user", "content": question}]},
                           config={"callbacks": [usage], "recursion_limit": 4 * CALL_LIMIT})
        msgs = res["messages"]
        ai = [x for x in msgs if x.type == "ai"]
        m["answer"] = next((x.content for x in reversed(ai) if isinstance(x.content, str) and "ANSWER" in x.content),
                           ai[-1].content if ai else None)
        used: dict[str, int] = {}
        for x in ai:
            for tc in getattr(x, "tool_calls", []) or []:
                used[tc["name"]] = used.get(tc["name"], 0) + 1
        m["tools_used"], m["tool_calls"], m["steps"] = used, sum(used.values()), len(ai)
        if m["answer"] is None or "ANSWER" not in str(m["answer"]):
            m["status"] = "no_answer"
    except BadRequestError as e:
        m["status"] = "overflow" if "context" in str(e).lower() or "maximum" in str(e).lower() else f"error: {str(e)[:200]}"
        m.update(tools_used={}, tool_calls=0, steps=0)
    except Exception as e:  # keep the run, record why
        m["status"] = f"error: {type(e).__name__}: {str(e)[:200]}"
        m.update(tools_used={}, tool_calls=0, steps=0)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    m.update(seconds=round(time.time() - t0, 1), model_calls_all=usage.calls, peak_prompt_tokens=usage.peak_in,
             total_prompt_tokens=usage.total_in, completion_tokens=usage.total_out, sub_tokens=0)
    return m


def main():
    want_t = [a for a in sys.argv[1:] if a in ("T1", "T2", "T3", "T4", "T5")]
    want_a = [a for a in sys.argv[1:] if a in ("da", "da+chunk")]
    runs = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r["question"], r["strategy"]) for r in runs}
    for t in build_tasks():
        if want_t and t["task"] not in want_t:
            continue
        for arm in ("da", "da+chunk"):
            if (want_a and arm not in want_a) or (t["q"], arm) in done:
                continue
            m = run(t["q"], arm)
            m["task"], m["parsed"] = t["task"], parse_answer(m["answer"] if isinstance(m["answer"], str) else None)
            m["correct"] = score(t, m["parsed"])
            m["gold"] = t.get("gold", t.get("gold_ids"))
            runs.append(m)
            OUT.write_text(json.dumps(runs, indent=1))
            print(f"{t['task']} {arm:9s} {m['status'][:28]:28s} correct={m['correct']!s:5} steps={m['steps']:3d} "
                  f"all_calls={m['model_calls_all']:3d} peak={m['peak_prompt_tokens']:6d} total={m['total_prompt_tokens']:7d} "
                  f"{m['seconds']:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
