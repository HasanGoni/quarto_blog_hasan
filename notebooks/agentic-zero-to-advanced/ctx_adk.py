"""Context-engineering post: the same tasks through Google ADK (google-adk), to compare with the hand-written harness and Deep Agents.

    .venv-adk/bin/python ctx_adk.py            # T1-T5 x 2 arms, resumes from out/ctx_adk_runs.json
    .venv-adk/bin/python ctx_adk.py T1 adk     # only some tasks / arms

Same model (local Qwen3.8-27B through LiteLLM's OpenAI route, temperature 0, thinking off), same questions, same scoring.
ADK ships NO filesystem tools and NO offloading, so the tools are my own harness functions wrapped as plain Python functions
(that is how ADK takes tools: name, type hints and docstring become the schema). Arms:
    adk        one LlmAgent with ls/glob/grep/read_file/search_reports/similar_cases
    adk+chunk  the same, plus a `chunk_analyst` LlmAgent exposed as a tool (AgentTool): reads ONE file, returns key facts
The call cap is 40 model calls (RunConfig.max_llm_calls), the same safety net as ctx_deepagents.py.
"""
import asyncio
import functools
import json
import os
import sys
import time
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

import ctx_harness as h
from ctx_eval import build_tasks, score

OUT = Path(__file__).parent / "out" / "ctx_adk_runs.json"
CALL_LIMIT = 40


class Usage:
    def __init__(self):
        self.calls = self.total_in = self.total_out = self.peak_in = 0

    def after_model(self, callback_context, llm_response):
        um = llm_response.usage_metadata
        if um:
            self.calls += 1
            self.total_in += um.prompt_token_count or 0
            self.total_out += um.candidates_token_count or 0
            self.peak_in = max(self.peak_in, um.prompt_token_count or 0)
        return None


def make_model() -> LiteLlm:
    return LiteLlm(model=f"openai/{h.MODEL}", api_base=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"],
                   temperature=0, max_tokens=h.MAX_OUT_TOKENS, extra_body=h.NO_THINK)


# --- the tools: plain functions, the docstring is the schema description ---------------------------
def as_text_error(fn):
    """Like my harness: a bad path or regex goes back to the model as text instead of ending the run
    (the first ADK pass let a FileNotFoundError kill one run; that run was redone)."""
    @functools.wraps(fn)
    def inner(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:
            return f"error: {e}"
    return inner


def ls(path: str = ".") -> str:
    """List a directory of the archive (exports/, reports/, policy/)."""
    return h.t_ls(path)


def glob(pattern: str) -> str:
    """Find files by name pattern, e.g. reports/test_0001*.md"""
    return h.t_glob(pattern)


def grep(pattern: str, path: str = ".", mode: str = "files") -> str:
    """Search file CONTENTS with a regular expression without opening the files.
    mode: 'files' (matching file paths, default), 'count' (how many), 'lines' (path:line:text)."""
    return h.t_grep(pattern, path, mode)


def read_file(path: str, offset: int = 0, limit: int = 100) -> str:
    """Read a file. Returns up to 100 lines starting at `offset`; page through with offset."""
    return h.t_read_file(path, offset, limit, paged=True)


def search_reports(query: str) -> str:
    """Semantic (by-meaning) search over the inspection reports. Returns the 4 best matches as file paths."""
    return h.t_search_reports(query, paths=True)


def similar_cases(image_id: str, k: int = 4) -> str:
    """Visual search: the k test images that look most like the given one (DINOv3 embeddings).
    Returns file paths under cases/, one record per image."""
    from ctx_visual import neighbours
    return "\n".join(f"cases/{i}.md" for i in neighbours(image_id, int(k)))


TOOLS = [as_text_error(f) for f in (ls, glob, grep, read_file, search_reports, similar_cases)]


def build_agent(arm: str, usage: Usage) -> LlmAgent:
    tools = list(TOOLS)
    if arm == "adk+chunk":
        analyst = LlmAgent(name="chunk_analyst", model=make_model(), instruction=h.ANALYST_SYSTEM, tools=[read_file],
                           description="Reads ONE file and returns only the facts that bear on the question, plus its "
                                       "source path. Pass exactly one file path per call.",
                           after_model_callback=usage.after_model)
        tools.append(AgentTool(agent=analyst))
    return LlmAgent(name="archive_agent", model=make_model(), instruction=h.SYSTEM, tools=tools,
                    after_model_callback=usage.after_model)


async def ask(question: str, arm: str, usage: Usage) -> tuple[str | None, dict]:
    agent = build_agent(arm, usage)
    svc = InMemorySessionService()
    runner = Runner(agent=agent, app_name="ctx", session_service=svc)
    sess = await svc.create_session(app_name="ctx", user_id="u")
    answer, used = None, {}
    async for ev in runner.run_async(user_id="u", session_id=sess.id, run_config=RunConfig(max_llm_calls=CALL_LIMIT),
                                     new_message=types.Content(role="user", parts=[types.Part(text=question)])):
        for fc in ev.get_function_calls() or []:
            used[fc.name] = used.get(fc.name, 0) + 1
        if ev.is_final_response() and ev.content and ev.content.parts:
            answer = "".join(p.text or "" for p in ev.content.parts)
    return answer, used


def run(question: str, arm: str) -> dict:
    m = {"strategy": arm, "question": question, "status": "ok", "answer": None}
    usage, t0 = Usage(), time.time()
    try:
        m["answer"], used = asyncio.run(ask(question, arm, usage))
        m["tools_used"], m["tool_calls"] = used, sum(used.values())
        if not m["answer"] or "ANSWER" not in m["answer"]:
            m["status"] = "no_answer"
    except Exception as e:
        m["status"] = f"error: {type(e).__name__}: {str(e)[:160]}"
        m.update(tools_used={}, tool_calls=0)
    m.update(seconds=round(time.time() - t0, 1), model_calls_all=usage.calls, peak_prompt_tokens=usage.peak_in,
             total_prompt_tokens=usage.total_in, completion_tokens=usage.total_out)
    return m


def main():
    want_t = [a for a in sys.argv[1:] if a in ("T1", "T2", "T3", "T4", "T5")]
    want_a = [a for a in sys.argv[1:] if a in ("adk", "adk+chunk")]
    runs = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r["question"], r["strategy"]) for r in runs}
    for t in build_tasks():
        if want_t and t["task"] not in want_t:
            continue
        for arm in ("adk", "adk+chunk"):
            if (want_a and arm not in want_a) or (t["q"], arm) in done:
                continue
            m = run(t["q"], arm)
            m["task"], m["parsed"] = t["task"], h.parse_answer(m["answer"] if isinstance(m["answer"], str) else None)
            m["correct"] = score(t, m["parsed"])
            m["gold"] = t.get("gold", t.get("gold_ids"))
            runs.append(m)
            OUT.write_text(json.dumps(runs, indent=1))
            print(f"{t['task']} {arm:9s} {m['status'][:34]:34s} correct={m['correct']!s:5} calls={m['model_calls_all']:3d} "
                  f"peak={m['peak_prompt_tokens']:6d} total={m['total_prompt_tokens']:7d} {m['seconds']:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
