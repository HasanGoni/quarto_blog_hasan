"""Context-engineering post: Google ADK with its OWN file tools (EnvironmentToolset: Execute, ReadFile, EditFile, WriteFile).

    .venv-adk/bin/python ctx_adk_env.py [T1 ...]

ctx_adk.py wraps my harness functions as ADK tools. This arm is what an ADK user gets out of the box instead: a real shell
(`Execute`, so grep is whatever the model writes in bash) and `ReadFile` with start_line/end_line and a 30,000-character
truncation. Each run gets a throwaway copy of the archive as its working directory, because the shell can write.
Everything else (model, questions, scoring, 40-call cap) is the same as ctx_adk.py. The toolset is marked experimental in ADK 2.13.
"""
import asyncio
import json
import shlex
import shutil
import sys
import tempfile
import time
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.environment import LocalEnvironment
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.environment import EnvironmentToolset
from google.genai import types

import ctx_adk as a
import ctx_harness as h
from ctx_corpus import ARCHIVE
from ctx_eval import build_tasks, score

OUT = Path(__file__).parent / "out" / "ctx_adk_env_runs.json"


class SandboxedEnvironment(LocalEnvironment):
    """ADK's LocalEnvironment runs `Execute` as a plain shell with my user's rights and a copy of my environment variables
    (API keys included). Every command runs in a throwaway Alpine container instead: no network, read-only root, all
    capabilities dropped, memory and process limits, only the archive copy mounted (at /work, writable), no environment
    variables passed. The shell is therefore busybox `sh`, not bash. (bubblewrap was tried first and cannot set up its network
    namespace on this machine. A first, unsandboxed version of this arm was stopped and discarded; its partial runs are
    out/old_ctx_adk_env_unsandboxed_runs.json.)"""

    async def execute(self, command: str, *, timeout: float | None = None):
        argv = ["docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--pids-limit", "64", "--memory", "512m", "-u", "1000:1000",
                "-v", f"{self.working_dir}:/work", "-w", "/work", "-e", "HOME=/work", "alpine:latest", "sh", "-c", command]
        return await super().execute(shlex.join(argv), timeout=timeout)


async def ask(question: str, usage: a.Usage, work: Path):
    toolset = EnvironmentToolset(environment=SandboxedEnvironment(working_dir=work))
    agent = LlmAgent(name="archive_agent", model=a.make_model(), instruction=h.SYSTEM,
                     tools=[toolset, a.search_reports, a.similar_cases], after_model_callback=usage.after_model)
    svc = InMemorySessionService()
    runner = Runner(agent=agent, app_name="ctx", session_service=svc)
    sess = await svc.create_session(app_name="ctx", user_id="u")
    answer, used = None, {}
    async for ev in runner.run_async(user_id="u", session_id=sess.id, run_config=RunConfig(max_llm_calls=a.CALL_LIMIT),
                                     new_message=types.Content(role="user", parts=[types.Part(text=question)])):
        for fc in ev.get_function_calls() or []:
            used[fc.name] = used.get(fc.name, 0) + 1
        if ev.is_final_response() and ev.content and ev.content.parts:
            answer = "".join(p.text or "" for p in ev.content.parts)
    return answer, used


def run(question: str) -> dict:
    m = {"strategy": "adk-env", "question": question, "status": "ok", "answer": None}
    usage, t0 = a.Usage(), time.time()
    root = Path(tempfile.mkdtemp(prefix="ctx_adkenv_"))
    shutil.copytree(ARCHIVE, root / "a", ignore=shutil.ignore_patterns("large_tool_results", "_*"))
    try:
        m["answer"], used = asyncio.run(ask(question, usage, (root / "a").resolve()))
        m["tools_used"], m["tool_calls"] = used, sum(used.values())
        if not m["answer"] or "ANSWER" not in m["answer"]:
            m["status"] = "no_answer"
    except Exception as e:
        m["status"] = f"error: {type(e).__name__}: {str(e)[:160]}"
        m.update(tools_used={}, tool_calls=0)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    m.update(seconds=round(time.time() - t0, 1), model_calls_all=usage.calls, peak_prompt_tokens=usage.peak_in,
             total_prompt_tokens=usage.total_in, completion_tokens=usage.total_out)
    return m


def main():
    want_t = [x for x in sys.argv[1:] if x in ("T1", "T2", "T3", "T4", "T5")]
    runs = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {r["question"] for r in runs}
    for t in build_tasks():
        if (want_t and t["task"] not in want_t) or t["q"] in done:
            continue
        m = run(t["q"])
        m["task"], m["parsed"] = t["task"], h.parse_answer(m["answer"] if isinstance(m["answer"], str) else None)
        m["correct"], m["gold"] = score(t, m["parsed"]), t.get("gold", t.get("gold_ids"))
        runs.append(m)
        OUT.write_text(json.dumps(runs, indent=1))
        print(f"{t['task']} adk-env  {m['status'][:34]:34s} correct={m['correct']!s:5} calls={m['model_calls_all']:3d} "
              f"peak={m['peak_prompt_tokens']:6d} total={m['total_prompt_tokens']:7d} {m['seconds']:6.1f}s "
              f"tools={m.get('tools_used')}", flush=True)


if __name__ == "__main__":
    main()
