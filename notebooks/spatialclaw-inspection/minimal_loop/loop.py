"""The smallest SpatialClaw-style loop for image inspection. Written for this series from the paper's description.

What it keeps from the paper: a persistent kernel, one cell per step, feedback = stdout + errors + variable summary,
an explicit step budget, a hard exit on `submit()`.
What it leaves out ON PURPOSE (later parts): the planner, show() images, the reflection gate, a real sandbox.
What it changes for inspection: the agent submits a MEASUREMENT; a plain rule decides PASS/FAIL; running out of budget
returns REVIEW, never a guess.
"""
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

from pydantic import BaseModel

from kernel_session import KernelSession

LIMIT_PCT = 1.5   # ILLUSTRATIVE spec: FAIL when voids cover more than 1.5 % of the part. Not a real standard.

SYSTEM = """You are an inspection agent working in a persistent Python notebook.

Task: measure how much of the part is void in ONE grayscale X-ray image, then submit the number.

The notebook already has: `np` (numpy), `ndi` (scipy.ndimage), and `img` (2-D uint8 array, 224 x 224).
The image shows one part on a background. The part and the background have different brightness.
Voids are small, roundish regions INSIDE the part that look different from the part around them.
The image may also contain cracks: thin, elongated lines inside the part. Cracks are NOT voids, do not count them.

Definition: void_ratio_pct = 100 * (void pixels) / (part pixels), where the part pixels are the whole part
silhouette (voids inside the part count as part area).

Rules:
- Each turn: one short line starting with "Purpose:", then exactly ONE ```python code block. Nothing else.
- Variables you create stay alive between turns. Use print() to look at results; you only see what you print.
- Look at numbers before you trust them (sizes, counts, a few sample values), and sanity-check the result.
- When you are confident, call submit(void_ratio_pct, "one-line note"). You do NOT decide pass or fail.
- You have a limited number of turns. Do not waste them."""


class Verdict(BaseModel):
    status: Literal["PASS", "FAIL", "REVIEW"]
    void_ratio_pct: float | None
    limit_pct: float
    reason: str


def decide(submission: dict | None, limit: float, why_none: str) -> Verdict:
    """The rule layer: ordinary code, no model. No measurement means REVIEW. A guess never becomes PASS."""
    if submission is None:
        return Verdict(status="REVIEW", void_ratio_pct=None, limit_pct=limit, reason=why_none)
    v = submission["void_ratio_pct"]
    return Verdict(status="FAIL" if v > limit else "PASS", void_ratio_pct=v, limit_pct=limit,
                   reason=f"rule: {v:.2f} {'>' if v > limit else '<='} {limit:.2f}")


@dataclass
class Episode:
    verdict: Verdict
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cells: list = field(default_factory=list)      # full trace: purpose, code, stdout, error


ChatFn = Callable[[list], tuple]                   # messages -> (text, {"prompt": int, "completion": int})
_BLOCK = re.compile(r"```python\s*\n(.*?)```", re.S)


def _feedback(res, steps_left: int) -> str:
    parts = []
    out = res.stdout.strip()
    if out:
        parts.append("stdout:\n" + (out if len(out) <= 1500 else out[:1500] + "\n...[truncated]"))
    if res.error:
        parts.append("error:\n" + res.error)
    if res.variables:
        parts.append("variables: " + "; ".join(f"{k}: {v}" for k, v in list(res.variables.items())[:25]))
    if not parts:
        parts.append("(no output)")
    parts.append(f"turns left: {steps_left}")
    return "\n".join(parts)


def run_episode(image, question: str, chat: ChatFn, max_steps: int = 8, max_failures: int = 4,
                limit: float = LIMIT_PCT, cell_timeout: int = 60) -> Episode:
    t0 = time.monotonic()
    ep = Episode(verdict=decide(None, limit, "not started"))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    failures, session = 0, KernelSession(image, timeout_sec=cell_timeout)
    sub, why_none = None, "out_of_steps"
    try:
        for step in range(1, max_steps + 1):
            ep.steps = step
            text, usage = chat(messages)
            ep.prompt_tokens += usage.get("prompt", 0)
            ep.completion_tokens += usage.get("completion", 0)
            messages.append({"role": "assistant", "content": text})
            m = _BLOCK.search(text)
            if not m:
                failures += 1
                ep.cells.append({"step": step, "purpose": text[:200], "code": "", "stdout": "", "error": "format: no python block"})
                messages.append({"role": "user", "content": "Format error: reply with one 'Purpose:' line and exactly one ```python block."})
            else:
                code = m.group(1)
                res = session.run(code)
                purpose = text.split("```")[0].strip()[:200]
                ep.cells.append({"step": step, "purpose": purpose, "code": code, "stdout": res.stdout[:2000],
                                 "error": res.error, "seconds": round(res.seconds, 2)})
                if res.error:
                    failures += 1
                if res.submission is not None:
                    sub = res.submission
                    break
                messages.append({"role": "user", "content": _feedback(res, max_steps - step)})
            if failures >= max_failures:
                why_none = "too_many_failures"
                break
    finally:
        session.close()
    ep.verdict = decide(sub, limit, why_none)
    ep.seconds = time.monotonic() - t0
    return ep


def openai_chat(base_url="http://127.0.0.1:8000/v1", model="qwen3.8-27b", temperature=0.3, max_tokens=1500) -> ChatFn:
    """The local model, served by vLLM. Thinking is switched off: Part 3 is about the loop, not about the model."""
    from openai import APIConnectionError, APITimeoutError, OpenAI
    client = OpenAI(base_url=base_url, api_key="none", timeout=180)

    def chat(messages):
        # A server restart is an infrastructure event, not an agent failure: wait and retry (up to ~5 minutes).
        for attempt in range(30):
            try:
                r = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                                                   extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                return r.choices[0].message.content or "", {"prompt": r.usage.prompt_tokens, "completion": r.usage.completion_tokens}
            except (APIConnectionError, APITimeoutError):
                time.sleep(10)
        raise RuntimeError("model server unreachable for 5 minutes")
    return chat
