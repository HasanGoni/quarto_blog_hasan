"""Part 8: three kinds of judge behind ONE interface, so the experiments cannot favour one by how they are called.

    judge.noul(state, instruction)            -> probability the statement is true, plus latency and tokens
    judge.choice(state, instruction, options) -> {option: probability}, plus latency and tokens

- LayaJudge     small typed model (Apache-2.0, runs locally, never generates text). Probabilities come straight from the model.
- LLMJudge      the local Qwen3-Coder, two ways of getting a probability out of a text generator:
                  "verbalized": the model writes a number ("probability": 0.8). This is what Part 7 did.
                  "logprob":    force a one-word answer and read the probability of the answer token.
- JevJudge      TypeSafe's hosted Jev over POST /v1/systemone (the request shape Laya's own server mirrors). Needs TYPESAFE_API_KEY.
                I never had a key, so this class is written from Laya's README and Jev's public docs and has NOT been run.
"""
import json
import math
import os
import re
import time

import requests

os.environ.setdefault("USE_TF", "0")  # laya README: transformers probes for TensorFlow and can deadlock model construction


class Result(dict):
    """{"p": float | {option: p}, "ms": float, "tokens": int}"""


class LayaJudge:
    def __init__(self, subfolder: str | None = None, device: str = "cuda", name: str | None = None):
        import torch
        if device == "cuda":
            torch.zeros(1, device="cuda")  # create the CUDA context first: if laya has to create it, the move to the GPU can fail with a
            # spurious "out of memory" on this shared-memory box and laya then silently falls back to CPU (10x slower)
        import laya
        self.name = name or f"laya-{subfolder or 'english'}"
        kw = {"subfolder": subfolder} if subfolder else {}
        self.agent = laya.load("convaiinnovations/laya", device=device, **kw)
        self.device = str(self.agent.device)
        if device == "cuda" and self.agent.device.type != "cuda":
            raise RuntimeError(f"{self.name} fell back to {self.device}; refusing to report CPU numbers as GPU numbers")

    def _predict(self, state, questions):
        import torch
        if self.agent.device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = self.agent.predict(state, questions)
        if self.agent.device.type == "cuda":
            torch.cuda.synchronize()
        return out["answers"], (time.perf_counter() - t) * 1000

    def noul(self, state, instruction) -> Result:
        ans, ms = self._predict(state, {"q": {"type": "noul", "instructions": instruction}})
        return Result(p=float(ans["q"]["noul"]), ms=ms, tokens=0)

    def choice(self, state, instruction, options: dict[str, str]) -> Result:
        ans, ms = self._predict(state, {"q": {"type": "choice", "instructions": instruction, "criteria": options}})
        probs = ans["q"].get("probabilities")
        if isinstance(probs, dict):
            p = {k: float(v) for k, v in probs.items()}
        else:  # list aligned with the options order
            p = dict(zip(options, map(float, probs)))
        return Result(p=p, ms=ms, tokens=0)


class LLMJudge:
    def __init__(self, mode: str = "verbalized"):
        from agent_loop import MODEL, client
        self.client, self.model, self.mode = client, MODEL, mode
        self.name = f"qwen3-coder-{mode}"
        self.device = "vLLM"

    @staticmethod
    def _text(state) -> str:
        return state if isinstance(state, str) else json.dumps(state)

    @staticmethod
    def _token_probs(r) -> dict[str, float]:
        """Probability mass per normalised token. "Yes", " yes" and "yes" are the same answer, so their probabilities ADD UP
        (an earlier version kept only the last variant it saw and silently dropped most of the mass)."""
        out: dict[str, float] = {}
        for t in r.choices[0].logprobs.content[0].top_logprobs:
            k = t.token.strip().lower()
            out[k] = out.get(k, 0.0) + math.exp(t.logprob)
        return out

    def _chat(self, system, user, **kw):
        t = time.perf_counter()
        r = self.client.chat.completions.create(model=self.model, temperature=0.0,
                                                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
        return r, (time.perf_counter() - t) * 1000

    def noul(self, state, instruction) -> Result:
        user = f"State:\n{self._text(state)}\n\nStatement to judge: {instruction}"
        if self.mode == "logprob":
            r, ms = self._chat("Answer with exactly one word: yes or no.", user, max_tokens=1, logprobs=True, top_logprobs=10)
            probs = self._token_probs(r)
            y, n = probs.get("yes", 0.0), probs.get("no", 0.0)
            return Result(p=y / (y + n) if y + n > 0 else 0.5, ms=ms, tokens=r.usage.total_tokens)
        r, ms = self._chat('Judge the statement about the state. Reply with ONLY JSON: {"probability": <0 to 1, that the statement is true>}', user, max_tokens=40)
        m = re.search(r"[-+]?\d*\.?\d+", r.choices[0].message.content or "")
        return Result(p=min(1.0, max(0.0, float(m.group()))) if m else 0.5, ms=ms, tokens=r.usage.total_tokens)

    def choice(self, state, instruction, options: dict[str, str]) -> Result:
        listing = "\n".join(f"- {k}: {v}" for k, v in options.items())
        user = f"State:\n{self._text(state)}\n\nQuestion: {instruction}\nOptions:\n{listing}"
        if self.mode == "logprob":
            keys = list(options)
            r, ms = self._chat("Answer with exactly one option name from the list, nothing else.", user, max_tokens=1, logprobs=True, top_logprobs=20)
            tops = self._token_probs(r)
            raw = {k: sum(pr for tok, pr in tops.items() if tok and k.lower().startswith(tok)) for k in keys}
            z = sum(raw.values()) or 1.0
            return Result(p={k: v / z for k, v in raw.items()}, ms=ms, tokens=r.usage.total_tokens)
        r, ms = self._chat('Reply with ONLY JSON: {"probabilities": {"<option>": <probability>, ...}} covering every option, summing to 1.', user, max_tokens=80)
        try:
            d = json.loads(re.search(r"\{.*\}", r.choices[0].message.content, re.S).group())["probabilities"]
            raw = {k: max(0.0, float(d.get(k, 0.0))) for k in options}
        except Exception:
            raw = {k: 1.0 for k in options}
        z = sum(raw.values()) or 1.0
        return Result(p={k: v / z for k, v in raw.items()}, ms=ms, tokens=r.usage.total_tokens)


class JevJudge:
    """TypeSafe Jev, hosted. NOT RUN in this series: I had no API key. Written from Laya's README (which says laya-serve mirrors
    Jev's POST /v1/systemone shape) and Jev's public page, so the request/response field names are the unverified part."""
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or os.environ.get("TYPESAFE_BASE_URL", "")).rstrip("/")
        self.key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not (self.base_url and self.key):
            raise RuntimeError("JevJudge needs TYPESAFE_BASE_URL and TYPESAFE_API_KEY (see console.typesafe.ai)")
        self.name, self.device = "jev", "hosted"

    def _post(self, state, questions):
        t = time.perf_counter()
        r = requests.post(f"{self.base_url}/v1/systemone", headers={"Authorization": f"Bearer {self.key}"},
                          json={"state": state if isinstance(state, dict) else {"document": state}, "questions": questions}, timeout=60)
        r.raise_for_status()
        return r.json()["answers"], (time.perf_counter() - t) * 1000

    def noul(self, state, instruction) -> Result:
        a, ms = self._post(state, {"q": {"type": "noul", "instructions": instruction}})
        return Result(p=float(a["q"]["noul"]), ms=ms, tokens=0)

    def choice(self, state, instruction, options) -> Result:
        a, ms = self._post(state, {"q": {"type": "choice", "instructions": instruction, "criteria": options}})
        probs = a["q"]["probabilities"]
        return Result(p=probs if isinstance(probs, dict) else dict(zip(options, probs)), ms=ms, tokens=0)
