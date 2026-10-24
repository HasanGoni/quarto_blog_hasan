"""Context-engineering post: a small hand-written retrieval harness, in the style of Part 5's loop.

Five ideas from "Context engineering starts with retrieval" as switches you can turn on one at a time:

    grep/glob     search by name or by text WITHOUT loading files        (cfg.grep)
    paged read    read_file returns 100 lines, the model sets offset     (cfg.paged)
    offload       any tool result over OFFLOAD_TOKENS is written to a file and replaced by a path
                  plus a short preview                                   (cfg.offload)
    paths         the semantic search tool returns file paths, not text  (cfg.search_returns_paths)
    delegate      an `analyze` tool: a fresh sub-context reads one file and returns the key facts (cfg.delegate)

This is NOT the Deep Agents library. It is ~150 lines written to measure the ideas on this series' data.
"""
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from openai import BadRequestError
from transformers import AutoModel, AutoTokenizer

from agent_loop import MODEL, client
from ctx_corpus import ARCHIVE

OFFLOAD_TOKENS = 2000   # the article uses ~20,000 for a big window; this server's window is 65,536, so scaled down
PAGE_LINES = 100
MAX_STEPS = 14
MAX_OUT_TOKENS = 8000  # first pass used 1200, which cut off step-by-step counting mid-list (see the post)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def est_tokens(text: str) -> int:
    return len(text) // 4  # rough: 4 characters per token. Real usage comes from the server, below.


@dataclass(frozen=True)
class Config:
    name: str
    grep: bool = False
    paged: bool = False
    offload: bool = False
    search_returns_paths: bool = False
    delegate: bool = False
    visual: bool = False     # adds similar_cases (DINOv3 + LanceDB), see ctx_visual.py
    grep_tools: bool = True  # False = search-only arm: the model may not use grep/glob


# ---------------------------------------------------------------------------------------------------
# semantic search over the 200 reports (a tiny vector store: one matrix, cosine similarity)
# ---------------------------------------------------------------------------------------------------
class ReportIndex:
    def __init__(self):
        self.paths = sorted((ARCHIVE / "reports").glob("*.md"))
        cache = ARCHIVE.parent / "ctx_report_embeddings.npy"
        self.tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
        self.model = AutoModel.from_pretrained(EMBED_MODEL).eval()
        if cache.exists():
            self.vecs = np.load(cache)
        else:
            self.vecs = self._embed([p.read_text() for p in self.paths])
            np.save(cache, self.vecs)

    @torch.no_grad()
    def _embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 32):
            b = self.tok(texts[i:i + 32], padding=True, truncation=True, max_length=512, return_tensors="pt")
            v = self.model(**b).last_hidden_state[:, 0]  # bge uses the [CLS] vector
            out.append(torch.nn.functional.normalize(v, dim=-1).numpy())
        return np.concatenate(out)

    def search(self, query: str, k: int = 4) -> list[Path]:
        q = self._embed(["Represent this sentence for searching relevant passages: " + query])[0]
        return [self.paths[i] for i in np.argsort(-(self.vecs @ q))[:k]]


_INDEX: ReportIndex | None = None


def index() -> ReportIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = ReportIndex()
    return _INDEX


# ---------------------------------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------------------------------
def _resolve(path: str) -> Path:
    p = (ARCHIVE / path.lstrip("/")).resolve()
    if ARCHIVE.resolve() not in p.parents and p != ARCHIVE.resolve():
        raise ValueError("path is outside the archive")
    return p


def _rel(p: Path) -> str:
    return str(p.relative_to(ARCHIVE))


def t_ls(path: str = ".") -> str:
    p = _resolve(path)
    if p.is_file():
        return _rel(p)
    items = sorted(p.iterdir())
    if len(items) > 30:  # a directory listing is also a payload; show the shape, not 200 names
        return "\n".join(_rel(i) for i in items[:10]) + f"\n... {len(items)} entries in total"
    return "\n".join(_rel(i) + ("/" if i.is_dir() else "") for i in items)


def t_glob(pattern: str) -> str:
    root = ARCHIVE.resolve()
    hits = sorted(h for h in ARCHIVE.glob(pattern) if root in h.resolve().parents)  # a "../" pattern must not escape
    return "\n".join(_rel(h) for h in hits[:200]) + (f"\n... {len(hits)} matches" if len(hits) > 200 else "") \
        or "no match"


def t_grep(pattern: str, path: str = ".", mode: str = "files") -> str:
    rx = re.compile(pattern)
    root = _resolve(path)
    files = [root] if root.is_file() else sorted(f for f in root.rglob("*") if f.is_file())
    files = [f for f in files if "large_tool_results" not in f.parts]
    hits, n_lines = [], 0
    for f in files:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if rx.search(line):
                n_lines += 1
                hits.append((f, i, line))
    if mode == "count":
        return f"{n_lines} matching lines in {len({h[0] for h in hits})} files"
    if mode == "lines":
        return "\n".join(f"{_rel(f)}:{i}:{line}" for f, i, line in hits[:300]) or "no match"
    return "\n".join(sorted({_rel(h[0]) for h in hits})) or "no match"


def t_read_file(path: str, offset: int = 0, limit: int = PAGE_LINES, *, paged: bool) -> str:
    lines = _resolve(path).read_text().splitlines()
    if not paged:
        return "\n".join(lines)
    chunk = lines[offset:offset + limit]
    tail = f"\n[{len(lines)} lines in total; showing {offset}-{offset + len(chunk) - 1}]"
    return "\n".join(f"{offset + i}: {l}" for i, l in enumerate(chunk)) + tail


def t_search_reports(query: str, *, paths: bool) -> str:
    hits = index().search(query, k=4)
    if paths:
        return "\n".join(_rel(h) for h in hits)
    return "\n\n=====\n\n".join(h.read_text() for h in hits)


ANALYST_SYSTEM = ("You are a chunk analyst. You are given ONE file and a question. Return only the facts in the "
                  "file that bear on the question, as 2 to 4 short bullet points, and finish with a line "
                  "'source: <path>'. Do not guess about files you were not given.")


def _spec(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}


def tool_specs(cfg: Config) -> list[dict]:
    s = [_spec("ls", "List a directory of the archive (exports/, reports/, policy/).",
               {"path": {"type": "string"}}, [])]
    if cfg.grep and cfg.grep_tools:
        s += [_spec("glob", "Find files by name pattern, e.g. reports/test_0001*.md",
                    {"pattern": {"type": "string"}}, ["pattern"]),
              _spec("grep", "Search file CONTENTS with a regular expression without opening the files. "
                    "mode: 'files' (matching file paths, default), 'count' (how many), 'lines' (path:line:text).",
                    {"pattern": {"type": "string"}, "path": {"type": "string", "description": "file or directory"},
                     "mode": {"type": "string", "enum": ["files", "count", "lines"]}}, ["pattern"])]
    rd = ("Read a file. Returns up to 100 lines starting at `offset`; page through with offset."
          if cfg.paged else "Read a whole file.")
    props = {"path": {"type": "string"}}
    if cfg.paged:
        props |= {"offset": {"type": "integer"}, "limit": {"type": "integer"}}
    s.append(_spec("read_file", rd, props, ["path"]))
    s.append(_spec("search_reports",
                   "Semantic (by-meaning) search over the inspection reports. Returns the 4 best matches"
                   + (" as file paths." if cfg.search_returns_paths else " as full report text."),
                   {"query": {"type": "string"}}, ["query"]))
    if cfg.visual:
        s.append(_spec("similar_cases",
                       "Visual search: the k test images that look most like the given one (DINOv3 embeddings). "
                       "Returns " + ("file paths under cases/, one record per image." if cfg.search_returns_paths
                                     else "the full case records as text."),
                       {"image_id": {"type": "string"}, "k": {"type": "integer"}}, ["image_id"]))
    if cfg.delegate:
        s.append(_spec("analyze", "Hand ONE file to a fresh analyst who reads it and returns only the key facts for "
                       "your question. Use it on retrieved files instead of reading them yourself. Several calls "
                       "in one turn run in parallel.",
                       {"path": {"type": "string"}, "question": {"type": "string"}}, ["path", "question"]))
    return s


# ---------------------------------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------------------------------
SYSTEM = """You answer questions about a QC archive of X-ray void inspections. The archive has three folders:
exports/ (inspection_log.csv: one row per image of the dataset), reports/ (one markdown report per inspected test
image), policy/ (dated reporting-policy documents). Use your tools to find the evidence, do not guess.
When you are done, reply with one final line: ANSWER: <json> in the format the question asks for."""


def _call(messages, tools=None, usage=None):
    kw = dict(model=MODEL, messages=messages, temperature=0, max_tokens=MAX_OUT_TOKENS, extra_body=NO_THINK)
    if tools:
        kw["tools"] = tools
    r = client.chat.completions.create(**kw)
    if usage is not None:
        usage["calls"] += 1
    return r


def run(question: str, cfg: Config, *, stuffed: str | None = None) -> dict:
    """Run one question. Returns the answer text and the numbers that matter for context."""
    off_dir = ARCHIVE / "large_tool_results"
    if off_dir.exists():
        shutil.rmtree(off_dir)
    m = {"strategy": cfg.name, "question": question, "status": "ok", "answer": None, "steps": 0, "tool_calls": 0,
         "peak_prompt_tokens": 0, "total_prompt_tokens": 0, "completion_tokens": 0, "sub_tokens": 0,
         "offloads": 0, "tools_used": {}, "biggest_tool_result_tokens": 0}
    user = question if stuffed is None else f"{stuffed}\n\n---\n\n{question}"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    tools = None if stuffed is not None else tool_specs(cfg)
    usage = {"calls": 0}
    try:
        for step in range(1, MAX_STEPS + 1):
            r = _call(messages, tools, usage)
            m["steps"] = step
            m["peak_prompt_tokens"] = max(m["peak_prompt_tokens"], r.usage.prompt_tokens)
            m["total_prompt_tokens"] += r.usage.prompt_tokens
            m["completion_tokens"] += r.usage.completion_tokens
            msg = r.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            if not msg.tool_calls:
                m["answer"] = msg.content
                return m
            calls = msg.tool_calls
            with ThreadPoolExecutor(max_workers=8) as ex:
                results = list(ex.map(lambda c: _run_tool(c, cfg, m, off_dir), calls))
            for c, res in zip(calls, results):
                messages.append({"role": "tool", "tool_call_id": c.id, "content": res})
        m["status"] = "max_steps"
    except BadRequestError as e:
        m["status"] = "overflow" if "context" in str(e).lower() or "maximum" in str(e).lower() else f"error: {e}"
    return m


def _run_tool(call, cfg: Config, m: dict, off_dir: Path) -> str:
    name = call.function.name
    try:
        a = json.loads(call.function.arguments or "{}")
        if name == "ls":
            out = t_ls(a.get("path", "."))
        elif name == "glob":
            out = t_glob(a["pattern"])
        elif name == "grep":
            out = t_grep(a["pattern"], a.get("path", "."), a.get("mode", "files"))
        elif name == "read_file":
            out = t_read_file(a["path"], int(a.get("offset", 0)), int(a.get("limit", PAGE_LINES)), paged=cfg.paged)
        elif name == "search_reports":
            out = t_search_reports(a["query"], paths=cfg.search_returns_paths)
        elif name == "similar_cases":
            import ctx_visual
            out = ctx_visual.similar(a["image_id"], int(a.get("k", 4)), paths=cfg.search_returns_paths)
        elif name == "analyze":
            sub = _call([{"role": "system", "content": ANALYST_SYSTEM},
                         {"role": "user", "content": f"Question: {a['question']}\n\nFile {a['path']}:\n"
                          f"{_resolve(a['path']).read_text()}"}])
            m["sub_tokens"] += sub.usage.total_tokens
            out = sub.choices[0].message.content
        else:
            out = f"unknown tool {name}"
    except Exception as e:  # a bad regex or path goes back to the model as text, like any other result
        out = f"error: {e}"
    m["tool_calls"] += 1
    m["tools_used"][name] = m["tools_used"].get(name, 0) + 1
    m["biggest_tool_result_tokens"] = max(m["biggest_tool_result_tokens"], est_tokens(out))
    # a paged read is already bounded (100 lines), so it is never offloaded again (that would loop forever)
    if cfg.offload and est_tokens(out) > OFFLOAD_TOKENS and not (name == "read_file" and cfg.paged):
        off_dir.mkdir(exist_ok=True)
        f = off_dir / f"{call.id}.txt"
        f.write_text(out)
        lines = out.splitlines()
        m["offloads"] += 1
        return (f"[result too large (~{est_tokens(out)} tokens); saved to {_rel(f)}. Use grep or read_file on it.]\n"
                + "\n".join(lines[:5]) + "\n...\n" + "\n".join(lines[-5:]))
    return out


ANSWER_RX = re.compile(r"ANSWER:\s*(\{.*\})", re.S)


def parse_answer(text: str | None) -> dict | None:
    if not text:
        return None
    mt = ANSWER_RX.search(text)
    try:
        return json.loads(mt.group(1)) if mt else None
    except json.JSONDecodeError:
        return None
