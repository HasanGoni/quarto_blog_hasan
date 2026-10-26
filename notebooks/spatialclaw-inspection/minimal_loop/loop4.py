"""Part 4: the Part 3 loop with eyes (show) and hands (inspection tools), switchable so each can be measured alone.

Reuses Part 3's rule layer (`decide`, `Verdict`), block parser and model client unchanged. Only the episode loop is new,
because feedback can now carry images.
"""
import time

from loop import LIMIT_PCT, SYSTEM, _BLOCK, Episode, decide, openai_chat  # noqa: F401  (openai_chat re-exported)
from session4 import Session4

SHOW_DOC = """
- `show(array, "title")` lets you LOOK at a 2-D (grayscale) or HxWx3 (RGB) array. The picture arrives with the next
  message, at 2x size. At most 2 pictures per turn. Use it to check what your masks really cover."""

TOOLS_DOC = """
Helper functions already defined in the notebook (numpy + scipy only):
- `part_mask(image)` -> bool silhouette of the part (assumes the part is darker than the background).
- `label_blobs(mask, min_area=3)` -> (label_image, blobs). Each blob is a dict with id, area (pixels), cy, cx,
  bbox (y0, x0, y1, x1) and elongation (about 1 = round, large = line-like). Biggest first.
- `overlay(image, [mask1, mask2])` -> RGB picture with the outlines of the masks (red, green, ...), for show().
- `crop(image, cy, cx, r=24, zoom=4)` -> an enlarged window around a point, for show().
These are building blocks. Deciding which regions are voids, and which are cracks or noise, is your job."""


def system_prompt(with_show: bool, with_tools: bool) -> str:
    return SYSTEM + (SHOW_DOC if with_show else "") + (TOOLS_DOC if with_tools else "")


def _content(res, steps_left: int, with_show: bool):
    """Feedback message: text, plus the pictures show() produced this turn (as image parts)."""
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
    imgs = getattr(res, "images", [])
    if imgs:
        parts.append("pictures attached: " + "; ".join(t or "(untitled)" for t, _ in imgs))
    parts.append(f"turns left: {steps_left}")
    text = "\n".join(parts)
    if not (with_show and imgs):
        return text
    return [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": u}} for _, u in imgs]


def run_episode4(image, question, chat, with_show=False, with_tools=False, max_steps=14, max_failures=4,
                 limit=LIMIT_PCT, cell_timeout=60, session_cls=Session4) -> Episode:
    t0 = time.monotonic()
    ep = Episode(verdict=decide(None, limit, "not started"))
    messages = [{"role": "system", "content": system_prompt(with_show, with_tools)}, {"role": "user", "content": question}]
    failures, session = 0, session_cls(image, timeout_sec=cell_timeout, with_show=with_show, with_tools=with_tools)
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
                ep.cells.append({"step": step, "purpose": text[:200], "code": "", "stdout": "", "error": "format: no python block", "images": []})
                messages.append({"role": "user", "content": "Format error: reply with one 'Purpose:' line and exactly one ```python block."})
            else:
                res = session.run(m.group(1))
                imgs = getattr(res, "images", [])
                ep.cells.append({"step": step, "purpose": text.split("```")[0].strip()[:200], "code": m.group(1), "stdout": res.stdout[:2000],
                                 "error": res.error, "seconds": round(res.seconds, 2),
                                 "images": [{"title": t, "png": u} for t, u in imgs]})
                if getattr(res, "fatal", None):            # the sandbox died or misbehaved: stop, never guess
                    why_none = res.fatal
                    break
                if res.error:
                    failures += 1
                if res.submission is not None:
                    sub = res.submission
                    break
                messages.append({"role": "user", "content": _content(res, max_steps - step, with_show)})
            if failures >= max_failures:
                why_none = "too_many_failures"
                break
    finally:
        session.close()
    ep.verdict = decide(sub, limit, why_none)
    ep.seconds = time.monotonic() - t0
    return ep
