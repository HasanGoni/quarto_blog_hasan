"""Part 4: structured output over the Part 3 tool result, with real validation + retry.

Takes the real `inspect_xray` detector JSON (Part 2's classical top-hat detector on the synthetic
void dataset: candidate boxes with a strength, no ground truth) and asks a real local LLM
(Qwen3-Coder-30B via vLLM, OpenAI-compatible) to turn it into a structured defect report:
bounding box -> defect type / severity / confidence / explanation.

The point of this script is the failure handling, not the happy path:
  1. Parse the model's response as JSON.
  2. Validate it against a Pydantic schema.
  3. If either step fails, retry ONCE with the exact error fed back to the model.
  4. If it still fails, return status="needs_review" with the raw text attached -- never
     silently fall back to an empty defect list, which is indistinguishable from a genuinely
     clean part.

Configure via .env (see .env.example):
  OPENAI_API_KEY=local
  OPENAI_BASE_URL=http://localhost:8000/v1
  AGENTIC_ZERO_MODEL=openai:qwen3-coder
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
import re
from pathlib import Path
from typing import List, Literal

import aisuite as ai
from pydantic import BaseModel, Field, ValidationError

import void_data as vd
from detector_tool import inspect_xray

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)
ENV_FILE = Path(__file__).parent / ".env"


def load_dotenv(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()
MODEL = os.environ.get("AGENTIC_ZERO_MODEL", "openai:gpt-4o-mini")
MAX_IMAGES = int(os.environ.get("AGENTIC_ZERO_MAX_IMAGES", "0")) or None  # 0/unset = all 600 held-out images
WORKERS = 8


class Defect(BaseModel):
    bbox: List[int] = Field(description="[x1, y1, x2, y2] in pixels")
    type: Literal["void", "inclusion", "crack", "porosity", "other"]
    severity: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


class DefectReport(BaseModel):
    defects: List[Defect]


SCHEMA_JSON = json.dumps(DefectReport.model_json_schema(), indent=2)

SYSTEM_PROMPT = f"""You are an industrial X-ray inspection assistant. You will be given the raw
output of a classical computer-vision void detector (candidate bounding boxes, each with a
strength, no semantic labels) for one X-ray image of a metal part. Turn it into a structured
defect report.

Output ONLY a single JSON object matching this schema -- no markdown fences, no prose before or
after it:

{SCHEMA_JSON}

Rules:
- Emit exactly one entry in "defects" per candidate box in the input, in the same order.
- Infer "type" and "severity" from the box's size, position, and the candidate's strength
  -- you do not have the image, only these numbers, so be conservative and say so in
  "explanation" when the numbers are genuinely ambiguous.
- If the detector reports zero candidate boxes, return {{"defects": []}} -- that is a valid,
  clean result, not a failure.
- Never invent a defect that is not backed by a candidate box in the input."""


def strip_fences(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def parse_and_validate(raw_text: str) -> tuple[DefectReport | None, str | None]:
    """Returns (report, None) on success, (None, error_message) on failure."""
    try:
        data = json.loads(strip_fences(raw_text))
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc}"
    try:
        return DefectReport(**data), None
    except ValidationError as exc:
        return None, f"ValidationError: {exc}"


def call_llm(client: "ai.Client", messages: list[dict]) -> str:
    response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.2)
    return response.choices[0].message.content or ""


def structured_report_for_image(client: "ai.Client", image_id: str) -> dict:
    tool_result = inspect_xray(image_id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Detector output for {image_id}:\n{tool_result}"},
    ]

    raw = call_llm(client, messages)
    report, error = parse_and_validate(raw)
    if report is not None:
        return {
            "image": image_id,
            "status": "ok",
            "attempts": 1,
            "defects": [d.model_dump() for d in report.defects],
        }

    # One retry: feed the exact error back and ask for a fix, nothing else.
    messages.append({"role": "assistant", "content": raw})
    messages.append(
        {
            "role": "user",
            "content": (
                f"That response failed validation with this error:\n{error}\n\n"
                "Return ONLY the corrected JSON object matching the schema above. "
                "No markdown fences, no explanation."
            ),
        }
    )
    raw_retry = call_llm(client, messages)
    report_retry, error_retry = parse_and_validate(raw_retry)
    if report_retry is not None:
        return {
            "image": image_id,
            "status": "recovered_after_retry",
            "attempts": 2,
            "first_attempt_error": error,
            "defects": [d.model_dump() for d in report_retry.defects],
        }

    return {
        "image": image_id,
        "status": "needs_review",
        "attempts": 2,
        "first_attempt_error": error,
        "second_attempt_error": error_retry,
        "raw_response": raw_retry[:800],
    }


def main():
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "Set OPENAI_API_KEY in .env or the environment. "
            "For local vLLM, use OPENAI_API_KEY=local and OPENAI_BASE_URL=http://localhost:8000/v1."
        )

    image_ids = vd.split_ids("test")  # the held-out split, all of it
    if MAX_IMAGES:
        image_ids = image_ids[:MAX_IMAGES]

    def work(image_id: str) -> dict:
        return structured_report_for_image(ai.Client(), image_id)  # one client per thread

    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(work, image_ids))
    for i, r in enumerate(results, 1):
        if r["status"] != "ok":
            print(f"[{i}/{len(results)}] {r['image']}: {r['status']}")

    n = len(results)
    ok = sum(1 for r in results if r["status"] == "ok")
    recovered = sum(1 for r in results if r["status"] == "recovered_after_retry")
    needs_review = sum(1 for r in results if r["status"] == "needs_review")
    first_attempt_failures = recovered + needs_review

    summary = {
        "model": MODEL,
        "n_images": n,
        "ok_first_try": ok,
        "recovered_after_retry": recovered,
        "needs_review": needs_review,
        "first_attempt_failure_rate": round(first_attempt_failures / n, 4),
        "retry_recovery_rate": round(recovered / first_attempt_failures, 4) if first_attempt_failures else None,
        "final_unrecoverable_rate": round(needs_review / n, 4),
    }

    (OUT_DIR / "structured_output_results.json").write_text(json.dumps(results, indent=2))
    (OUT_DIR / "structured_output_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
