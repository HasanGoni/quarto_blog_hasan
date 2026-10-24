"""Part 3: one tool-calling LLM call in front of the Part 2 classical detector.

Configure via .env or environment variables:
  OPENAI_API_KEY=sk-...          # cloud OpenAI
  OPENAI_BASE_URL=http://localhost:8000/v1  # local vLLM (OpenAI-compatible)
  AGENTIC_ZERO_MODEL=openai:qwen3-coder     # model id served by vLLM
"""
import os
import re
from pathlib import Path

import aisuite as ai

import json

from detector_tool import inspect_xray

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)
DEFAULT_IMAGE = "test_00009"
N_CHECK = 10  # the first ten held-out images, in order (not picked by outcome)
MODEL = os.environ.get("AGENTIC_ZERO_MODEL", "openai:gpt-4o-mini")
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


def strip_tool_markup(text: str) -> str:
    return re.sub(r"<tool_call>.*", "", text, flags=re.DOTALL).strip()


def run_tool_call(image_id: str = DEFAULT_IMAGE) -> str:
    load_dotenv()
    client = ai.Client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an industrial X-ray inspection assistant. The tool reports candidate voids; it has no answer key, so do not claim a candidate is a real defect. "
                    "Always call inspect_xray before answering. "
                    "Explain what the detector found in plain English for a human operator. "
                    "Be honest about false positives and missed defects when the numbers say so."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Inspect this frame and tell me whether it looks like a pass or "
                    f"needs human review. Image id: {image_id}. Keep it under 100 words."
                ),
            },
        ],
        tools=[inspect_xray],
        max_turns=3,
    )
    return strip_tool_markup(response.choices[0].message.content or "")


WORDS = {"no": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
COUNT_RE = re.compile(r"\b(\d+|" + "|".join(WORDS) + r")\s+(?:\w+\s+)?(?:candidate|box|region|detection|spot|area)", re.I)


def stated_counts(text: str) -> set[int]:
    return {int(g) if g.isdigit() else WORDS[g.lower()] for g in COUNT_RE.findall(text)}


def consistency_check(n: int = N_CHECK) -> dict:
    """Does the explanation quote the number of candidates the tool really returned?"""
    rows = []
    for i in range(n):
        image_id = f"test_{i:05d}"
        truth = json.loads(inspect_xray(image_id))["candidate_boxes"]
        text = run_tool_call(image_id)
        said = stated_counts(text)
        rows.append({"image": image_id, "tool_count": truth, "stated": sorted(said),
                     "consistent": said == {truth}, "explanation": text})
    return {"n": n, "consistent": sum(r["consistent"] for r in rows), "rows": rows}


def main():
    load_dotenv()
    tool_only = inspect_xray(DEFAULT_IMAGE)
    (OUT_DIR / "tool_result.json").write_text(tool_only)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "Set OPENAI_API_KEY in .env or the environment. "
            "For local vLLM, use OPENAI_API_KEY=local and OPENAI_BASE_URL=http://localhost:8000/v1."
        )

    explanation = run_tool_call()
    model_slug = MODEL.split(":", 1)[-1].replace("/", "-")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    backend = "vllm-local" if "localhost" in base_url or "127.0.0.1" in base_url else "openai-cloud"
    explanation_path = OUT_DIR / f"llm_explanation_{backend}_{model_slug}.txt"
    explanation_path.write_text(explanation)
    (OUT_DIR / "llm_explanation.txt").write_text(explanation)

    combined = (
        f"=== Backend: {backend} | model: {MODEL} ===\n\n"
        "=== Tool result (inspect_xray) ===\n"
        f"{tool_only}\n\n"
        "=== LLM explanation ===\n"
        f"{explanation}\n"
    )
    (OUT_DIR / "first_tool_call_output.txt").write_text(combined)
    print(combined)

    check = consistency_check()
    (OUT_DIR / "consistency.json").write_text(json.dumps(check, indent=2))
    for r in check["rows"]:
        print(r["image"], "tool:", r["tool_count"], "stated:", r["stated"], "OK" if r["consistent"] else "MISMATCH")
    print(f"consistent: {check['consistent']}/{check['n']}")


if __name__ == "__main__":
    main()
