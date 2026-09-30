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

from detector_tool import DATA_DIR, inspect_casting_xray

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)
DEFAULT_IMAGE = DATA_DIR / "C0001_0001.png"
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


def run_tool_call(image_path: Path = DEFAULT_IMAGE) -> str:
    load_dotenv()
    client = ai.Client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an industrial X-ray inspection assistant. "
                    "Always call inspect_casting_xray before answering. "
                    "Explain what the detector found in plain English for a human operator. "
                    "Be honest about false positives and missed defects when the numbers say so."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Inspect this frame and tell me whether it looks like a pass or "
                    f"needs human review. Image path: {image_path}. Keep it under 150 words."
                ),
            },
        ],
        tools=[inspect_casting_xray],
        max_turns=3,
    )
    return strip_tool_markup(response.choices[0].message.content or "")


def main():
    load_dotenv()
    if not DEFAULT_IMAGE.exists():
        raise FileNotFoundError(f"GDXray image not found: {DEFAULT_IMAGE}")

    tool_only = inspect_casting_xray(str(DEFAULT_IMAGE))
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
        "=== Tool result (inspect_casting_xray) ===\n"
        f"{tool_only}\n\n"
        "=== LLM explanation ===\n"
        f"{explanation}\n"
    )
    (OUT_DIR / "first_tool_call_output.txt").write_text(combined)
    print(combined)


if __name__ == "__main__":
    main()
