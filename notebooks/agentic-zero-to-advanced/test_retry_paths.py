"""Deterministically exercise the retry and needs_review paths of structured_output_demo.py.

The real 72-image run never triggered retry or needs_review (Qwen3-Coder-30B was 100% valid on
first try), so those code paths were untested against real inputs. This script proves they work
by injecting known-bad/known-good LLM responses through a fake client with the same
`.chat.completions.create(...)` interface aisuite uses -- no mocking of our own logic, only of
what the LLM would have said.
"""
import json
from pathlib import Path
from types import SimpleNamespace

from structured_output_demo import structured_report_for_image, DefectReport

FAKE_IMAGE = Path(__file__).parent  # inspect_casting_xray is monkeypatched below, path unused


class FakeResponse:
    def __init__(self, text: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=text))]


class ScriptedClient:
    """Returns each entry in `script` in order, one per call."""

    def __init__(self, script: list[str]):
        self._script = list(script)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, model, messages, temperature=0.2):
        return FakeResponse(self._script.pop(0))


VALID_JSON = json.dumps({
    "defects": [
        {"bbox": [1, 2, 3, 4], "type": "void", "severity": "low",
         "confidence": 0.5, "explanation": "test"}
    ]
})
BROKEN_JSON = "{ this is not valid json at all"
INVALID_SCHEMA = json.dumps({"defects": [{"bbox": [1, 2], "type": "not_a_real_type"}]})


def run_case(name: str, script: list[str], monkeypatch_tool: str = '{"candidate_boxes": 1}'):
    import structured_output_demo as mod
    original = mod.inspect_casting_xray
    mod.inspect_casting_xray = lambda path: monkeypatch_tool
    try:
        client = ScriptedClient(script)
        result = structured_report_for_image(client, Path("fake.png"))
    finally:
        mod.inspect_casting_xray = original
    print(f"\n=== {name} ===")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r1 = run_case("Case 1: valid on first try (sanity check)", [VALID_JSON])
    assert r1["status"] == "ok" and r1["attempts"] == 1

    r2 = run_case("Case 2: broken JSON first, valid on retry", [BROKEN_JSON, VALID_JSON])
    assert r2["status"] == "recovered_after_retry" and r2["attempts"] == 2

    r3 = run_case("Case 3: fails validation both times -> needs_review", [INVALID_SCHEMA, BROKEN_JSON])
    assert r3["status"] == "needs_review" and r3["attempts"] == 2
    assert "raw_response" in r3

    print("\nAll three code paths (ok / recovered_after_retry / needs_review) confirmed working.")
