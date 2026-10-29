"""Shared test helpers: paths and a fake Anthropic client (no network)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
sys.path.insert(0, str(ROOT))


def fake_response(payload: dict, stop_reason: str = "end_turn") -> NS:
    return NS(
        stop_reason=stop_reason,
        content=[NS(type="text", text=json.dumps(payload, ensure_ascii=False))],
        usage=NS(input_tokens=100, output_tokens=50, cache_read_input_tokens=900, cache_creation_input_tokens=0),
    )


class FakeClaude:
    """Answers each request with a function of the request payload and records every request."""

    def __init__(self, answer, stop_reason: str = "end_turn"):
        self.answer = answer
        self.stop_reason = stop_reason
        self.requests: list[dict] = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        payload = json.loads(kwargs["messages"][-1]["content"])
        return fake_response(self.answer(payload), self.stop_reason)

    def payloads(self) -> list[dict]:
        return [json.loads(r["messages"][-1]["content"]) for r in self.requests]


def echo_translator(prefix: str = "T:"):
    """Translates by prefixing the source, so every tag and placeholder is kept."""
    def answer(payload):
        if payload["task"] == "translate":
            return {"translations": [{"id": s["id"], "target": prefix + s["source"]} for s in payload["segments"]]}
        return {"segments": [{"id": s["id"], "issues": []} for s in payload["segments"]]}
    return answer
