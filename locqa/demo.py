"""
Offline demo client.

DemoClient has the same `client.beta.messages.create(...)` interface as `anthropic.Anthropic`,
but answers from canned JSON files in demo/canned/ instead of calling the API. The rest of the
pipeline (prompt building, batching, validation, the repair round, QA, scoring, reports) is the
same code that runs with a real API key.

The canned answers were written for the bundled sample files only. They are clearly marked as
demo data in every report, and they include deliberate mistakes so the QA checks have
something to find.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

from .errors import LocQAError

ROOT = Path(__file__).resolve().parent.parent
CANNED_DIR = ROOT / "demo" / "canned"
SAMPLES_DIR = ROOT / "samples"


class DemoClient:
    def __init__(self, canned_dir: Path = CANNED_DIR):
        self.canned_dir = Path(canned_dir)
        self.requests: list[dict] = []
        self._cache: dict[str, dict] = {}
        self.beta = NS(messages=NS(create=self._create))

    def _canned(self, name: str) -> dict:
        if name not in self._cache:
            path = self.canned_dir / name
            if not path.exists():
                raise LocQAError(f"Demo mode has no canned responses for this request ({name}). "
                                 "Demo mode only works with the bundled sample files and languages bg/de.")
            self._cache[name] = json.loads(path.read_text(encoding="utf-8"))
        return self._cache[name]

    def _create(self, **kwargs) -> NS:
        self.requests.append(kwargs)
        payload = json.loads(kwargs["messages"][-1]["content"])
        task, lang = payload["task"], payload["target_lang"]
        data = self._canned(f"{task}_{lang}.json")
        if task == "translate":
            out = {"translations": [self._translation(data, seg, lang) for seg in payload["segments"]]}
        elif task == "review":
            out = {"segments": [{"id": s["id"], "issues": data.get(s["id"], [])} for s in payload["segments"]]}
        else:
            raise LocQAError(f"Demo mode does not support the task '{task}'.")
        return NS(
            stop_reason="end_turn",
            content=[NS(type="text", text=json.dumps(out, ensure_ascii=False))],
            usage=NS(input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )

    @staticmethod
    def _translation(data: dict, seg: dict, lang: str) -> dict:
        entry = data.get(seg["id"])
        if entry is None or entry["source"] != seg["source"]:
            raise LocQAError(f"Demo mode has no canned {lang} translation for segment '{seg['id']}'. "
                             "Demo mode only works with the unmodified sample files.")
        target = entry["target"]
        if isinstance(target, list):  # [first attempt, answer in the repair round]
            target = target[1] if "previous_attempt" in seg else target[0]
        return {"id": seg["id"], "target": target}
