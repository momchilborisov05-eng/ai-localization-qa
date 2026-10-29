"""
JSON key/value resource files (i18next, Phrase Strings, most web/app frameworks).

Nested objects and arrays are supported; every string leaf is one segment whose id is the dotted
key path (e.g. "cart.title"). The output keeps the same structure and key order.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .errors import LocQAError
from .models import Document, Segment

KeyPath = tuple[str | int, ...]


def _walk(node, path: KeyPath, out: list[tuple[KeyPath, str]]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            _walk(value, path + (key,), out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk(value, path + (index,), out)
    elif isinstance(node, str):
        out.append((path, node))


class JsonDocument(Document):
    format_name = "json"

    def __init__(self, path: Path, data, source_lang: str, segments: list[Segment], paths: dict[str, KeyPath]):
        super().__init__(path, source_lang, "", segments)
        self.data = data
        self._paths = paths

    @classmethod
    def load(cls, path: Path, source_lang: str = "en", target_path: Path | None = None) -> "JsonDocument":
        data = _read(path)
        leaves: list[tuple[KeyPath, str]] = []
        _walk(data, (), leaves)
        targets: dict[KeyPath, str] = {}
        if target_path is not None:
            found: list[tuple[KeyPath, str]] = []
            _walk(_read(target_path), (), found)
            targets = dict(found)
        segments, paths = [], {}
        for key_path, text in leaves:
            seg_id = ".".join(str(p) for p in key_path)
            segments.append(Segment(seg_id, text, targets.get(key_path, "")))
            paths[seg_id] = key_path
        return cls(path, data, source_lang, segments, paths)

    def save(self, path: Path, target_lang: str) -> list[str]:
        """Write the translated JSON. Untranslated keys fall back to the source text."""
        out = copy.deepcopy(self.data)
        for seg in self.segments:
            *parents, last = self._paths[seg.id]
            node = out
            for key in parents:
                node = node[key]
            node[last] = seg.target or seg.source
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")
        return []


def _read(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        raise LocQAError(f"{path} is not valid JSON: {e}") from e
