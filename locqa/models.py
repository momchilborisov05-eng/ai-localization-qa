from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Segment:
    """One translatable unit. Inline tags in source/target are ⟦n⟧ tokens (see tags.py)."""

    id: str
    source: str
    target: str = ""
    tags: dict[str, str] = field(default_factory=dict)  # token -> original inline XML
    note: str = ""
    max_chars: int | None = None
    needs_review: bool = False  # set by the pipeline when QA finds errors


class Document:
    """A bilingual file. Subclasses know how to read and write their format."""

    format_name = "base"

    def __init__(self, path: Path, source_lang: str, target_lang: str, segments: list[Segment]):
        self.path = Path(path)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.segments = segments

    def save(self, path: Path, target_lang: str) -> None:
        raise NotImplementedError
