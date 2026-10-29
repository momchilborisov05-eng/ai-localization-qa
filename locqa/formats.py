"""Pick the right reader for a file and name the translated output."""

from __future__ import annotations

from pathlib import Path

from .errors import LocQAError
from .jsonres import JsonDocument
from .models import Document
from .xliff import XliffDocument

XLIFF_SUFFIXES = {".xlf", ".xliff", ".mxliff", ".sdlxliff"}


def load_document(path: Path, source_lang: str = "en", target_path: Path | None = None) -> Document:
    path = Path(path)
    if not path.exists():
        raise LocQAError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix in XLIFF_SUFFIXES:
        return XliffDocument.load(path)
    if suffix == ".json":
        return JsonDocument.load(path, source_lang, target_path)
    raise LocQAError(f"Unsupported file type '{suffix}'. Use XLIFF 1.2/2.0 (.xlf, .xliff, .mxliff) or JSON.")


def display_path(path: Path | str) -> str:
    """A short path for reports: relative to the current folder when possible, with forward slashes."""
    path = Path(path).resolve()
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            continue
    return path.name


def output_name(path: Path, source_lang: str, target_lang: str) -> str:
    """product_copy.en.xlf -> product_copy.bg.xlf; strings.json -> strings.bg.json"""
    path = Path(path)
    stem = path.stem
    for lang in {source_lang, source_lang.split("-")[0]}:
        if stem.lower().endswith(f".{lang.lower()}"):
            stem = stem[: -len(lang) - 1]
            break
    return f"{stem}.{target_lang}{path.suffix}"
