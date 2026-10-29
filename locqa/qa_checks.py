"""
Deterministic QA checks (no LLM), similar to the QA step in a CAT tool or Xbench.

Each check takes a segment and returns issues. File-level checks (consistency) look at all
segments together. Severity is "error" (must fix before delivery) or "warning" (check it).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from .glossary import Glossary
from .models import Segment
from .tags import PLACEHOLDER_RE, TOKEN_RE, protected_diff, tag_order_problem, visible

ERROR, WARNING = "error", "warning"

CHECK_LABELS = {
    "tags": "Tags & placeholders",
    "numbers": "Numbers",
    "units": "Units",
    "punctuation": "Punctuation",
    "whitespace": "Whitespace",
    "length": "Length limit",
    "glossary": "Glossary",
    "dnt": "Do-not-translate",
    "untranslated": "Untranslated",
    "consistency": "Consistency",
}


@dataclass
class QAIssue:
    segment_id: str
    check: str
    severity: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- helpers ----------

_NUMBER_RE = re.compile(r"\d+(?:[.,   ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")
# canonical unit -> accepted spellings in target languages (Latin and Cyrillic)
UNITS: dict[str, list[str]] = {
    "mm": ["mm", "мм"], "cm": ["cm", "см"], "km": ["km", "км"], "m": ["m", "м"],
    "kg": ["kg", "кг"], "g": ["g", "г", "гр"], "ml": ["ml", "мл"], "l": ["l", "л"],
    "km/h": ["km/h", "км/ч"], "mph": ["mph"], "h": ["h", "ч", "Std"],
    "°C": ["°C"], "°F": ["°F"], "%": ["%"],
}
_UNIT_RE = re.compile(
    r"\d\s?(" + "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True)) + r")(?![\w])"
)
_END_PUNCT = ".!?:;…"
_CLOSERS = "\"'”“»«)]⟧ "


def _plain(text: str) -> str:
    """Text without tags and placeholders, for number checks."""
    return PLACEHOLDER_RE.sub(" ", TOKEN_RE.sub(" ", text))


def numbers_in(text: str) -> Counter:
    """Numbers as digit strings, so 20,000 / 20.000 / 20 000 and 4.5 / 4,5 compare equal."""
    return Counter(re.sub(r"\D", "", n) for n in _NUMBER_RE.findall(_plain(text)))


def _end_char(text: str) -> str:
    stripped = visible(text).rstrip(_CLOSERS)
    if stripped.endswith("..."):
        return "…"
    return stripped[-1:] if stripped else ""


# ---------- segment checks ----------

def check_tags(seg: Segment, lang: str, glossary: Glossary) -> list[QAIssue]:
    if not seg.target:
        return []
    issues = []
    missing, extra = protected_diff(seg.source, seg.target)
    if missing:
        issues.append(QAIssue(seg.id, "tags", ERROR, f"Missing in target: {' '.join(missing)}"))
    if extra:
        issues.append(QAIssue(seg.id, "tags", ERROR, f"Not in source: {' '.join(extra)}"))
    if not missing and not extra:
        problem = tag_order_problem(seg.target)
        if problem:
            issues.append(QAIssue(seg.id, "tags", ERROR, f"Tag order: {problem}"))
    return issues


def check_numbers(seg: Segment, lang: str, glossary: Glossary) -> list[QAIssue]:
    if not seg.target:
        return []
    src, tgt = numbers_in(seg.source), numbers_in(seg.target)
    spelled = {re.sub(r"\D", "", n): n for text in (seg.target, seg.source) for n in _NUMBER_RE.findall(_plain(text))}
    show = lambda counter: ", ".join(spelled[n] for n in sorted(counter.elements()))  # noqa: E731
    issues = []
    if src - tgt:
        issues.append(QAIssue(seg.id, "numbers", ERROR, f"Number(s) missing or changed: {show(src - tgt)}"))
    if tgt - src:
        issues.append(QAIssue(seg.id, "numbers", ERROR, f"Number(s) not in source: {show(tgt - src)}"))
    return issues


def check_units(seg: Segment, lang: str, glossary: Glossary) -> list[QAIssue]:
    if not seg.target:
        return []
    target_units = set()
    for unit, spellings in UNITS.items():
        if re.search(r"\d\s?(?:" + "|".join(re.escape(s) for s in spellings) + r")(?![\w])", _plain(seg.target)):
            target_units.add(unit)
    missing = sorted(set(_UNIT_RE.findall(_plain(seg.source))) - target_units)
    if missing:
        return [QAIssue(seg.id, "units", ERROR, f"Unit missing after a number: {', '.join(missing)}")]
    return []


def check_punctuation(seg: Segment, lang: str, glossary: Glossary) -> list[QAIssue]:
    if not seg.target:
        return []
    issues = []
    src_end, tgt_end = _end_char(seg.source), _end_char(seg.target)
    src_p = src_end if src_end in _END_PUNCT else ""
    tgt_p = tgt_end if tgt_end in _END_PUNCT else ""
    if src_p != tgt_p:
        issues.append(QAIssue(seg.id, "punctuation", WARNING,
                              f"Ending punctuation differs: source '{src_p or 'none'}', target '{tgt_p or 'none'}'"))
    target = visible(seg.target)
    for open_, close in ("()", "[]"):
        if target.count(open_) != target.count(close):
            issues.append(QAIssue(seg.id, "punctuation", WARNING, f"Unbalanced brackets {open_}{close}"))
    return issues


def check_whitespace(seg: Segment, lang: str, glossary: Glossary) -> list[QAIssue]:
    if not seg.target:
        return []
    issues = []
    src, tgt = seg.source, seg.target
    if (src[:1].isspace(), src[-1:].isspace()) != (tgt[:1].isspace(), tgt[-1:].isspace()):
        issues.append(QAIssue(seg.id, "whitespace", WARNING, "Leading/trailing whitespace differs from source"))
    if "  " in tgt.strip() and "  " not in src.strip():
        issues.append(QAIssue(seg.id, "whitespace", WARNING, "Double space in target"))
    if re.search(r"\w [,.;:!?](?!\w)", tgt) and not re.search(r"\w [,.;:!?](?!\w)", src):
        issues.append(QAIssue(seg.id, "whitespace", WARNING, "Space before punctuation"))
    return issues


def check_length(seg: Segment, lang: str, glossary: Glossary) -> list[QAIssue]:
    if seg.max_chars and seg.target:
        length = len(visible(seg.target))
        if length > seg.max_chars:
            return [QAIssue(seg.id, "length", ERROR, f"{length} characters, limit is {seg.max_chars}")]
    return []


def check_glossary(seg: Segment, lang: str, glossary: Glossary) -> list[QAIssue]:
    if not seg.target or not glossary:
        return []
    issues = []
    target = visible(seg.target)
    for term in glossary.find_in(visible(seg.source), lang):
        if term.satisfied_by(target, lang):
            continue
        if term.dnt:
            issues.append(QAIssue(seg.id, "dnt", ERROR, f"'{term.source}' must stay untranslated"))
        else:
            issues.append(QAIssue(seg.id, "glossary", ERROR, f"'{term.source}' should be translated as '{term.preferred(lang)}'"))
    return issues


def check_untranslated(seg: Segment, lang: str, glossary: Glossary) -> list[QAIssue]:
    if not visible(seg.source).strip():
        return []
    if not seg.target.strip():
        return [QAIssue(seg.id, "untranslated", ERROR, "Target is empty")]
    if visible(seg.target).strip() != visible(seg.source).strip():
        return []
    # identical is fine when every word is a DNT term or a glossary term that stays the same
    remaining = visible(seg.source)
    for term in glossary.find_in(remaining, lang) if glossary else []:
        if term.dnt or term.preferred(lang).lower() == term.source.lower():
            remaining = term.pattern.sub(" ", remaining)
    words = [w for w in re.findall(r"[^\W\d_]{2,}", remaining) if not w.isupper()]  # XS, XXL, EN are codes
    if words:
        return [QAIssue(seg.id, "untranslated", WARNING, "Target is identical to source")]
    return []


SEGMENT_CHECKS: list[Callable[[Segment, str, Glossary], list[QAIssue]]] = [
    check_untranslated, check_tags, check_numbers, check_units, check_punctuation,
    check_whitespace, check_length, check_glossary,
]


# ---------- file checks ----------

def check_consistency(segments: Iterable[Segment]) -> list[QAIssue]:
    """The same source should get the same translation everywhere in the file."""
    groups: dict[str, list[Segment]] = defaultdict(list)
    for seg in segments:
        if seg.target.strip():
            groups[" ".join(visible(seg.source).split()).lower()].append(seg)
    issues = []
    for group in groups.values():
        variants = {" ".join(s.target.split()) for s in group}
        if len(variants) > 1:
            ids = ", ".join(s.id for s in group)
            for seg in group:
                issues.append(QAIssue(seg.id, "consistency", WARNING,
                                      f"Same source translated {len(variants)} different ways ({ids})"))
    return issues


def run_qa(segments: list[Segment], target_lang: str, glossary: Glossary | None = None) -> list[QAIssue]:
    glossary = glossary or Glossary()
    issues: list[QAIssue] = []
    for seg in segments:
        for check in SEGMENT_CHECKS:
            issues.extend(check(seg, target_lang, glossary))
    issues.extend(check_consistency(segments))
    return issues
