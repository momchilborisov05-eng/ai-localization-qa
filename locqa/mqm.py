"""
MQM-style scoring of LLM review findings.

Categories follow the MQM Core top level (accuracy, fluency, terminology, style, locale
convention, markup). Penalty points per error: minor 1, major 5, critical 25. The score is
normalized by the evaluated word count, like the MQM scoring model:

    score = 100 * (1 - penalty points / source words)

A file passes when the score is at or above the threshold (default 95).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

CATEGORIES = ["accuracy", "fluency", "terminology", "style", "locale_convention", "markup"]
SEVERITIES = ["minor", "major", "critical"]
SEVERITY_WEIGHTS = {"minor": 1, "major": 5, "critical": 25}
DEFAULT_THRESHOLD = 95.0


@dataclass
class ReviewIssue:
    segment_id: str
    category: str
    severity: str
    explanation: str
    suggested_fix: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MQMResult:
    score: float
    penalty: int
    words: int
    passed: bool
    threshold: float
    by_category: dict[str, int]
    by_severity: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


def mqm_score(issues: list[ReviewIssue], word_count: int, threshold: float = DEFAULT_THRESHOLD) -> MQMResult:
    penalty = sum(SEVERITY_WEIGHTS.get(i.severity, 0) for i in issues)
    words = max(word_count, 1)
    score = round(max(0.0, 100 * (1 - penalty / words)), 2)
    categories = Counter(i.category for i in issues)
    severities = Counter(i.severity for i in issues)
    return MQMResult(
        score=score,
        penalty=penalty,
        words=word_count,
        passed=score >= threshold,
        threshold=threshold,
        by_category={c: categories.get(c, 0) for c in CATEGORIES},
        by_severity={s: severities.get(s, 0) for s in SEVERITIES},
    )
