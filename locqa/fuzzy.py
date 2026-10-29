"""
Fuzzy match scoring, implemented from scratch (no external fuzzy-matching library).

CAT tools score translation-memory matches mostly on word-level edit distance, so a one-word
change in a ten-word sentence gives roughly 90%. The score here blends:

  * 70% word-level similarity: 1 - Levenshtein(words) / max(word count)
  * 30% character-level similarity (difflib ratio), which rewards small spelling differences

100 is reserved for identical strings; a whitespace-only difference gives 99.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Sequence

WORD_RE = re.compile(r"⟦/?\d+⟧|\{\{?[^{}\s]+\}?\}|\w+(?:[-'’.,]\w+)*|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text)


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance (insert, delete, substitute) between two sequences of characters or words."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _normalize(text: str) -> str:
    return " ".join(text.split())


def fuzzy_score(a: str, b: str) -> int:
    """Similarity of two segments as a TM match percentage (0-100)."""
    if a == b:
        return 100
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0
    if na == nb:
        return 99
    wa, wb = [w.lower() for w in tokenize(na)], [w.lower() for w in tokenize(nb)]
    word_sim = 1 - levenshtein(wa, wb) / max(len(wa), len(wb), 1)
    char_sim = SequenceMatcher(None, na, nb, autojunk=False).ratio()
    return min(99, round(100 * (0.7 * word_sim + 0.3 * char_sim)))
