"""
Term base loaded from CSV.

Columns: term, do_not_translate, case_sensitive, note, plus one column per target language
(e.g. bg, de). A target cell may list accepted variants separated by "|"; the first is preferred.

    term,bg,de,do_not_translate,case_sensitive,note
    binding,автомати,Bindung,no,no,Snowboard bindings
    MIPS,,,yes,yes,Trademark

Source matching is whole-word and accepts English plurals (binding -> bindings). Target
compliance uses a light stemming rule so inflected forms count: each word of the approved term
must start a word in the translation, with up to 2 trailing characters dropped (Bulgarian
articles, German case endings). Irregular forms (Naht/Nähte) go in the variants list.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from .errors import LocQAError
from .tm import same_language

RESERVED = {"term", "do_not_translate", "case_sensitive", "note"}
_YES = {"yes", "y", "true", "1", "x"}
_WORD = re.compile(r"\w+")


def _stem(word: str) -> str:
    return word[: max(4, len(word) - 2)]


@dataclass
class GlossaryTerm:
    source: str
    targets: dict[str, list[str]] = field(default_factory=dict)
    dnt: bool = False
    case_sensitive: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        words = [re.escape(w) for w in self.source.split()]
        if words and not self.dnt:
            words[-1] += r"(?:s|es)?"
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self.pattern = re.compile(r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)", flags)

    def target_for(self, lang: str) -> list[str]:
        for key, variants in self.targets.items():
            if key.lower() == lang.lower():
                return variants
        for key, variants in self.targets.items():
            if same_language(key, lang):
                return variants
        return []

    def preferred(self, lang: str) -> str:
        if self.dnt:
            return self.source
        variants = self.target_for(lang)
        return variants[0] if variants else ""

    def found_in(self, text: str) -> bool:
        return bool(self.pattern.search(text))

    def satisfied_by(self, target_text: str, lang: str) -> bool:
        """Is the approved translation (or the untouched DNT term) used in the target?"""
        if self.dnt:
            if self.case_sensitive:
                return self.source in target_text
            return self.source.lower() in target_text.lower()
        target_words = [w.lower() for w in _WORD.findall(target_text)]
        for variant in self.target_for(lang):
            required = [w.lower() for w in _WORD.findall(variant)]
            if required and all(any(t.startswith(_stem(r)) for t in target_words) for r in required):
                return True
        return False


class Glossary:
    def __init__(self, terms: list[GlossaryTerm] | None = None):
        # longest terms first, so "sintered base" is reported before "base"
        self.terms = sorted(terms or [], key=lambda t: -len(t.source))

    def __len__(self) -> int:
        return len(self.terms)

    @classmethod
    def load(cls, path: Path | None) -> "Glossary":
        if path is None:
            return cls()
        path = Path(path)
        if not path.exists():
            raise LocQAError(f"Glossary not found: {path}")
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if "term" not in (reader.fieldnames or []):
                raise LocQAError(f"{path}: the glossary CSV needs a 'term' column (plus one column per target language).")
            languages = [c for c in reader.fieldnames if c and c.strip().lower() not in RESERVED]
            terms = []
            for row in reader:
                source = (row.get("term") or "").strip()
                if not source:
                    continue
                targets = {
                    lang: [v.strip() for v in (row.get(lang) or "").split("|") if v.strip()]
                    for lang in languages
                }
                terms.append(GlossaryTerm(
                    source=source,
                    targets={k: v for k, v in targets.items() if v},
                    dnt=(row.get("do_not_translate") or "").strip().lower() in _YES,
                    case_sensitive=(row.get("case_sensitive") or "").strip().lower() in _YES,
                    note=(row.get("note") or "").strip(),
                ))
        return cls(terms)

    def for_language(self, lang: str) -> list[GlossaryTerm]:
        return [t for t in self.terms if t.dnt or t.target_for(lang)]

    def find_in(self, source_text: str, lang: str) -> list[GlossaryTerm]:
        """Terms with a rule for this language that occur in the source text."""
        return [t for t in self.for_language(lang) if t.found_in(source_text)]
