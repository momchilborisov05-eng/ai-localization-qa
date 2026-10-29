"""
Linguist feedback loop: turn post-edits into numbers and into better resources.

Input: a CSV exported from the review step (e.g. Phrase TMS post-editing analysis or an LQA
sheet) with the columns

    segment_id, target_lang, source, mt, post_edit, category, comment

Output:
  * post-edit effort per segment and per language: character edit distance, normalized edit
    distance, word-level edit rate (TER-like), share of segments left unchanged, effort buckets
  * the confirmed post-edits added to the translation memory (a new TMX, the original is untouched)
  * recurring MT -> post-edit term substitutions as glossary candidates, cross-checked against
    the glossary: either a new term, or an existing term the MT did not follow (a prompt or
    enforcement problem rather than a missing term)
  * a before/after comparison: deterministic QA findings on MT vs post-edit, and TM leverage
    on a source file before vs after the TM update
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from .errors import LocQAError
from .fuzzy import levenshtein
from .glossary import Glossary
from .models import Segment
from .qa_checks import ERROR, run_qa
from .tags import visible
from .tm import TranslationMemory, same_language

REQUIRED_COLUMNS = {"segment_id", "target_lang", "source", "mt", "post_edit"}
EFFORT_BUCKETS = [(0.0, "none"), (0.15, "light"), (0.40, "medium"), (1.01, "heavy")]
_WORDS = re.compile(r"\w+(?:[-'’.,]\w+)*|[^\w\s]")


@dataclass
class FeedbackRow:
    segment_id: str
    target_lang: str
    source: str
    mt: str
    post_edit: str
    category: str = ""
    comment: str = ""


def load_feedback(path: Path) -> list[FeedbackRow]:
    path = Path(path)
    if not path.exists():
        raise LocQAError(f"Feedback file not found: {path}")
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise LocQAError(f"{path}: missing column(s) {', '.join(sorted(missing))}. "
                             f"Expected: {', '.join(sorted(REQUIRED_COLUMNS))}, category, comment.")
        rows = [
            FeedbackRow(r["segment_id"].strip(), r["target_lang"].strip(), r["source"], r["mt"], r["post_edit"],
                        (r.get("category") or "").strip().lower(), (r.get("comment") or "").strip())
            for r in reader
        ]
    rows = [r for r in rows if r.post_edit.strip()]
    if not rows:
        raise LocQAError(f"{path} has no rows with a post-edit.")
    return rows


# ---------- metrics ----------

def effort_bucket(normalized_distance: float) -> str:
    if normalized_distance == 0:
        return "none"
    for limit, name in EFFORT_BUCKETS[1:]:
        if normalized_distance <= limit:
            return name
    return "heavy"


def segment_metrics(mt: str, post_edit: str) -> dict:
    mt_v, pe_v = visible(mt), visible(post_edit)
    distance = levenshtein(mt_v, pe_v)
    normalized = distance / max(len(mt_v), len(pe_v), 1)
    mt_words, pe_words = _WORDS.findall(mt_v), _WORDS.findall(pe_v)
    word_edits = levenshtein(mt_words, pe_words)
    return {
        "char_edit_distance": distance,
        "normalized_edit_distance": round(normalized, 3),
        "word_edits": word_edits,
        "edit_rate": round(word_edits / max(len(pe_words), 1), 3),  # TER-like, without shifts
        "effort": effort_bucket(normalized),
        "unchanged": mt_v == pe_v,
    }


def substitutions(mt: str, post_edit: str) -> list[tuple[str, str]]:
    """Word spans the linguist replaced: [(mt phrase, post-edit phrase), ...]."""
    a, b = _WORDS.findall(visible(mt)), _WORDS.findall(visible(post_edit))
    out = []
    matcher = SequenceMatcher(None, [w.lower() for w in a], [w.lower() for w in b], autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "replace":
            old = " ".join(w for w in a[i1:i2] if re.search(r"\w", w))
            new = " ".join(w for w in b[j1:j2] if re.search(r"\w", w))
            if old and new and not old.replace(",", "").replace(".", "").isdigit():
                out.append((old, new))
    return out


@dataclass
class GlossaryCandidate:
    target_lang: str
    mt_term: str
    post_edit_term: str
    occurrences: int
    categories: list[str]
    status: str                 # "new-candidate" or "glossary-not-followed"
    glossary_term: str = ""     # the existing source term, for "glossary-not-followed"
    segments: list[str] = field(default_factory=list)
    example_source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def glossary_candidates(rows: list[FeedbackRow], glossary: Glossary, min_occurrences: int = 2) -> list[GlossaryCandidate]:
    """A substitution becomes a candidate when it recurs, or when the linguist tagged it as terminology."""
    groups: dict[tuple[str, str, str], list[FeedbackRow]] = defaultdict(list)
    originals: dict[tuple[str, str, str], tuple[str, str]] = {}
    for row in rows:
        for old, new in substitutions(row.mt, row.post_edit):
            key = (row.target_lang, old.lower(), new.lower())
            groups[key].append(row)
            originals.setdefault(key, (old, new))
    candidates = []
    for key, group in groups.items():
        categories = sorted({r.category for r in group if r.category})
        if len(group) < min_occurrences and "terminology" not in categories:
            continue
        lang = key[0]
        old, new = originals[key]
        linked = ""
        for term in glossary.terms:
            if all(term.found_in(visible(r.source)) for r in group) and not term.dnt \
                    and term.satisfied_by(new, lang) and not term.satisfied_by(old, lang):
                linked = term.source
                break
        candidates.append(GlossaryCandidate(
            target_lang=lang, mt_term=old, post_edit_term=new, occurrences=len(group), categories=categories,
            status="glossary-not-followed" if linked else "new-candidate", glossary_term=linked,
            segments=[r.segment_id for r in group], example_source=group[0].source,
        ))
    candidates.sort(key=lambda c: (c.status != "glossary-not-followed", -c.occurrences, c.target_lang))
    return candidates


def write_candidates_csv(candidates: list[GlossaryCandidate], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_lang", "mt_term", "post_edit_term", "occurrences", "status", "glossary_term",
                         "categories", "segments", "example_source"])
        for c in candidates:
            writer.writerow([c.target_lang, c.mt_term, c.post_edit_term, c.occurrences, c.status, c.glossary_term,
                             "|".join(c.categories), "|".join(c.segments), c.example_source])


# ---------- TM update and before/after ----------

def update_tm(rows: list[FeedbackRow], tm: TranslationMemory, source_lang: str = "en") -> dict[str, int]:
    counts = Counter()
    for row in rows:
        counts[tm.add(row.source, row.post_edit, source_lang, row.target_lang, origin="feedback")] += 1
    return {k: counts.get(k, 0) for k in ("added", "updated", "unchanged")}


def tm_leverage(sources: list[str], tm: TranslationMemory, target_lang: str,
                reuse_threshold: int = 100, fuzzy_min: int = 75) -> dict[str, int]:
    counts = {"exact": 0, "fuzzy": 0, "no_match": 0}
    for source in sources:
        matches = tm.lookup(source, target_lang, min_score=fuzzy_min, limit=1)
        if matches and matches[0].score >= reuse_threshold:
            counts["exact"] += 1
        elif matches:
            counts["fuzzy"] += 1
        else:
            counts["no_match"] += 1
    return counts


def qa_before_after(rows: list[FeedbackRow], glossary: Glossary) -> dict[str, dict[str, int]]:
    """Deterministic QA errors/warnings on the MT vs on the linguist's post-edit, per language."""
    out = {}
    for lang in sorted({r.target_lang for r in rows}):
        lang_rows = [r for r in rows if r.target_lang == lang]
        result = {}
        for label, attr in (("mt", "mt"), ("post_edit", "post_edit")):
            segments = [Segment(r.segment_id, r.source, getattr(r, attr)) for r in lang_rows]
            issues = run_qa(segments, lang, glossary)
            result[f"{label}_errors"] = sum(i.severity == ERROR for i in issues)
            result[f"{label}_warnings"] = sum(i.severity != ERROR for i in issues)
        out[lang] = result
    return out


def analyse(rows: list[FeedbackRow], glossary: Glossary) -> dict:
    """Post-edit metrics, per language and overall."""
    per_segment = []
    for row in rows:
        per_segment.append({"segment_id": row.segment_id, "target_lang": row.target_lang,
                            "category": row.category, "comment": row.comment, "source": row.source,
                            "mt": row.mt, "post_edit": row.post_edit, **segment_metrics(row.mt, row.post_edit)})
    languages = {}
    for lang in sorted({r["target_lang"] for r in per_segment}):
        items = [r for r in per_segment if same_language(r["target_lang"], lang)]
        languages[lang] = _aggregate(items)
    return {
        "overall": _aggregate(per_segment),
        "languages": languages,
        "segments": per_segment,
        "qa_before_after": qa_before_after(rows, glossary),
    }


def _aggregate(items: list[dict]) -> dict:
    n = len(items)
    effort = Counter(i["effort"] for i in items)
    categories = Counter(i["category"] for i in items if i["category"] and i["category"] != "none")
    return {
        "segments": n,
        "unchanged_pct": round(100 * sum(i["unchanged"] for i in items) / n, 1) if n else 0.0,
        "avg_normalized_edit_distance": round(sum(i["normalized_edit_distance"] for i in items) / n, 3) if n else 0.0,
        "avg_edit_rate": round(sum(i["edit_rate"] for i in items) / n, 3) if n else 0.0,
        "effort": {name: effort.get(name, 0) for _, name in EFFORT_BUCKETS},
        "categories": dict(categories.most_common()),
    }
