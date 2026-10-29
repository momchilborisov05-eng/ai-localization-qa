"""
End-to-end run for one file and one target language:

  load file -> TM lookup (reuse exact matches, keep fuzzy matches as reference)
  -> Claude translation in batches (with repair round) -> deterministic QA
  -> Claude MQM review -> MQM score -> write translated file with segment states
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .errors import LocQAError
from .formats import display_path, load_document, output_name
from .glossary import Glossary
from .llm import LLM, Reviewer, Translator
from .models import Segment
from .mqm import DEFAULT_THRESHOLD, MQMResult, ReviewIssue, mqm_score
from .qa_checks import ERROR, QAIssue, run_qa
from .tags import visible
from .tm import TMMatch, TranslationMemory


@dataclass
class Resources:
    glossary: Glossary = field(default_factory=Glossary)
    tm: TranslationMemory = field(default_factory=TranslationMemory)
    style_guide: str = ""
    domain_context: str = ""


@dataclass
class Options:
    batch_size: int = 15
    reuse_threshold: int = 100   # TM matches at or above this score are reused without the LLM
    fuzzy_min: int = 75          # TM matches at or above this score are sent as reference
    review: bool = True
    mqm_threshold: float = DEFAULT_THRESHOLD
    source_lang: str = "en"


@dataclass
class SegmentResult:
    id: str
    source: str
    target: str
    origin: str           # "tm" (reused match), "mt" (Claude) or "existing" (QA of a translated file)
    tm_score: int
    max_chars: int | None
    note: str
    qa: list[QAIssue] = field(default_factory=list)
    review: list[ReviewIssue] = field(default_factory=list)
    repaired: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id, "source": self.source, "target": self.target, "origin": self.origin,
            "tm_score": self.tm_score, "max_chars": self.max_chars, "length": len(visible(self.target)),
            "note": self.note, "repaired": self.repaired,
            "qa": [i.to_dict() for i in self.qa], "review": [i.to_dict() for i in self.review],
        }


@dataclass
class RunResult:
    input_file: str
    target_lang: str
    output_file: str
    file_format: str
    segments: list[SegmentResult]
    mqm: MQMResult | None
    usage: dict
    unwritable: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        qa = [i for s in self.segments for i in s.qa]
        origins = Counter(s.origin for s in self.segments)
        fuzzy = sum(1 for s in self.segments if s.origin == "mt" and s.tm_score)
        return {
            "segments": len(self.segments),
            "source_words": word_count(s.source for s in self.segments),
            "tm_reused": origins.get("tm", 0),
            "mt_with_fuzzy": fuzzy,
            "mt_new": origins.get("mt", 0) - fuzzy,
            "existing": origins.get("existing", 0),
            "repaired": sum(s.repaired for s in self.segments),
            "qa_errors": sum(i.severity == ERROR for i in qa),
            "qa_warnings": sum(i.severity != ERROR for i in qa),
            "qa_by_check": dict(Counter(i.check for i in qa).most_common()),
            "segments_with_errors": sum(any(i.severity == ERROR for i in s.qa) for s in self.segments),
            "review_issues": sum(len(s.review) for s in self.segments),
            "mqm": self.mqm.to_dict() if self.mqm else None,
            "usage": self.usage,
            "unwritable": self.unwritable,
        }

    def to_dict(self) -> dict:
        return {
            "input_file": self.input_file, "target_lang": self.target_lang, "output_file": self.output_file,
            "format": self.file_format, "summary": self.summary(), "segments": [s.to_dict() for s in self.segments],
        }


def word_count(texts) -> int:
    return sum(len(re.findall(r"\w+", visible(t))) for t in texts)


def translate_file(path: Path, target_lang: str, out_dir: Path, resources: Resources, client,
                   options: Options | None = None, effort: str = "medium") -> RunResult:
    options = options or Options()
    doc = load_document(path, options.source_lang)
    source_lang = doc.source_lang or options.source_lang
    llm = LLM(client, effort=effort)

    origin: dict[str, str] = {}
    best: dict[str, int] = {}
    references: dict[str, list[TMMatch]] = {}
    to_translate: list[Segment] = []
    for seg in doc.segments:
        if not visible(seg.source).strip():
            seg.target, origin[seg.id], best[seg.id] = seg.source, "tm", 0
            continue
        matches = resources.tm.lookup(seg.source, target_lang, min_score=options.fuzzy_min)
        best[seg.id] = matches[0].score if matches else 0
        if matches and matches[0].score >= options.reuse_threshold:
            seg.target, origin[seg.id] = matches[0].entry.target, "tm"
        else:
            origin[seg.id] = "mt"
            references[seg.id] = matches
            to_translate.append(seg)

    repaired: set[str] = set()
    if to_translate:
        translator = Translator(llm, resources.glossary, source_lang, target_lang, resources.style_guide,
                                resources.domain_context, batch_size=options.batch_size)
        outcome = translator.translate(to_translate, references)
        for seg in to_translate:
            seg.target = outcome.targets.get(seg.id, "")
        repaired = set(outcome.repaired)

    result = _check_and_review(doc.segments, source_lang, target_lang, resources, llm, options, origin, best)
    for seg_result in result[0]:
        seg_result.repaired = seg_result.id in repaired

    out_path = Path(out_dir) / output_name(Path(path), source_lang, target_lang)
    unwritable = doc.save(out_path, target_lang)
    segments, mqm = result
    return RunResult(display_path(path), target_lang, display_path(out_path), doc.format_name, segments, mqm,
                     llm.usage.to_dict(), unwritable)


def qa_file(path: Path, target_lang: str | None, resources: Resources, client=None, options: Options | None = None,
            target_path: Path | None = None, effort: str = "medium") -> RunResult:
    """QA an already translated file (e.g. a bilingual XLIFF exported from Phrase TMS)."""
    options = options or Options()
    doc = load_document(path, options.source_lang, target_path)
    lang = target_lang or doc.target_lang
    if not lang:
        raise LocQAError(f"{path} does not say which target language it has. Pass --target-lang.")
    llm = LLM(client, effort=effort) if client is not None else None
    origin = {s.id: "existing" for s in doc.segments}
    best = {s.id: 0 for s in doc.segments}
    segments, mqm = _check_and_review(doc.segments, doc.source_lang or options.source_lang, lang, resources,
                                      llm, options, origin, best)
    return RunResult(display_path(path), lang, "", doc.format_name, segments, mqm, llm.usage.to_dict() if llm else {})


def _check_and_review(segments: list[Segment], source_lang: str, target_lang: str, resources: Resources,
                      llm: LLM | None, options: Options, origin: dict, best: dict):
    qa_issues = run_qa(segments, target_lang, resources.glossary)
    by_segment: dict[str, list[QAIssue]] = {}
    for issue in qa_issues:
        by_segment.setdefault(issue.segment_id, []).append(issue)

    review_issues: list[ReviewIssue] = []
    mqm = None
    if options.review and llm is not None:
        reviewer = Reviewer(llm, resources.glossary, source_lang, target_lang, resources.style_guide)
        review_issues = reviewer.review(segments)
        mqm = mqm_score(review_issues, word_count(s.source for s in segments), options.mqm_threshold)
    review_by_segment: dict[str, list[ReviewIssue]] = {}
    for issue in review_issues:
        review_by_segment.setdefault(issue.segment_id, []).append(issue)

    results = []
    for seg in segments:
        qa = by_segment.get(seg.id, [])
        seg.needs_review = any(i.severity == ERROR for i in qa)
        results.append(SegmentResult(seg.id, seg.source, seg.target, origin[seg.id], best[seg.id], seg.max_chars,
                                     seg.note, qa, review_by_segment.get(seg.id, [])))
    return results, mqm
