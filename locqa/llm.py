"""
Claude-powered translation and MQM review.

- Segments are sent in batches (one request per ~15 segments), not one request per segment.
- The shared instructions, domain context, style guide and the full glossary for the target
  language go in the system prompt, marked for prompt caching: every batch after the first
  reads them from the cache.
- Each batch carries only what changes: the segments, the glossary terms found in them and
  the TM fuzzy matches to use as reference.
- Answers use Structured Outputs, so they always match a JSON schema.
- Translations are validated: every tag token and placeholder must come back unchanged.
  Segments that fail are sent back once with the problem described (a repair round).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .errors import LocQAError
from .glossary import Glossary
from .models import Segment
from .mqm import CATEGORIES, SEVERITIES, ReviewIssue
from .tags import protected_diff, tag_order_problem
from .tm import TMMatch

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}

LANGUAGE_NAMES = {
    "bg": "Bulgarian", "de": "German", "en": "English", "fr": "French", "it": "Italian", "es": "Spanish",
    "pt": "Portuguese", "nl": "Dutch", "pl": "Polish", "cs": "Czech", "ro": "Romanian", "sv": "Swedish",
    "da": "Danish", "no": "Norwegian", "fi": "Finnish", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
}


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code.lower().replace("_", "-").split("-")[0], code)


TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "target": {"type": "string"}},
                "required": ["id", "target"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["translations"],
    "additionalProperties": False,
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {"type": "string", "enum": CATEGORIES},
                                "severity": {"type": "string", "enum": SEVERITIES},
                                "explanation": {"type": "string"},
                                "suggested_fix": {"type": "string"},
                            },
                            "required": ["category", "severity", "explanation", "suggested_fix"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["id", "issues"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["segments"],
    "additionalProperties": False,
}


class LLMError(LocQAError):
    """The model refused, was cut off, or returned something unusable."""


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class LLM:
    """Thin wrapper around the Anthropic client: one structured-output request, usage tracking."""

    def __init__(self, client, effort: str = "medium"):
        self.client = client
        self.effort = effort
        self.usage = Usage()

    def ask(self, system: list[dict], payload: dict, schema: dict, max_tokens: int = 16000) -> dict:
        extra = {}
        if MODEL in FALLBACK_MODELS:
            extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        response = self.client.beta.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=1)}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            **extra,
        )
        u = response.usage
        cached = getattr(u, "cache_read_input_tokens", 0) or 0
        written = getattr(u, "cache_creation_input_tokens", 0) or 0
        self.usage.requests += 1
        self.usage.input_tokens += u.input_tokens + cached + written
        self.usage.cached_tokens += cached
        self.usage.output_tokens += u.output_tokens
        if response.stop_reason == "refusal":
            raise LLMError("The model declined to process this batch.")
        if response.stop_reason == "max_tokens":
            raise LLMError("The response was cut off. Try a smaller --batch-size.")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("The model returned no text.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"The model returned invalid JSON: {e}") from e


def make_batches(segments: list[Segment], max_segments: int, max_chars: int = 6000) -> list[list[Segment]]:
    """Group segments so each request stays small enough for reliable, complete answers."""
    batches: list[list[Segment]] = []
    current: list[Segment] = []
    size = 0
    for seg in segments:
        if current and (len(current) >= max_segments or size + len(seg.source) > max_chars):
            batches.append(current)
            current, size = [], 0
        current.append(seg)
        size += len(seg.source)
    if current:
        batches.append(current)
    return batches


def _glossary_table(glossary: Glossary, lang: str) -> str:
    rows = ["| Source term | Approved translation | Note |", "|---|---|---|"]
    for term in sorted(glossary.for_language(lang), key=lambda t: t.source.lower()):
        target = "DO NOT TRANSLATE (keep exactly as in source)" if term.dnt else " / ".join(term.target_for(lang))
        rows.append(f"| {term.source} | {target} | {term.note} |")
    return "\n".join(rows) if len(rows) > 2 else "(no glossary)"


def _validation_problem(seg: Segment, target: str) -> str | None:
    missing, extra = protected_diff(seg.source, target)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {' '.join(missing)}")
        if extra:
            parts.append(f"unexpected {' '.join(extra)}")
        return "Tags/placeholders do not match the source: " + "; ".join(parts) + "."
    order = tag_order_problem(target)
    return f"Tag order is wrong: {order}." if order else None


@dataclass
class TranslationOutcome:
    targets: dict[str, str] = field(default_factory=dict)
    repaired: list[str] = field(default_factory=list)   # fixed in the repair round
    unresolved: list[str] = field(default_factory=list)  # still invalid after the repair round


class Translator:
    def __init__(self, llm: LLM, glossary: Glossary, source_lang: str, target_lang: str,
                 style_guide: str = "", domain_context: str = "", batch_size: int = 15, repair_rounds: int = 1):
        self.llm = llm
        self.glossary = glossary
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.batch_size = batch_size
        self.repair_rounds = repair_rounds
        self.system = self._system(style_guide, domain_context)

    def _system(self, style_guide: str, domain_context: str) -> list[dict]:
        src, tgt = language_name(self.source_lang), language_name(self.target_lang)
        text = (
            f"You are a senior {tgt} translator specialised in action sports (surf, snowboard, mountain bike). "
            f"You translate {src} e-commerce and marketing copy into {tgt} for publication, as part of a "
            "professional localization workflow with a CAT tool, a translation memory and a term base.\n\n"
            "Rules:\n"
            "1. Translate every segment you receive. Return each id exactly once.\n"
            "2. Tokens such as ⟦1⟧, ⟦/1⟧, ⟦2⟧ are inline formatting tags. Copy each one exactly once and "
            "unchanged. Keep a pair ⟦n⟧...⟦/n⟧ around the words that correspond to the wrapped source words. "
            "Never add, drop or renumber tokens.\n"
            "3. Placeholders such as {count}, {{name}}, %s, %1$s and HTML tags such as <b>...</b> are code: copy "
            "them unchanged and never translate the names inside them.\n"
            "4. Glossary: use the approved translation for every glossary term, inflected as the grammar requires. "
            "Terms marked DO NOT TRANSLATE stay exactly as in the source. Product and model names stay in English.\n"
            "5. tm_matches are earlier human translations of similar sentences, with a match score. Reuse their "
            "wording and terminology and change only what differs in the new source (numbers, sizes, extra words).\n"
            "6. max_chars is a hard limit on visible characters (tokens do not count). Stay within it, shortening "
            "the wording if needed without losing key information.\n"
            "7. Keep every number and its value. Format numbers, units and currency for the target locale as the "
            "style guide says.\n"
            "8. Write natural, idiomatic copy that a native rider would write, following the style guide. Do not "
            "translate word for word. The segment's context note, when present, explains where the text is used.\n\n"
            f"## Domain context\n{domain_context.strip() or '(none)'}\n\n"
            f"## Style guide\n{style_guide.strip() or '(none)'}\n\n"
            f"## Glossary ({self.source_lang} -> {self.target_lang})\n{_glossary_table(self.glossary, self.target_lang)}"
        )
        # Same text for every batch of this language -> cached after the first request.
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]

    def _segment_payload(self, seg: Segment, matches: list[TMMatch]) -> dict:
        item: dict = {"id": seg.id, "source": seg.source}
        if seg.note:
            item["context"] = seg.note
        if seg.max_chars:
            item["max_chars"] = seg.max_chars
        terms = self.glossary.find_in(seg.source, self.target_lang)
        if terms:
            item["glossary"] = [{"source": t.source, "target": t.preferred(self.target_lang),
                                 "do_not_translate": t.dnt} for t in terms]
        if matches:
            item["tm_matches"] = [m.as_reference() for m in matches]
        return item

    def _request(self, items: list[dict]) -> dict[str, str]:
        payload = {"task": "translate", "source_lang": self.source_lang, "target_lang": self.target_lang,
                   "segments": items}
        result = self.llm.ask(self.system, payload, TRANSLATION_SCHEMA)
        return {t["id"]: t["target"] for t in result["translations"]}

    def translate(self, segments: list[Segment], matches: dict[str, list[TMMatch]] | None = None) -> TranslationOutcome:
        matches = matches or {}
        outcome = TranslationOutcome()
        by_id = {s.id: s for s in segments}
        for batch in make_batches(segments, self.batch_size):
            answers = self._request([self._segment_payload(s, matches.get(s.id, [])) for s in batch])
            outcome.targets.update({k: v for k, v in answers.items() if k in by_id})

        problems = self._problems(segments, outcome.targets)
        for _ in range(self.repair_rounds):
            if not problems:
                break
            items = []
            for seg_id, problem in problems.items():
                item = self._segment_payload(by_id[seg_id], matches.get(seg_id, []))
                item["previous_attempt"] = outcome.targets.get(seg_id, "")
                item["problem"] = problem
                items.append(item)
            fixed = {}
            for batch in make_batches([by_id[i] for i in problems], self.batch_size):
                ids = {s.id for s in batch}
                fixed.update(self._request([it for it in items if it["id"] in ids]))
            for seg_id in problems:
                if seg_id in fixed:
                    outcome.targets[seg_id] = fixed[seg_id]
            still = self._problems([by_id[i] for i in problems], outcome.targets)
            outcome.repaired += [i for i in problems if i not in still]
            problems = still
        outcome.unresolved = list(problems)
        return outcome

    @staticmethod
    def _problems(segments: list[Segment], targets: dict[str, str]) -> dict[str, str]:
        problems = {}
        for seg in segments:
            if seg.id not in targets:
                problems[seg.id] = "This segment was missing from your answer."
            else:
                problem = _validation_problem(seg, targets[seg.id])
                if problem:
                    problems[seg.id] = problem
        return problems


class Reviewer:
    """MQM-style review of translated segments by Claude."""

    def __init__(self, llm: LLM, glossary: Glossary, source_lang: str, target_lang: str,
                 style_guide: str = "", batch_size: int = 20):
        self.llm = llm
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.batch_size = batch_size
        tgt = language_name(target_lang)
        text = (
            f"You are a senior {tgt} reviewer (LQA) for action-sports e-commerce copy. You evaluate translations "
            f"from {language_name(source_lang)} into {tgt} with an MQM-style error typology.\n\n"
            "For each segment, list the real errors in the translation. Do not report preferential changes: if a "
            "translation is correct and natural, return an empty issues list.\n\n"
            "Categories:\n"
            "- accuracy: mistranslation, omission, addition, wrong numbers or product facts\n"
            "- fluency: grammar, spelling, punctuation, unnatural phrasing\n"
            "- terminology: glossary or industry terms used wrongly or inconsistently\n"
            "- style: tone and brand voice from the style guide, awkward or overly literal copy\n"
            "- locale_convention: number, unit, currency, date and quotation mark formats\n"
            "- markup: tag tokens (⟦n⟧) or placeholders lost, moved to the wrong words, or altered\n\n"
            "Severity:\n"
            "- minor: noticeable but does not change the meaning or hurt usability\n"
            "- major: changes the meaning or clearly hurts quality; a customer would notice\n"
            "- critical: wrong product facts, safety or legal problems, or offensive content; could cause returns or harm\n\n"
            "explanation: one short English sentence. suggested_fix: the full corrected target segment, keeping "
            "every token and placeholder.\n\n"
            f"## Style guide\n{style_guide.strip() or '(none)'}\n\n"
            f"## Glossary ({source_lang} -> {target_lang})\n{_glossary_table(glossary, target_lang)}"
        )
        self.system = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]

    def review(self, segments: list[Segment]) -> list[ReviewIssue]:
        issues: list[ReviewIssue] = []
        known = {s.id for s in segments}
        todo = [s for s in segments if s.target.strip()]
        for batch in make_batches(todo, self.batch_size):
            payload = {
                "task": "review", "source_lang": self.source_lang, "target_lang": self.target_lang,
                "segments": [{"id": s.id, "source": s.source, "target": s.target} for s in batch],
            }
            result = self.llm.ask(self.system, payload, REVIEW_SCHEMA)
            for item in result["segments"]:
                if item["id"] not in known:
                    continue
                for issue in item["issues"]:
                    issues.append(ReviewIssue(item["id"], issue["category"], issue["severity"],
                                              issue["explanation"], issue["suggested_fix"]))
        return issues
