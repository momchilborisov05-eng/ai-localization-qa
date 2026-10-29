"""
Translation memory: load TMX or CSV, look up fuzzy matches, add confirmed translations, save TMX.

Inline tags in TMX segments (<bpt>, <ept>, <ph>, <it>) are native code and are left out of the
match text. Segments stored by this tool keep the ⟦n⟧ tag tokens as plain text.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from lxml import etree

from .errors import LocQAError
from .fuzzy import fuzzy_score

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def same_language(a: str, b: str) -> bool:
    """'de', 'de-DE' and 'de_AT' all count as German."""
    return a.lower().replace("_", "-").split("-")[0] == b.lower().replace("_", "-").split("-")[0]


@dataclass(frozen=True)
class TMEntry:
    source: str
    target: str
    source_lang: str
    target_lang: str
    origin: str = "tm"  # "tm" (imported) or "feedback" (confirmed by a linguist)


@dataclass(frozen=True)
class TMMatch:
    score: int
    entry: TMEntry

    def as_reference(self) -> dict:
        return {"score": self.score, "source": self.entry.source, "target": self.entry.target}


class TranslationMemory:
    def __init__(self, entries: list[TMEntry] | None = None):
        self.entries: list[TMEntry] = list(entries or [])

    def __len__(self) -> int:
        return len(self.entries)

    # ---------- loading ----------

    @classmethod
    def load(cls, path: Path | None, source_lang: str = "en") -> "TranslationMemory":
        if path is None:
            return cls()
        path = Path(path)
        if not path.exists():
            raise LocQAError(f"Translation memory not found: {path}")
        if path.suffix.lower() == ".tmx":
            return cls(_read_tmx(path))
        if path.suffix.lower() == ".csv":
            return cls(_read_csv(path, source_lang))
        raise LocQAError(f"Unsupported TM format '{path.suffix}'. Use .tmx or .csv.")

    # ---------- lookup ----------

    def lookup(self, source: str, target_lang: str, min_score: int = 75, limit: int = 3) -> list[TMMatch]:
        """Best matches for a source segment, highest score first."""
        matches = []
        for entry in self.entries:
            if not same_language(entry.target_lang, target_lang):
                continue
            # cheap filters before the full score
            shorter, longer = sorted((len(source), len(entry.source)))
            if longer and shorter / longer < min_score / 100 - 0.25:
                continue
            if entry.source != source and SequenceMatcher(None, source, entry.source).quick_ratio() < 0.5:
                continue
            score = fuzzy_score(source, entry.source)
            if score >= min_score:
                matches.append(TMMatch(score, entry))
        matches.sort(key=lambda m: (-m.score, m.entry.origin != "feedback"))
        return matches[:limit]

    # ---------- updating ----------

    def add(self, source: str, target: str, source_lang: str, target_lang: str, origin: str = "feedback") -> str:
        """Add or update a translation. Returns 'added', 'updated' or 'unchanged'."""
        for i, entry in enumerate(self.entries):
            if entry.source == source and same_language(entry.target_lang, target_lang):
                if entry.target == target:
                    return "unchanged"
                self.entries[i] = TMEntry(source, target, source_lang, entry.target_lang, origin)
                return "updated"
        self.entries.append(TMEntry(source, target, source_lang, target_lang, origin))
        return "added"

    def save_tmx(self, path: Path) -> None:
        source_lang = self.entries[0].source_lang if self.entries else "en"
        root = etree.Element("tmx", version="1.4")
        etree.SubElement(root, "header", creationtool="locqa", creationtoolversion="1.0", segtype="sentence",
                         adminlang="en", srclang=source_lang, datatype="plaintext", **{"o-tmf": "locqa"})
        body = etree.SubElement(root, "body")
        groups: dict[tuple[str, str], list[TMEntry]] = {}
        for entry in self.entries:
            groups.setdefault((entry.source_lang, entry.source), []).append(entry)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for (src_lang, source), entries in groups.items():
            tu = etree.SubElement(body, "tu")
            if any(e.origin == "feedback" for e in entries):
                tu.set("changedate", stamp)
                etree.SubElement(tu, "prop", type="x-origin").text = "linguist-feedback"
            tuv = etree.SubElement(tu, "tuv", {XML_LANG: src_lang})
            etree.SubElement(tuv, "seg").text = source
            for entry in entries:
                tuv = etree.SubElement(tu, "tuv", {XML_LANG: entry.target_lang})
                etree.SubElement(tuv, "seg").text = entry.target
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        etree.ElementTree(root).write(str(path), xml_declaration=True, encoding="UTF-8", pretty_print=True)


def _seg_text(seg) -> str:
    parts = [seg.text or ""]
    for child in seg:
        if isinstance(child.tag, str) and etree.QName(child).localname in ("hi", "sub"):
            parts.append(_seg_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _read_tmx(path: Path) -> list[TMEntry]:
    try:
        root = etree.parse(str(path), etree.XMLParser(resolve_entities=False, no_network=True)).getroot()
    except etree.XMLSyntaxError as e:
        raise LocQAError(f"{path} is not valid TMX: {e}") from e
    header = root.find("header")
    header_lang = header.get("srclang", "") if header is not None else ""
    entries = []
    for tu in root.iter("tu"):
        variants = {}
        for tuv in tu.findall("tuv"):
            lang = tuv.get(XML_LANG) or tuv.get("lang") or ""
            seg = tuv.find("seg")
            if lang and seg is not None:
                variants[lang] = _seg_text(seg)
        source_lang = tu.get("srclang") or header_lang
        if not source_lang or source_lang == "*all*":
            source_lang = next(iter(variants), "")
        source_key = next((lang for lang in variants if same_language(lang, source_lang)), None)
        if source_key is None:
            continue
        for lang, text in variants.items():
            if lang != source_key and text.strip():
                entries.append(TMEntry(variants[source_key], text, source_key, lang))
    return entries


def _read_csv(path: Path, source_lang: str) -> list[TMEntry]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"source", "target", "target_lang"} - set(reader.fieldnames or [])
        if missing:
            raise LocQAError(f"{path}: TM CSV needs the columns source, target, target_lang (missing: {', '.join(sorted(missing))}).")
        return [
            TMEntry(row["source"], row["target"], row.get("source_lang") or source_lang, row["target_lang"])
            for row in reader
            if row["source"].strip() and row["target"].strip()
        ]
